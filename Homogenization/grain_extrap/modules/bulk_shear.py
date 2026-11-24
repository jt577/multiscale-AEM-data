import os, math, numpy as np, torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.algorithms import PCG

torch.set_default_dtype(torch.float32)

def get_strain(u, bulk, shear, KX, KY, KZ, E):
    u_stack = u.permute(3,0,1,2)                           # [3,Nx,Ny,Nz]
    u_hat   = torch.fft.rfftn(u_stack, dim=(1,2,3))
    d_ux_hat = -1j * KX * u_hat
    d_uy_hat = -1j * KY * u_hat
    d_uz_hat = -1j * KZ * u_hat

    d_all_hat = torch.stack([d_ux_hat, d_uy_hat, d_uz_hat], dim=0)   # [3,3,Nx,Ny,Nz//2+1]
    d_all     = torch.fft.irfftn(d_all_hat, s=u.shape[:3], dim=(2,3,4))

    strain = torch.zeros((3,3,*u.shape[:3]), dtype=torch.float32, device=bulk.device)
    # fill Voigt‐style
    strain[0,0] = E[0,0] + d_all[0,0]
    strain[1,1] = E[1,1] + d_all[1,1]
    strain[2,2] = E[2,2] + d_all[2,2]
    for i,j in ((0,1),(0,2),(1,2)):
        strain[i,j] = E[i,j] + 0.5*(d_all[i,j] + d_all[j,i])
        strain[j,i] = strain[i,j]
    return strain


def get_stress(u, bulk, shear, KX, KY, KZ, E):
    strain = get_strain(u, bulk, shear, KX, KY, KZ, E)
    Id = torch.eye(3, device=u.device)
    tr = torch.einsum('kkxyz->xyz', strain)
    return ((bulk - 2/3*shear) * torch.einsum('ij,xyz->ijxyz', Id, tr)
            + shear*(strain + strain.permute(1,0,2,3,4)))


def jac_obs_mechanical(x, x_shape, bulk, shear, KX, KY, KZ, E, dx):
    u      = x.reshape(x_shape)
    stress = get_stress(u, bulk, shear, KX, KY, KZ, E)
    shat   = torch.fft.rfftn(stress, dim=(2,3,4))
    div_hat = torch.stack([
        -1j*(KX*shat[0,0] + KY*shat[0,1] + KZ*shat[0,2]),
        -1j*(KX*shat[1,0] + KY*shat[1,1] + KZ*shat[1,2]),
        -1j*(KX*shat[2,0] + KY*shat[2,1] + KZ*shat[2,2]),
    ], dim=0)                        # [3,Nx,Ny,Nz//2+1]
    div = torch.fft.irfftn(div_hat, s=u.shape[:3], dim=(1,2,3))  # [3,Nx,Ny,Nz]
    grad = (dx**2)*(-1) * div.permute(1,2,3,0).reshape(-1)
    mean_stress = torch.mean(stress, dim=(2,3,4))

    # build 4th-order C for this E
    C = torch.zeros((3,3,3,3), dtype=torch.float32, device=u.device)
    # off-diagonals
    if E[0,1]==1:
        C[:, :, 0,1] = mean_stress/2; C[:,:,1,0] = mean_stress/2
    elif E[0,2]==1:
        C[:, :, 0,2] = mean_stress/2; C[:,:,2,0] = mean_stress/2
    elif E[1,2]==1:
        C[:, :, 1,2] = mean_stress/2; C[:,:,2,1] = mean_stress/2
    # diagonals
    elif E[0,0]==1:
        C[:,:,0,0] = mean_stress
    elif E[1,1]==1:
        C[:,:,1,1] = mean_stress
    elif E[2,2]==1:
        C[:,:,2,2] = mean_stress

    return grad, C


def calculate_effective_bulk_and_shear(bulk, shear, outfile, parallel=True):
    """
    bulk   : Tensor [Nx,Ny,Nz]
    shear  : Tensor [Nx,Ny,Nz]
    outfile: path for logs
    parallel: if True, on CPU uses 6 threads
    """
    # pick device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prep
    bulk  = bulk.to(torch.float32).to(device)
    shear = shear.to(torch.float32).to(device)
    N = torch.tensor(bulk.shape, device=device)
    L = N / torch.min(N)
    dx,dy,dz = L[0]/N[0], L[1]/N[1], L[2]/N[2]

    # wave-numbers
    kx = 2*math.pi*torch.fft.fftfreq(N[0], d=dx, device=device)
    ky = 2*math.pi*torch.fft.fftfreq(N[1], d=dy, device=device)
    kz = 2*math.pi*torch.fft.rfftfreq(N[2], d=dz, device=device)
    KX,KY,KZ = torch.meshgrid(kx,ky,kz, indexing='ij')

    # preconditioner M_inv
    K2 = KX**2 + KY**2 + KZ**2
    M = torch.zeros((N[0],N[1],kz.size(0),3,3), device=device)
    M[...,0,0] = 4/3*KX**2 + K2 + 1;  M[...,1,1] = 4/3*KY**2 + K2 + 1;  M[...,2,2] = 4/3*KZ**2 + K2 + 1
    M[...,0,1] = M[...,1,0] = 4/3*KX*KY
    M[...,0,2] = M[...,2,0] = 4/3*KX*KZ
    M[...,1,2] = M[...,2,1] = 4/3*KY*KZ
    M_inv = torch.linalg.inv((dx**2)*M).to(torch.complex64)

    def apply_precond(r_flat):
        r = r_flat.view(*bulk.shape,3)
        r_hat = torch.fft.rfftn(r, dim=(0,1,2))
        z_hat = torch.einsum('...ij,...j->...i', M_inv, r_hat)
        z = torch.fft.irfftn(z_hat, s=tuple(bulk.shape), dim=(0,1,2))
        return z.reshape(-1)

    # initial guess
    u0 = torch.randn(*bulk.shape,3, dtype=torch.float32, device=device)

    comps = ['xy','xz','yz','xx','yy','zz']
    results = {}

    def _solve(cmp):
        # build E
        E = torch.zeros((3,3), dtype=torch.float32, device=device)
        if cmp=='xy': E[0,1]=E[1,0]=1
        elif cmp=='xz': E[0,2]=E[2,0]=1
        elif cmp=='yz': E[1,2]=E[2,1]=1
        elif cmp=='xx': E[0,0]=1
        elif cmp=='yy': E[1,1]=1
        elif cmp=='zz': E[2,2]=1

        with open(outfile,'a') as f:
            print(f'[{cmp}] starting PCG...', file=f)

        x0 = u0.flatten()
        u = PCG(x0,
                args=(u0.shape, bulk, shear, KX, KY, KZ, E, dx),
                jac_obs=jac_obs_mechanical,
                apply_precond=apply_precond,
                outfile=outfile).reshape(u0.shape)

        Cc = jac_obs_mechanical(u, u.shape, bulk, shear, KX, KY, KZ, E, dx)[1]
        with open(outfile,'a') as f:
            print(f'[{cmp}] done.', file=f)
        return cmp, Cc

    if device.type=='cpu' and parallel:
        # Optionally cap BLAS threads per PyTorch thread here:
        torch.set_num_threads(1); torch.set_num_interop_threads(1)

        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_solve, c):c for c in comps}
            for f in as_completed(futs):
                cmp, Cc = f.result()
                results[cmp] = Cc
    else:
        for c in comps:
            cmp, Cc = _solve(c)
            results[cmp] = Cc

    # assemble & symmetrize
    C = sum(results.values())
    C = 0.25*(C
              + C.permute(1,0,2,3)
              + C.permute(0,1,3,2)
              + C.permute(1,0,3,2))

    return C.cpu()
