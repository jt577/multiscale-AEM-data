# modules/conductivity.py

import os
import math
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from modules.algorithms import PCG

torch.set_default_dtype(torch.float32)


def _build_k_grids(N, L, device, dtype=torch.float32):
    dx, dy, dz = (L / N)
    kx = 2*math.pi * torch.fft.fftfreq(N[0], d=dx, device=device, dtype=dtype)
    ky = 2*math.pi * torch.fft.fftfreq(N[1], d=dy, device=device, dtype=dtype)
    kz = 2*math.pi * torch.fft.rfftfreq(N[2], d=dz, device=device, dtype=dtype)
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    return KX, KY, KZ, KX**2 + KY**2 + KZ**2, dx


def jac_obs_electrostatic(x, x_shape, sigma, KX, KY, KZ, E, cond_comp, dx):
    phi = x.reshape(x_shape)
    phi_hat = torch.fft.rfftn(phi, dim=(0,1,2))
    dphi_hat = torch.stack([-1j*KX*phi_hat, -1j*KY*phi_hat, -1j*KZ*phi_hat], dim=0)
    dphi = torch.fft.irfftn(dphi_hat, s=phi.shape, dim=(1,2,3))

    # build J depending on direction
    Jx = sigma * (E - dphi[0]) if cond_comp=='xx' else sigma * (-dphi[0])
    Jy = sigma * (E - dphi[1]) if cond_comp=='yy' else sigma * (-dphi[1])
    Jz = sigma * (E - dphi[2]) if cond_comp=='zz' else sigma * (-dphi[2])

    eff = torch.zeros((3,3), dtype=torch.float32, device=sigma.device)
    idx = {'xx':0,'yy':1,'zz':2}[cond_comp]
    eff[idx,0] = torch.mean(Jx/E)
    eff[idx,1] = torch.mean(Jy/E)
    eff[idx,2] = torch.mean(Jz/E)

    div_J = dx**2 * torch.fft.irfftn(
        -1j*(KX*torch.fft.rfftn(Jx, dim=(0,1,2)) +
             KY*torch.fft.rfftn(Jy, dim=(0,1,2)) +
             KZ*torch.fft.rfftn(Jz, dim=(0,1,2))),
        s=phi.shape, dim=(0,1,2))

    return div_J.flatten(), eff


def _make_preconditioner(N, K2, dx):
    denom = dx**2 * (K2 + 1.0)
    def apply_precond(r_flat):
        r = r_flat.view(N[0],N[1],N[2])
        z_hat = torch.fft.rfftn(r, dim=(0,1,2)) / denom
        z = torch.real(torch.fft.irfftn(z_hat, s=(N[0],N[1],N[2]), dim=(0,1,2)))
        return z.reshape(-1)
    return apply_precond


def calculate_effective_conductivity(sigma, outfile, parallel=True):
    """
    sigma    : 3D torch.Tensor on CPU
    outfile  : path to append logs to
    parallel : if True, on CPU will spawn 3 threads
    """
    # pick device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prep data
    sigma = sigma.to(torch.float32).to(device)
    N     = torch.tensor(sigma.shape, device=device)
    L     = N / torch.min(N)
    KX, KY, KZ, K2, dx = _build_k_grids(N, L, device)
    apply_precond = _make_preconditioner(N, K2, dx)
    phi0 = torch.randn(*sigma.shape, dtype=torch.float32, device=device)

    directions = ['xx','yy','zz']
    results = {}

    def _solve_one(cond_comp):
        with open(outfile, 'a') as f:
            print(f"[{cond_comp}] starting PCG...", file=f)

        E  = 1.0
        x0 = phi0.flatten()
        phi = PCG(x0,
                  args=(phi0.shape, sigma, KX, KY, KZ, E, cond_comp, dx),
                  jac_obs=jac_obs_electrostatic,
                  apply_precond=apply_precond,
                  outfile=outfile).reshape(phi0.shape)

        _, eff = jac_obs_electrostatic(phi, phi.shape,
                                       sigma, KX, KY, KZ, E, cond_comp, dx)
        with open(outfile, 'a') as f:
            print(f"[{cond_comp}] done.", file=f)
        return cond_comp, eff

    if device.type=='cpu' and parallel:
        # optionally limit intra-op threads to avoid oversubscription
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        with ThreadPoolExecutor(max_workers=3) as exec:
            futures = { exec.submit(_solve_one, d): d for d in directions }
            for fut in as_completed(futures):
                comp, eff = fut.result()
                results[comp] = eff
    else:
        # serial on GPU (or if parallel=False)
        for d in directions:
            comp, eff = _solve_one(d)
            results[comp] = eff

    # assemble and symmetrize
    C = results['xx'] + results['yy'] + results['zz']
    C = 0.5*(C + C.T)

    return C.cpu()
