import numpy as np
import torch
import math
from scipy.ndimage import distance_transform_edt
import os
import matplotlib.pyplot as plt
# Create device objects for each GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def lamellar_field(N, vol_frac, axis='x', device='cpu'):
    """
    Simple 1D modulation → lamellae.  
    Phase B occupies the region 0 ≤ coord < vol_frac along chosen axis.
    """
    coords = torch.linspace(0, 1, N, device=device)
    if axis == 'x':
        coord = coords.view(N, 1, 1).expand(N, N, N)
    elif axis == 'y':
        coord = coords.view(1, N, 1).expand(N, N, N)
    else:  # 'z'
        coord = coords.view(1, 1, N).expand(N, N, N)
    maskB = coord < vol_frac
    return maskB

def gyroid_field(N, vol_frac, device='cpu'):
    """
    Level-set of the Ia-3d (double gyroid) function:
    f(x,y,z) = sin(2πx)cos(2πy) + sin(2πy)cos(2πz) + sin(2πz)cos(2πx).
    Thresholded to achieve exact vol_frac.
    """
    coords = torch.linspace(0, 1, N, device=device)
    X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
    f = (torch.sin(2*math.pi*X)*torch.cos(2*math.pi*Y)
       + torch.sin(2*math.pi*Y)*torch.cos(2*math.pi*Z)
       + torch.sin(2*math.pi*Z)*torch.cos(2*math.pi*X))
    flat = f.flatten()
    k = int(round(vol_frac * flat.numel()))
    # find cutoff so that exactly k points satisfy f > cutoff
    cutoff = torch.topk(flat, k, largest=True).values.min()
    maskB = f > cutoff
    return maskB

def one_hot_field(maskB: torch.Tensor,
                  smooth_width: float = 0.0,
                  n_iters: int = 12) -> torch.Tensor:
    """
    Build a (2, N, N, N) tensor where channel 1 is phase-B, channel 0 its complement.
    If smooth_width > 0, do a linear ramp over ±smooth_width grid-points
    and shift it so that mean(field[1]) == original vol_frac.
    This version pads the mask periodically so that interfaces at the cell edges
    get smoothed as well.
    """
    vol_frac = float(maskB.float().mean().item())
    N = maskB.shape[0]

    # one-hot if no smoothing requested
    if smooth_width <= 0:
        mask = maskB.bool()

        # create a float64 field directly
        field = torch.zeros((2, *mask.shape),
                            dtype=torch.float64,
                            device=mask.device)

        # assign channel 1 as float64
        field[1] = mask.to(torch.float64)   # now field[1].dtype == torch.float64

        # safe to subtract
        field[0] = 1.0 - field[1]

        return field

    # 1) tile mask 3×3×3 so boundaries become interior
    m = maskB.cpu().numpy().astype(bool)
    M = np.tile(m, (3, 3, 3))

    # 2) distance transforms on the big array
    dt_in  = distance_transform_edt(M)
    dt_out = distance_transform_edt(~M)

    # 3) signed distance on the big array
    D = dt_in - dt_out

    # 4) crop out the central N×N×N block
    start, end = N, 2*N
    d = torch.from_numpy(D[start:end, start:end, start:end]).to(maskB.device).float()

    # 5) binary‐search for the offset d0 to match the volume fraction
    lo, hi = -smooth_width, smooth_width
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        f   = torch.clamp(0.5*(d - mid)/smooth_width + 0.5, 0.0, 1.0)
        if float(f.mean().item()) > vol_frac:
            lo = mid
        else:
            hi = mid
    d0 = 0.5 * (lo + hi)

    # 6) final smooth field
    f = torch.clamp(0.5*(d - d0)/smooth_width + 0.5, 0.0, 1.0)

    # pack into two channels
    field = torch.zeros((2, N, N, N), device=maskB.device, dtype=f.dtype)
    field[1] = f
    field[0] = 1.0 - f
    return field

def plot_mid_slices(rho, phase=1, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    _, D, _, _ = rho.shape
    x0 = y0 = z0 = D // 2
    vol = rho[phase]
    views = {
        f'XY_z{z0}': vol[:, :, z0],
        f'XZ_y{y0}': vol[:, y0, :],
        f'YZ_x{x0}': vol[x0, :, :],
    }
    for nm, img in views.items():
        plt.figure()
        plt.imshow(img.cpu(), origin='lower')
        plt.title(nm)
        plt.colorbar()
        plt.savefig(os.path.join(out_dir, f'{nm}.png'))
        plt.close()

# coarse is 18, fine is 36, finer is 54, and finest is 72
grid_res = 'finest'

if grid_res == 'coarse':
    N = 18
elif grid_res == 'fine':
    N = 36
elif grid_res == 'finer':
    N = 54
elif grid_res == 'finest':
    N = 72

volB    = 1/3 # fix volume fraction of conducting phase to be 1/3 to match the MacBeth (2025) paper
device  = "cuda" if torch.cuda.is_available() else "cpu"

smooth_width = 0
# Lamellar
mask = lamellar_field(N, volB, axis='x', device=device)
lamellae = one_hot_field(mask, smooth_width=smooth_width).to(torch.float64)

# Gyroid
mask = gyroid_field(N, volB, device=device)
gyroid  = one_hot_field(mask, smooth_width=smooth_width).to(torch.float64)

print("Shapes:", lamellae.shape, gyroid.shape)
print("Vol fractions:", 
        lamellae[1].mean().item(), 
        gyroid[1].mean().item())

plot_mid_slices(lamellae, phase=1, out_dir=f'./plots/lamellae_1u1t_{grid_res}')
plot_mid_slices(gyroid, phase=1, out_dir=f'./plots/gyroid_1u1t_{grid_res}')

torch.save(lamellae, f'fields/lamellae_1u1t_{grid_res}.pt')
torch.save(gyroid, f'fields/gyroid_1u1t_{grid_res}.pt')