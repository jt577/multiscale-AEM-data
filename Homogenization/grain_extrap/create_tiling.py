#!/usr/bin/env python3
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Choose device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def mosaic_random_rotations(rho_small, tiles):
    """
    Tile a (C, N, N, N) one-hot field into (C, tiles*N, tiles*N, tiles*N),
    applying an independent random 3D rotation to each tile.
    """
    C, N, _, _ = rho_small.shape
    out = torch.zeros((C, tiles*N, tiles*N, tiles*N),
                      dtype=rho_small.dtype, device=device)

    # reference grid in [-1,1]^3
    lin = torch.linspace(-1, 1, N, device=device, dtype=rho_small.dtype)
    base_grid = torch.stack(torch.meshgrid(lin, lin, lin, indexing='ij'),
                            dim=-1).unsqueeze(0)  # shape: (1, N, N, N, 3)

    rho_in = rho_small.unsqueeze(0)  # (1, C, N, N, N)

    for i in range(tiles):
        for j in range(tiles):
            for k in range(tiles):
                # 1) random unit quaternion
                q = torch.randn(4, device=device, dtype=rho_small.dtype)
                q /= q.norm()

                w, x, y, z = q
                # 2) build 3×3 rotation matrix
                R = torch.tensor([
                    [1-2*(y*y+z*z),   2*(x*y - w*z), 2*(x*z + w*y)],
                    [2*(x*y + w*z), 1-2*(x*x+z*z),   2*(y*z - w*x)],
                    [2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x+y*y)]
                ], device=device, dtype=rho_small.dtype)

                # 3) rotate the grid
                rot = base_grid @ R           # still in [-1,1]

                # 3b) wrap into [-1,1] again so everything samples periodically
                #     (x+1) mod 2 → [0,2), then -1 → [-1,1)
                rot_wrapped = torch.remainder(rot + 1, 2.0) - 1.0

                # 4) now sample — no border-smearing because all coords are in bounds
                vox = F.grid_sample(
                    rho_in,
                    rot_wrapped,
                    mode='nearest',
                    padding_mode='border',  # now never actually used
                    align_corners=True
                )[0]


                # 5) place into the big cube
                out[:, i*N:(i+1)*N, j*N:(j+1)*N, k*N:(k+1)*N] = vox

    return out

def plot_mid_slices(rho_big, phase=0, out_dir='plots'):
    os.makedirs(out_dir, exist_ok=True)
    _, D, _, _ = rho_big.shape
    x0 = y0 = z0 = D // 2
    vol = rho_big[phase]
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

morph_list   = ['lamellae', 'gyroid']
units_list   = [1]
tiles_list   = [2]
suff         = '_coarse'
num_samples  = 1
torch.manual_seed(42)  # for reproducibility
for morph in morph_list:
    for num_units in units_list:
        for tiles_per_dim in tiles_list:
            for s in range(num_samples):
                # load small cell (expects a (C,N,N,N) tensor or list-of-2 tensors)
                p = f'./fields/{morph}_{num_units}u1t{suff}.pt'
                small = torch.load(p, map_location='cpu')
                if isinstance(small, list):
                    rho_small = torch.stack(small, dim=0)
                else:
                    rho_small = small
                rho_small = rho_small.to(device).float()

                # enforce two-phase one-hot if needed
                if rho_small.shape[0] == 3:
                    tmp = torch.zeros((2, *rho_small.shape[1:]),
                                        dtype=rho_small.dtype, device=device)
                    tmp[0] = rho_small[0] + rho_small[1]
                    tmp[1] = rho_small[2]
                    rho_small = tmp

                # generate and save
                rho_big = mosaic_random_rotations(rho_small, tiles_per_dim)
                out_path = f'./fields/{morph}_{num_units}u{tiles_per_dim}t{suff}_s{s}.pt'
                torch.save(rho_big.cpu(), out_path)
                print('Saved:', out_path)

                plot_mid_slices(rho_big, phase=1, out_dir=f'./plots/{morph}_{num_units}u{tiles_per_dim}t{suff}_s{s}')