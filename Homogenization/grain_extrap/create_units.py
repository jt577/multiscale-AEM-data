#!/usr/bin/env python3
import os, math, torch, matplotlib.pyplot as plt
import torch.nn.functional as F

# ─── user params ────────────────────────────────────────────────────
morph_list = ['lamellae', 'gyroid']
units_list = [2,3,4,5]
suff = '_coarse'

# ────────────────────────────────────────────────────────────────────────────
for morph in morph_list:
    for units in units_list:
        # new_N = 16  # number of grid points per unit per dimension
        small_field_path = f'./fields/{morph}_1u1t{suff}.pt'
        output_field_path = f'./fields/{morph}_{units}u1t{suff}.pt'
        plot_phase, plot_dir = 1, f'./plots/{morph}_{units}u1t{suff}'
        ell = 1.0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        def one_hot_field(maskB):
            """
            Build a (2, N, N, N) tensor where channel 1 is maskB and channel 0 is the complement.
            """
            N = maskB.shape[-1]
            field = torch.zeros((2, N, N, N), device=maskB.device)
            field[1] = maskB.float()
            field[0] = 1.0 - field[1]
            return field

        def downsample_field(field, new_N, mode='trilinear', align_corners=True):
            """
            Resample a field tensor of shape (C, N, N, N) → (C, new_N, new_N, new_N).
            - mode='trilinear' for smooth densities
            - mode='nearest' for strict one-hot fields
            """
            # field: (C,N,N,N) → (1,C,N,N,N)
            x = field.unsqueeze(0)
            if mode == 'nearest':
                y = F.interpolate(x, size=(new_N,)*3, mode=mode)
            else:
                y = F.interpolate(x, size=(new_N,)*3, mode=mode, align_corners=align_corners)
            return y.squeeze(0)    # back to (C,new_N,new_N,new_N)

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
        # ────────────────────────────────────────────────────────────────────────────


        rho_small = torch.load(small_field_path, map_location='cpu')
        rho_small = torch.stack(rho_small, dim=0).to(device) if isinstance(rho_small, list) else rho_small
        rho_small_new = torch.zeros((2, rho_small.shape[-1], rho_small.shape[-1], rho_small.shape[-1]), device=device, dtype=torch.float64)
        if rho_small.shape[0] == 3:
            rho_small_new[0] = rho_small[0] + rho_small[1]
            rho_small_new[1] = rho_small[2]
            rho_small = rho_small_new

        assert rho_small.shape[0] == 2 and rho_small.ndim == 4, \
            'Expect (2,N,N,N) one-hot input'

        # # if your rho_small is one-hot and you want to preserve hard phases:
        # rho_small = downsample_field(rho_small, new_N, mode='nearest')
        # # otherwise, for smooth density use trilinear:
        # rho_small = downsample_field(rho_small, new_N, mode='trilinear')

        rho_small = one_hot_field(rho_small[1,...]).to(torch.float64)

        Nbig = rho_small.shape[-1] * units
        rho_big = torch.zeros((2, Nbig, Nbig, Nbig), device=device, dtype=torch.float64)
        for i in range(units):
            for j in range(units):
                for k in range(units):
                    low_i = i * rho_small.shape[-1]
                    hi_i = (i + 1) * rho_small.shape[-1]
                    low_j = j * rho_small.shape[-1]
                    hi_j = (j + 1) * rho_small.shape[-1]
                    low_k = k * rho_small.shape[-1]
                    hi_k = (k + 1) * rho_small.shape[-1]

                    rho_big[:, low_i:hi_i, low_j:hi_j, low_k:hi_k] = rho_small

        torch.save(rho_big.cpu(), output_field_path)
        print('image field saved →', output_field_path)

        plot_mid_slices(rho_big.cpu(), phase=plot_phase, out_dir=plot_dir)
        print('slice PNGs in', plot_dir)

