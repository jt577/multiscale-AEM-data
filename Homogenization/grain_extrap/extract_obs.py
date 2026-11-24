import os
import re
import numpy as np

# --- main script -----------------------------------------------------------

units_l = [1,2,3,4,5]
tiles_l = [1,2,3,4,5]
samples_l = [0,1,2,3,4]

units_g = [1,2,3,4,5]
tiles_g = [1,2,3,4,5]
samples_g = [0,1,2,3,4]

# Voigt
cvl = np.zeros((len(units_l), len(tiles_l), len(samples_l))) 
cvg   = np.zeros((len(units_g), len(tiles_g), len(samples_g)))
evl = np.zeros((len(units_l), len(tiles_l), len(samples_l)))
evg   = np.zeros((len(units_g), len(tiles_g), len(samples_g)))

# Lamellar case
for i, unit in enumerate(units_l):
    for j, tile in enumerate(tiles_l):
        for s, sample in enumerate(samples_l):
            fname_lam = f'./outputs/lamellae_{unit}u{tile}t_coarse_s{sample}_conductivity.txt'
            if os.path.exists(fname_lam):
                try:
                    with open(fname_lam, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        if 'Voigt sigma' in line:
                            cvl[i, j, s] = float(line.strip().split()[2])
                except ValueError as e:
                    print(f"Warning (lamellar {unit},{tile}): {e}")
            
            fname_lam = f'./outputs/lamellae_{unit}u{tile}t_coarse_s{sample}_elasticity.txt'
            if os.path.exists(fname_lam):
                try:
                    with open(fname_lam, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        if 'Voigt' in line:
                            evl[i, j, s] = float(line.strip().split()[4])
                except ValueError as e:
                    print(f"Warning (lamellar {unit},{tile}): {e}")

# Gyroid case
for i, unit in enumerate(units_g):
    for j, tile in enumerate(tiles_g):
        for s, sample in enumerate(samples_g):
            fname_lam = f'./outputs/gyroid_{unit}u{tile}t_coarse_s{sample}_conductivity.txt'
            if os.path.exists(fname_lam):
                try:
                    with open(fname_lam, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        if 'Voigt sigma' in line:
                            cvg[i, j, s] = float(line.strip().split()[2])
                except ValueError as e:
                    print(f"Warning (gyroid {unit},{tile}): {e}")

            fname_lam = f'./outputs/gyroid_{unit}u{tile}t_coarse_s{sample}_elasticity.txt'
            if os.path.exists(fname_lam):
                try:
                    with open(fname_lam, 'r') as f:
                        lines = f.readlines()
                    for line in lines:
                        if 'Voigt' in line:
                            evg[i, j, s] = float(line.strip().split()[4])
                except ValueError as e:
                    print(f"Warning (gyroid {unit},{tile}): {e}")
# Save results
np.save('cvl.npy', cvl)
np.save('cvg.npy',   cvg)
np.save('evl.npy', evl)
np.save('evg.npy',   evg)

print("Done. Arrays saved.")
