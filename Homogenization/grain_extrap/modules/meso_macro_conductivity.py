import torch, os
import numpy as np
from modules.conductivity import calculate_effective_conductivity
from modules.bulk_shear import calculate_effective_bulk_and_shear
from scipy.optimize import brentq

def meso_conductivity(rhos, conductivities, outfile):
    """
    Calculates mesoscopic conductivity and elasticity tensors by solving the continuity equation and mechanical equilibrium.
    """
    with open(outfile, 'a') as file:
        print('\n------------------------------------', file=file)
        print('Calculating mesoscopic conductivity tensor...\n', file=file)

    # Build the conductivity field by summing each phase: shape [nx, ny, nz]
    conductivity_list = []
    for i in range(rhos.shape[0]):
        conductivity_list.append(rhos[i, ...] * conductivities[i])
    micro_sigma = sum(conductivity_list)

    # Calculate effective conductivity
    meso_sigma = calculate_effective_conductivity(micro_sigma, outfile)

    with open(outfile, 'a') as file:
        print('\n------------------------------------', file=file)
        print('Mesoscopic sigma:\n', file=file)
        print(meso_sigma.cpu().numpy(), file=file)
        print('\n------------------------------------', file=file)

    return meso_sigma


def macro_conductivity(meso_sigma, outfile):
    """
    Calculates macroscopic conductivity and elasticity tensors using the self-consistent effective medium approximation.
    """
    with open(outfile, 'a') as file:
        print('\n------------------------------------', file=file)
        print('Calculating macroscopic conductivity...\n', file=file)
    
    meso_sigma = meso_sigma.cpu().numpy()

    # Conductivity
    def objfun(x, sigma):
        # 3x3 Identity
        δ = np.eye(3)
        # Assemble matrices
        M = sigma + 2 * x * δ
        N = sigma @ np.linalg.inv(M)
        # Objective function
        f = np.linalg.trace(N) - 1
        return f
    
    with open(outfile, 'a') as file:
        print("Brent to solve nonlinear equation.", file=file)
    
    macro_iso_sigma, r = brentq(f=objfun, a=0.0, b=1.0, args=(meso_sigma,), xtol=1e-10, maxiter=100, disp=True, full_output=True)
    
    if r.converged == False:
        with open(outfile, 'a') as file:
            print("Warning: Brentq did not converge!", file=file)
    else:
        with open(outfile, 'a') as file:
            print("Brentq converged", file=file)

    with open(outfile, 'a') as file:        
        print(f"\nEMA sigma: {macro_iso_sigma}", file=file)
    
    # Calculate Voigt, Reuss, and Hill
    sigma_V = 1/3 * np.einsum('ii->', meso_sigma)
    sigma_R = (1/3 * np.einsum('ii->', np.linalg.inv(meso_sigma)))**(-1)
    sigma_H = 1/2 * (sigma_V + sigma_R)
    with open(outfile, 'a') as f:
        f.write(f"Voigt sigma: {sigma_V}\n")
        f.write(f"Reuss sigma: {sigma_R}\n")
        f.write(f"Hill sigma: {sigma_H}\n")
        
    return macro_iso_sigma