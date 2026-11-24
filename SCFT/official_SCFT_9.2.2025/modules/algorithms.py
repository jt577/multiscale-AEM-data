import torch
import math
import numpy as np
from scipy.signal import savgol_filter

def simpson(f: torch.Tensor, a: float, b: float, N: int, dim: int) -> torch.Tensor:
    """
    Simpson's rule over a specified dimension of f.

    Args:
      f   : Tensor of rank ≥1, with f.shape[dim] == N+1
      a,b : integration limits
      N   : number of intervals (must be even)
      dim : the axis in f to integrate over

    Returns:
      Tensor of rank f.ndim-1 with that dimension integrated out.
    """
    if N % 2 != 0:
        raise ValueError("Number of intervals N must be even for Simpson's rule.")
    if f.shape[dim] != N+1:
        raise ValueError(f"Size of dim={dim} must be N+1={N+1}, got {f.shape[dim]}")

    h = (b - a) / N

    # build Simpson coefficients [1,4,2,4,…,2,4,1]
    coeff = f.new_ones(N+1)
    coeff[1:N:2] = 4.0
    coeff[2:N-1:2] = 2.0

    # reshape coeff so it broadcasts along `dim`
    shape = [1] * f.ndim
    shape[dim] = N+1
    coeff = coeff.view(shape)

    # weighted sum and final factor
    weighted = coeff * f
    summed   = torch.sum(weighted, dim=dim)
    return (h/3.0) * summed

def Picard(SOL, POLY, GRID, DIR, func, tol_SCFT=1e-2, tol_UC=1e-2, maxIt=1000, beta=0.1):
    """
    Solves the self consistent relation x = f(x) using mixed damped Picard iteration and Pulay mixing.
    """
    DIR.log("Picard to solve nonlinear equation")
    for k in range(maxIt):
        SOL.iter = k

        # Update iterate
        Fx = func(SOL, POLY, GRID, DIR)
        g = Fx - SOL.x
        SOL.x += beta * g

        err_SCFT_f = torch.sqrt(torch.sum(g**2) / torch.sum(SOL.x**2)).item()
        # Update unit cell parameters
        err_UC_f = GRID.update_ucparams(SOL)
        # Print iteration info
        freeE_f     = SOL.freeEn.item()                
        sanity_f    = SOL.sanity.item()                
        DIR.log(
            f"Iteration {k}: SCFT error={err_SCFT_f:.5e}   UC error={err_UC_f:.5e}   UC lr={GRID.Lattice_stepsize:.3e}   free energy={freeE_f:.5e}   sanity={sanity_f:.5e}"
        )
        # Dump iteration data
        DIR.dumpIteration(k, SOL, POLY, GRID)
        if err_SCFT_f < tol_SCFT and err_UC_f < tol_UC:
            DIR.log(f"SCFT error converged within tol {tol_SCFT} and UC error converged within tol {tol_UC} after {k} iterations.")
            break
        elif math.isnan(freeE_f):
            DIR.log(f"WARNING: NaN detected. Stopping simulation.")
            break
        
        # Adaptive step size for unit cell parameters (ensures errors of SCFT and UC are similar)
        GRID.Lattice_stepsize = GRID.Lattice_stepsize * math.exp(0.1 * (err_UC_f - err_SCFT_f) / err_SCFT_f)
        if GRID.Lattice_stepsize < 1e-5 or GRID.Lattice_stepsize > 1e1:
            GRID.Lattice_stepsize = 1e-5 if GRID.Lattice_stepsize < 1e-5 else 1e1

    if k == maxIt-1:
        DIR.log(f"WARNING: Picard did not converge after {maxIt} iterations.")