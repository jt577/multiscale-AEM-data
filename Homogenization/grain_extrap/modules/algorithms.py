import torch
import math
import numpy as np

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

def Picard(SOL, POLY, GRID, DIR, func, tol=1e-2, maxIt=1000, beta=0.1):
    """
    Solves the self consistent relation x = f(x) using mixed damped Picard iteration and Pulay mixing.
    """
    DIR.log("Picard to solve nonlinear equation")
    for k in range(maxIt):
        SOL.iter = k
        # Update the iterate
        xNew = func(SOL, POLY, GRID, DIR)
        g = xNew - SOL.x
        err = torch.sqrt(torch.sum(g**2) / torch.sum(SOL.x**2))
        SOL.x += beta * g
        # Update unit cell parameters
        if GRID.fix == 'none':
            GRID.update_ucparams(GRID.ucparams - GRID.Lattice_stepsize * SOL.dfdu)
            dfdu_norm_f = torch.linalg.norm(SOL.dfdu).item()
        elif GRID.fix == 'cubic':
            new_ucparams = GRID.ucparams.clone()
            dfdl = torch.sum(SOL.dfdu[0:3])
            new_ucparams[0:3] = GRID.ucparams[0:3] - GRID.Lattice_stepsize * dfdl
            GRID.update_ucparams(new_ucparams)
            dfdu_norm_f = torch.linalg.norm(dfdl).item()
        elif GRID.fix == 'angles':
            new_ucparams = GRID.ucparams.clone()
            new_ucparams[0:3] = GRID.ucparams[0:3] - GRID.Lattice_stepsize * SOL.dfdu[0:3]
            GRID.update_ucparams(new_ucparams)
            dfdu_norm_f = torch.linalg.norm(SOL.dfdu[0:3]).item()
        elif GRID.fix == 'Lattice':
            dfdu_norm_f = 0.0
        # Print iteration info
        err_f       = err.item()                       
        freeE_f     = SOL.freeEn.item()                
        sanity_f    = SOL.sanity.item()                
        DIR.log(
            f"Picard iteration {k}:\n"
            f"  SCFT error={err_f:.5e}\n"
            f"  |dfdu|={dfdu_norm_f:.5e}\n"
            f"  Lattice=\n{GRID.Lattice}\n"
            f"  free energy={freeE_f:.5e}\n"
            f"  sanity={sanity_f:.5e}\n"
        )
        if k % SOL.dumpFreq == 0:
            DIR.log("\nDumping plots and SOL...\n")
            DIR.dump(SOL, POLY, GRID)
        if err < tol and dfdu_norm_f < tol:
            DIR.log(f"SCFT error and |dfdu| converged within tol {tol} after {k} iterations.")
            return SOL.x
    DIR.log(f"Picard did not converge after {maxIt} iterations.")
    return SOL.x

def PCG(x0, args, jac_obs, apply_precond,
        maxit=1000, tol=0.5, outfile=None):
    """
    Preconditioned Conjugate Gradient to solve A x = b,
    given:
      - x0          initial guess (flat tensor)
      - args        passed to jac(x, *args)
      - jac(x,args) returns r = b - A x     (flat tensor)
      - apply_precond(r) returns M^{-1} r   (same shape)
    """

    # --- 0) INITIALIZE ---
    x = x0.clone()
    # compute b implicitly: with x=0, jac(0)==b
    b = jac_obs(torch.zeros_like(x), *args)  [0]

    # initial residual r0 = b - A x0  <==> use jac(x0)
    r, obs_tensor = jac_obs(x, *args)

    # apply Jacobi precond: z0 = M^{-1}r0
    z = apply_precond(r)

    # set initial search direction p0 = z0
    p = z.clone()

    # dot(r,z) for the beta formula
    rz_old = torch.dot(r, z)

    for k in range(maxit):
        # 1) compute A·p via your jac:  jac(p) = b - A p  =>  A p = b - jac(p)
        Ap = b - jac_obs(p, *args)[0]

        # 2) CG step size
        alpha = rz_old / torch.dot(p, Ap)

        # 3) update solution and residual
        x = x + alpha * p
        r = r - alpha * Ap

        # 3b) update observation tensor
        obs_tensor = jac_obs(x, *args)[1]

        # logging / convergence check
        res_norm = torch.norm(r)
        if k % 1 == 0 and outfile:
            with open(outfile, 'a') as f:
                f.write(f" PCG it {k:4d} |r|/N = {res_norm:.3e}, |obs| = {obs_tensor.norm(p='fro')}\n")
        if res_norm < tol:
            with open(outfile, 'a') as f:
                f.write(f" PCG converged at it {k:4d} |r|/N = {res_norm:.3e}, |obs| = {obs_tensor.norm(p='fro')}\n")
            break

        # 4) precondition the new residual
        z = apply_precond(r)

        # 5) update β and direction
        rz_new = torch.dot(r, z)
        beta  = rz_new / rz_old
        p     = z + beta * p
        rz_old = rz_new

    return x

def broyden(
    x0, func, args=(), H0=None, outfile=None,
    maxit=100, tol=1e-10,
    alpha_init=1.0,       # Initial step size for line search
    c=1e-4,               # Sufficient decrease parameter
    rho=0.5               # Contraction factor for line search
):
    """
    A modified Broyden's method implementation that uses a backtracking line search
    to pick the step size (alpha) each iteration.

    Parameters:
    -----------
    x0 : ndarray
        Initial guess for the root of func(x)=0.
    func : callable
        Function for which we want to solve func(x)=0.
        Should accept x plus any extra args via *args, return an ndarray.
    args : tuple, optional
        Extra arguments to pass to func.
    H0 : ndarray, optional
        Initial (inverse) Jacobian approximation.
        If None, use the identity matrix.
    maxit : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance on the norm of func(x).
    alpha_init : float, optional
        Initial step size guess for the line search each iteration.
    c : float, optional
        Parameter controlling the required decrease in the norm of f
        for the line search to accept the step.
    rho : float, optional
        Contraction factor to reduce alpha if the sufficient decrease
        condition is not met.

    Returns:
    --------
    x : ndarray
        Approximate solution after maximum iterations or convergence.
    """
    # If no initial H0 provided, use the identity.
    if H0 is None:
        H0 = np.eye(len(x0))

    # Evaluate function at initial guess
    x = x0
    f = func(x, *args)
    f_norm = np.linalg.norm(f)
    if outfile is not None:
        with open(outfile, 'a') as file:
            print(f'Broyden to solve nonlinear equation', file=file)
            print(f'Iteration 0: residual = {f_norm:.3e}', file=file)
    else:
        print(f'Broyden to solve nonlinear equation', file=file)
        print(f'Iteration 0: residual = {f_norm:.3e}')
        if f_norm < tol:
            print('Broyden converged on initial guess.')
            return x

    H = H0.copy()

    for i in range(1, maxit+1):
        # Compute direction p = -H * f
        p = -H @ f

        #------------------------------------------------------------------
        # Backtracking line search to find alpha that (roughly) reduces
        # the norm of f(x). The "sufficient decrease" condition here is:
        #
        #   ||f(x + alpha * p)|| <= (1 - c * alpha) * ||f(x)||.
        #
        # This is a typical inexact line search criterion for root-finding.
        #------------------------------------------------------------------
        alpha = alpha_init
        f_norm_current = np.linalg.norm(f)
        while True:
            x_candidate = x + alpha * p
            f_candidate = func(x_candidate, *args)
            if np.linalg.norm(f_candidate) <= (1 - c * alpha) * f_norm_current:
                # Accept the step
                break
            alpha *= rho
            # If alpha gets too small, break to avoid infinite loops
            if alpha < 1e-16:
                break

        # Take the step with the accepted alpha
        x_new = x_candidate
        f_new = f_candidate
        f_new_norm = np.linalg.norm(f_new)

        # Print iteration info
        if outfile is not None:
            with open(outfile, 'a') as file:
                print(f'Iteration {i}: residual = {f_new_norm:.3e}, alpha = {alpha:.3e}', file=file)
        else:
            print(f'Iteration {i}: residual = {f_new_norm:.3e}, alpha = {alpha:.3e}')

        # Check for convergence
        if f_new_norm < tol:
            if outfile is not None:
                with open(outfile, 'a') as file:
                    print('Broyden converged.', file=file)
            else:
                print('Broyden converged.')
            return x_new

        # Update the inverse Jacobian approximation using the Broyden update
        s = x_new - x              # Step in x
        y = f_new - f              # Change in function
        z = H @ y
        denom = np.dot(s, z)
        if abs(denom) > 1e-20:     # To avoid dividing by near-zero
            H = H + np.outer((s - z), (s.T @ H)) / denom

        # Prepare for next iteration
        x = x_new
        f = f_new
    if outfile is not None:
        with open(outfile, 'a') as file:
            print(f'Broyden did not converge within {maxit} iterations.', file=file)
    else:
        print(f'Broyden did not converge within {maxit} iterations.')
    return x
