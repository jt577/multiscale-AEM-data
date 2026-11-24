import torch
import numpy as np
import os
import time
from modules.algorithms import simpson

def pseudospectral(w, SOL, POLY, GRID, backwards=False):
    """
    Solve the 3D PDE:
        dq/ds = laplacian(q) - w(s, r) * q(s, r),
    with a pseudo-spectral approach.
    """
    # Spectral step helper function
    def spectralStep(f, factor):
        f_hat = torch.fft.fftn(f)
        res_hat = torch.exp(-factor * GRID.k_g_inv_k) * f_hat
        res = torch.fft.ifftn(res_hat).real
        return res
    # Check if performing forward or backward propagation
    if backwards:
        w = torch.flip(w, dims=[0])
        cumFrac = POLY.cumFracRev
        nS = POLY.nSRev
    else:
        cumFrac = POLY.cumFrac
        nS = POLY.nS
    # Chain step size
    ds = POLY.ds
    # Initial condition: q(s=0) as ones
    qInit = torch.ones((GRID.nGrid, GRID.nGrid, GRID.nGrid), dtype=SOL.dtype, device=SOL.device)
    # Initialize empty list of propagators
    q = []
    for i in range(POLY.nBlocks):
        q.append(torch.zeros((nS[i]+1, GRID.nGrid, GRID.nGrid, GRID.nGrid), dtype=SOL.dtype, device=SOL.device))
    # Solve for propagators
    qTemp = q[0].clone()
    qTemp[0] = qInit
    for i in range(POLY.nBlocks):
        for j in range(nS[i]):
            fullstep = torch.exp(-ds/2.0 * w[i]) * qTemp[j]
            fullstep = spectralStep(fullstep, ds)
            qTemp[j+1] = torch.exp(-ds/2.0 * w[i]) * fullstep
        q[i] = qTemp.clone()
        if i < POLY.nBlocks - 1:
            qTemp = q[i+1].clone()
            qTemp[0] = q[i][-1]
    return q

def SCFTeqns(SOL, POLY, GRID, DIR):
    """
    Evaluates the new iterate based on the SCFT equations.
    """
    # Chemical potentials
    w = (SOL.x).view(POLY.nBlocks, GRID.nGrid, GRID.nGrid, GRID.nGrid)
    # Forward propagators
    q = pseudospectral(w, SOL, POLY, GRID, backwards=False)
    # Backward propagators
    qc = pseudospectral(w, SOL, POLY, GRID, backwards=True)
    # Reverse qc
    qcRev = []
    for prop in qc:
        propRev = torch.flip(prop, dims=[0])
        qcRev.append(propRev)
    qcRev = qcRev[::-1]
    # Laplacian of q
    laplacian_q = []
    k_dg_invdu_k = GRID.k_dg_invdu_k
    for prop in q:
        prop_hat = torch.fft.fftn(prop, dim=[1, 2, 3])
        prop_extended = prop.view(1, prop.shape[0], prop.shape[1], prop.shape[2], prop.shape[3])
        k_dg_invdu_k_extended = k_dg_invdu_k.view(k_dg_invdu_k.shape[0], 1, k_dg_invdu_k.shape[1], k_dg_invdu_k.shape[2], k_dg_invdu_k.shape[3])
        laplacian_q.append(torch.fft.ifftn(-k_dg_invdu_k_extended * prop_hat, dim=[2,3,4]).real)
    # Q (spatial integral over last block propagator)
    Q = 1 / GRID.V * torch.fft.fftn(q[-1][-1])[0,0,0].real * GRID.dV
    # Convolutions
    conv = torch.zeros_like(w)
    for i in range(POLY.nBlocks):
        conv[i] = simpson(f=q[i]*qcRev[i], a=(POLY.cumFrac[i-1] if i > 0 else 0), b=POLY.cumFrac[i], N=POLY.nS[i], dim=0)
    # Sanity checks
    Qc = 1 / GRID.V * torch.fft.fftn(qc[-1][-1])[0,0,0].real * GRID.dV # for sanity check
    SOL.sanity = 0
    Q_check = torch.abs(Q - Qc) / Q
    SOL.sanity += Q_check
    for i in range(POLY.nBlocks):
        conv_check = torch.abs(torch.fft.fftn(conv[i])[0,0,0].real * GRID.dV / Q - POLY.fracs[i] * GRID.V)
        SOL.sanity += conv_check
    # Block densities
    SOL.rho = conv / Q
    # Lagrange multiplier
    chiInv = torch.linalg.inv(POLY.chi)
    ones = torch.ones(POLY.nBlocks, dtype=SOL.dtype, device=SOL.device)
    nu = ones @ chiInv / (ones @ chiInv @ ones)
    nu = nu.view(-1, 1, 1, 1)
    nu = torch.sum(nu * w, dim=0)
    # New chemical potentials
    w = torch.einsum('ij,jxyz->ixyz', POLY.chi, SOL.rho) + nu
    # Set zeroth fourier mode of w to zero
    wHat = torch.fft.fftn(w, dim=[1, 2, 3])
    wHat[:,0,0,0] = 0.0
    w = torch.fft.ifftn(wHat, dim=[1, 2, 3]).real
    # Free energy
    interaction = 1/2 * torch.einsum('ixyz,ij,jxyz->xyz', SOL.rho, POLY.chi, SOL.rho) # 1/2 to prevent double counting
    rwProd = torch.einsum('ixyz,ixyz->xyz', SOL.rho, w)
    SOL.freeEn = (-torch.log(Q)
                    + 1/GRID.V * (torch.fft.fftn(interaction)[0,0,0].real - torch.fft.fftn(rwProd)[0,0,0].real) * GRID.dV
    )
    # Derivative of free energy wrt lattice
    sumTemp = 0
    for i in range(POLY.nBlocks):
        qcRev_extended = qcRev[i].view(1, qcRev[i].shape[0], qcRev[i].shape[1], qcRev[i].shape[2], qcRev[i].shape[3])
        conv_temp = simpson(f=laplacian_q[i]*qcRev_extended, a=(POLY.cumFrac[i-1] if i > 0 else 0), b=POLY.cumFrac[i], N=POLY.nS[i], dim=1)
        sumTemp += torch.fft.fftn(conv_temp, dim=[1,2,3])[:,0,0,0].real * GRID.dV
    SOL.dfdu = -sumTemp / (Q * GRID.V)
    # Flatten to iterate
    xNew = w.view(-1)
    return xNew