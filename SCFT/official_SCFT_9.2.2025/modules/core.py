import torch
import numpy as np
import os
import time
from modules.algorithms import simpson

def pseudospectral(w, SOL, POLY, GRID, species, cumFrac, nS, N):
    """
    Solve the 3D PDE:
        dq/ds = laplacian(q) - w(s, r) * q(s, r),
    with a pseudospectral approach.
    If solving for backwards propagator, be sure to use POLY.speciesRev, POLY.cumFracRev and POLY.nSRev instead of POLY.species, POLY.cumFrac and POLY.nS.
    """
    # Spectral step helper function
    def spectralStep(f, factor):
        f_hat = torch.fft.fftn(f)
        res_hat = torch.exp(-factor * GRID.k_g_inv_k) * f_hat
        res = torch.fft.ifftn(res_hat).real
        return res

    # Chain step size
    ds = POLY.ds * N / POLY.N_mean # Scale by N / N_mean

    # Number of blocks
    nBlocks = len(species)

    # Initial condition: q(s=0) as ones
    qInit = torch.ones(*GRID.gridShape, dtype=SOL.dtype, device=SOL.device)

    # Initialize empty list of propagators
    q = []
    for i in range(nBlocks):
        q.append(torch.zeros((nS[i]+1, *GRID.gridShape), dtype=SOL.dtype, device=SOL.device))

    # Solve for propagators
    qTemp = q[0].clone()
    qTemp[0] = qInit

    # Iterate over blocks
    for i in range(nBlocks):
        # Iterate over steps for this block
        for j in range(nS[i]):
            fullstep = torch.exp(-ds/2.0 * w[species[i]]) * qTemp[j]
            fullstep = spectralStep(fullstep, ds)
            qTemp[j+1] = torch.exp(-ds/2.0 * w[species[i]]) * fullstep

        q[i] = qTemp.clone()

        if i < nBlocks - 1:
            qTemp = q[i+1].clone()
            qTemp[0] = q[i][-1]

    return q

def SCFTeqns(SOL, POLY, GRID, DIR):
    """
    Evaluates the new iterate based on the SCFT equations.
    """
    # Chemical potentials
    w = (SOL.x).view(POLY.nSpecies, *GRID.gridShape)

    # Initialize relevant quantities
    SOL.sanity = 0
    SOL.rho = torch.zeros((POLY.nSpecies, *GRID.gridShape), dtype=SOL.dtype, device=SOL.device)
    SOL.dfdu = 0
    logQ = 0

    # Iterate over the types of dispersed polymers
    for i in range(POLY.mixSize):
        # Get info for this polymer for indexing simplicity
        N = POLY.N[i]
        species = POLY.species[i]
        fracs = POLY.fracs[i]
        cumFrac = POLY.cumFrac[i]
        nS = POLY.nS[i]
        speciesRev = POLY.speciesRev[i]
        cumFracRev = POLY.cumFracRev[i]
        nSRev = POLY.nSRev[i]
        nBlocks = len(species)
        disp_weight = POLY.disp_weights[i]

        # Forward propagators
        q = pseudospectral(w, SOL, POLY, GRID, species, cumFrac, nS, N)

        # Backward propagators
        qc = pseudospectral(w, SOL, POLY, GRID, speciesRev, cumFracRev, nSRev, N)

        # Reverse backward propagators
        qcRev = []
        for prop in qc:
            propRev = torch.flip(prop, dims=[0])
            qcRev.append(propRev)
        qcRev = qcRev[::-1]

        # Laplacian of q (for evaluating derivative wrt lattice parameters)
        laplacian_q_disp = []
        k_dg_invdu_k = GRID.k_dg_invdu_k
        laplacian_q = []
        for prop in q:
            prop_hat = torch.fft.fftn(prop, dim=[1, 2, 3])
            prop_extended = prop.view(1, prop.shape[0], prop.shape[1], prop.shape[2], prop.shape[3])
            k_dg_invdu_k_extended = k_dg_invdu_k.view(k_dg_invdu_k.shape[0], 1, k_dg_invdu_k.shape[1], k_dg_invdu_k.shape[2], k_dg_invdu_k.shape[3])
            laplacian_q.append(torch.fft.ifftn(-k_dg_invdu_k_extended * prop_hat, dim=[2,3,4]).real)

        # Q (spatial integral over last block propagator)
        Q = 1 / GRID.V * torch.fft.fftn(q[-1][-1])[0,0,0].real * GRID.dV

        # Convolutions (before accounting for repeated species types if any)
        conv = torch.zeros((nBlocks, *GRID.gridShape), dtype=SOL.dtype, device=SOL.device)
        for j in range(nBlocks):
            conv[j] = simpson(f=q[j]*qcRev[j], a=(cumFrac[j-1] if j > 0 else 0), b=cumFrac[j], N=nS[j], dim=0)

        # Sanity checks
        Qc = 1 / GRID.V * torch.fft.fftn(qc[-1][-1])[0,0,0].real * GRID.dV # for sanity check
        Q_check = torch.abs(Q - Qc) / Q
        SOL.sanity += Q_check
        for j in range(nBlocks):
            conv_check = torch.abs(torch.fft.fftn(conv[j])[0,0,0].real * GRID.dV / Q - fracs[j] * GRID.V)
            SOL.sanity += conv_check

        # Species volume fractions
        for j in range(nBlocks):
            SOL.rho[species[j]] += disp_weight * conv[j] / Q

        # Weighted logarithm of Q
        logQ += disp_weight * POLY.N_mean / N * torch.log(Q)

        # Derivative of free energy wrt lattice parameters
        sumTemp = 0
        for j in range(nBlocks):
            qcRev_extended = qcRev[j].view(1, *qcRev[j].shape)
            conv_temp = simpson(f=laplacian_q[j]*qcRev_extended, a=(cumFrac[j-1] if j > 0 else 0), b=cumFrac[j], N=nS[j], dim=1)
            sumTemp += torch.fft.fftn(conv_temp, dim=[1,2,3])[:,0,0,0].real * GRID.dV

        SOL.dfdu += - disp_weight * sumTemp / (Q * GRID.V)
    
    # Iterate over the types of solvent if present
    if POLY.solvent == True:
        for i in range(POLY.solventMixSize):
            species = POLY.solvent_species[i]
            disp_weight = POLY.solvent_weights[i]

            # boltzmann factor
            exp_w = torch.exp(-w[species] / POLY.N_mean)

            # solvent Q
            Q = 1 / GRID.V * torch.fft.fftn(exp_w)[0,0,0].real * GRID.dV

            # Update logQ
            logQ += disp_weight * POLY.N_mean * torch.log(Q)

            # solvent volume fractions
            SOL.rho[species] += disp_weight * exp_w / Q

    # Lagrange multiplier
    chiInv = torch.linalg.inv(POLY.chi)
    ones = torch.ones(POLY.nSpecies, dtype=SOL.dtype, device=SOL.device)
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
    SOL.freeEn = -logQ + 1/GRID.V * (torch.fft.fftn(interaction)[0,0,0].real - torch.fft.fftn(rwProd)[0,0,0].real) * GRID.dV

    # Flatten to iterate
    xNew = w.view(-1)

    return xNew