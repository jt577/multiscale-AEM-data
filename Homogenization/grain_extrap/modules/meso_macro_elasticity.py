import torch, os
import numpy as np
from modules.conductivity import calculate_effective_conductivity
from modules.bulk_shear import calculate_effective_bulk_and_shear
from modules.algorithms import broyden

def meso_elasticity(rhos, bulk_moduli, shear_moduli, outfile):
    """
    Calculates mesoscopic elasticity tensor by solving for mechanical equilibrium.
    """
    with open(outfile, 'a') as file:
        print('\n------------------------------------', file=file)
        print('Calculating mesoscopic elasticity tensor...\n', file=file)

    # Calculate elastic tensor
    bulk_list = []
    shear_list = []
    for i in range(rhos.shape[0]):
        bulk_list.append(rhos[i, ...] * bulk_moduli[i])
        shear_list.append(rhos[i, ...] * shear_moduli[i])
    bulk_field = sum(bulk_list)
    shear_field = sum(shear_list)

    # Calculate isotropic bulk and shear modulus
    meso_C = calculate_effective_bulk_and_shear(bulk_field, shear_field, outfile)

    with open(outfile, 'a') as file:
        print('\n------------------------------------', file=file)
        print('Mesoscopic C:\n', file=file)
        print(meso_C.cpu().numpy(), file=file)
        print('\n------------------------------------', file=file)

    return meso_C


def macro_elasticity(meso_C, outfile):
    """
    Calculates macroscopic conductivity and elasticity tensors using the self-consistent effective medium approximation.
    """
    with open(outfile, 'a') as file:
        print('\n------------------------------------', file=file)
        print('Calculating macroscopic elastic moduli...\n', file=file)
    
    meso_C = meso_C.cpu().numpy()

    # Bulk and Shear moduli
    def objfun(x, C):
        # 3x3 Identity
        δ = np.eye(3)
        # Bulk and shear moduli of isotropic material
        K = x[0]
        mu = x[1]
        # Isotropic elasticity tensor
        Cstar =  (K - 2/3 * mu) * np.einsum('ij,kl->ijkl', δ, δ) + mu * (np.einsum('ik,jl->ijkl', δ, δ) + np.einsum('il,jk->ijkl', δ, δ))
        # Difference between elasticity tensors
        ΔC = C - Cstar
        # Constants
        p1 = 1 / (3 * K + 4 * mu)
        p2 = 3 * (K + 2 * mu) / (5 * mu * (3 * K + 4 * mu))
        # projectors :  I'  (volumetric)  and  I''  (deviatoric)
        I_prime = np.einsum('ij,kl->ijkl', δ, δ) / 3.0
        I_sym   = 0.5*(np.einsum('ik,jl->ijkl', δ, δ) +
                    np.einsum('il,jk->ijkl', δ, δ))
        I_sec   = I_sym - I_prime
        # Assemble A
        δδ     = np.einsum('mk,nl->mnkl', δ, δ)
        A = (np.einsum('ijmn,mnkl->ijkl', I_prime, δδ + p1*ΔC) +
            np.einsum('ijmn,mnkl->ijkl', I_sec, δδ + p2*ΔC))
        A_99  = A.reshape(9, 9)
        # ----------  Moore–Penrose pseudo-inverse (works for rank-deficient) ---
        A_pinv_99 = np.linalg.pinv(A_99)           # shape (9,9)
        A_pinv    = A_pinv_99.reshape(3,3,3,3)     # back to 4-tensor
        # Return objective function
        f = np.zeros(2)
        f[0] = np.einsum('iijj->', A_pinv) - 3
        f[1] = np.einsum('ijij->', A_pinv) - 6

        return f

    x0 = np.array([0.5, 0.5])
    x = broyden(x0, func=objfun, args=(meso_C,), tol=1e-10, maxit=100, outfile=outfile)
    macro_K, macro_Mu = x[0], x[1]

    with open(outfile, 'a') as file:        
        print(f"\nEMA K: {macro_K}, Mu: {macro_Mu}", file=file)

    def invert_C(C):
        """
        Invert the elastic tensor C.
        """
        C_99 = C.reshape(9, 9)
        C_pinv_99 = np.linalg.pinv(C_99)
        C_pinv = C_pinv_99.reshape(3, 3, 3, 3)
        return C_pinv
    
    K_V = 1/9 * np.einsum('iijj->', meso_C)
    Mu_V = 1/10 * (np.einsum('ijij->', meso_C) - 1/3 * np.einsum('iijj->', meso_C))
    meso_S = invert_C(meso_C) # Calculate compliance tensor
    K_R = (np.einsum('iijj->', meso_S))**(-1)
    Mu_R = (2/5 * (np.einsum('ijij->', meso_S) - 1/3 * np.einsum('iijj->', meso_S)))**(-1)
    K_H = 1/2 * (K_V + K_R)
    Mu_H = 1/2 * (Mu_V + Mu_R)
    with open(outfile, 'a') as f:
        f.write(f"Voigt K: {K_V}, Mu: {Mu_V}\n")
        f.write(f"Reuss K: {K_R}, Mu: {Mu_R}\n")
        f.write(f"Hill K: {K_H}, Mu: {Mu_H}\n")
    
    return macro_K, macro_Mu