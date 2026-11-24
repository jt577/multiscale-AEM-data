import os
import shutil
import numpy as np
import torch
import time
from modules.dump import projection, isosurface
from modules.seed import make_seed

class INFILEclass:
    """
    Class to hold input file parameters.
    """
    def __init__(self, infile_path, out_dir_path):
        self.ds = None
        self.chi = None
        self.nGrid = None
        self.fracs = None
        self.seed = None
        self.epsilon = None
        self.period = None
        self.SOL_file = None
        self.dump_freq = None
        self.nIter = None
        self.picard_stepsize = None
        self.picard_tol = None
        self.Lattice_stepsize = None
        self.fix = "none"
        self.dir_name = out_dir_path
        self.infile_path = infile_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Lattice = torch.zeros((3, 3), device=self.device, dtype=torch.float64)
        self.startTime = time.time()

        with open(self.infile_path, 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if stripped_line:
                split_line = stripped_line.split()
                if split_line[0] == 'ds':
                    self.ds = float(split_line[1])
                elif split_line[0] == 'chi':
                    numPhases = int(split_line[1])
                    self.chi = torch.zeros((numPhases, numPhases), device=self.device, dtype=torch.float64)
                    for j in range(numPhases):
                        split_line_chi = lines[i+j+1].strip().split()
                        for k in range(numPhases):
                            self.chi[j, k] = float(split_line_chi[k])
                elif split_line[0] == 'Lattice':
                    split_line1 = lines[i+1].strip().split()
                    split_line2 = lines[i+2].strip().split()
                    split_line3 = lines[i+3].strip().split()
                    self.Lattice[0, :] = torch.tensor([float(split_line1[0]), float(split_line1[1]), float(split_line1[2])], device=self.device, dtype=torch.float64)
                    self.Lattice[1, :] = torch.tensor([float(split_line2[0]), float(split_line2[1]), float(split_line2[2])], device=self.device, dtype=torch.float64)
                    self.Lattice[2, :] = torch.tensor([float(split_line3[0]), float(split_line3[1]), float(split_line3[2])], device=self.device, dtype=torch.float64)
                elif split_line[0] == 'Lattice_stepsize':
                    self.Lattice_stepsize = float(split_line[1])
                elif split_line[0] == 'fix':
                    self.fix = split_line[1]
                elif split_line[0] == 'nGrid':
                    self.nGrid = int(split_line[1])
                elif split_line[0] == 'fracs':
                    self.fracs = torch.zeros(numPhases - 1, device=self.device, dtype=torch.float64)
                    for j in range(numPhases-1):
                        self.fracs[j] = float(split_line[j+1])
                elif split_line[0] == 'seed':
                    self.seed = split_line[1]
                elif split_line[0] == 'epsilon':
                    self.epsilon = float(split_line[1])
                elif split_line[0] == 'period':
                    self.period = float(split_line[1])
                elif split_line[0] == 'SOL_file':
                    self.SOL_file = split_line[1]
                elif split_line[0] == 'dump_freq':
                    self.dump_freq = int(split_line[1])
                elif split_line[0] == 'nIter':
                    self.nIter = int(split_line[1])
                elif split_line[0] == 'picard_stepsize':
                    self.picard_stepsize = float(split_line[1])
                elif split_line[0] == 'picard_tol':
                    self.picard_tol = float(split_line[1])

        if self.chi is None:
            raise ValueError("chi parameters are missing")
        if self.ds is None:
            raise ValueError("ds is missing.")
        if self.Lattice is None:
            raise ValueError("Lattice is missing.")
        if self.Lattice_stepsize is None:
            raise ValueError("Lattice_stepsize is missing.")
        if self.nGrid is None:
            raise ValueError("nGrid is missing.")
        if self.fracs is None:
            raise ValueError("fracs is missing.")
        if self.epsilon is None and self.seed is not None:
            raise ValueError("epsilon is missing.")
        if self.period is None and self.seed is not None:
            raise ValueError("period is missing.")
        if self.dump_freq is None:
            raise ValueError("dump_freq is missing.")
        if self.nIter is None:
            raise ValueError("nIter is missing.")
        if self.picard_stepsize is None:
            raise ValueError("picard_stepsize is missing.")
        if self.picard_tol is None:
            raise ValueError("picard_tol is missing.")
        
    def write_input(self, DIR):
        # Read the input file lines
        with open(self.infile_path, "r") as file:
            in_lines = file.readlines()
        # Open the output file for appending
        with open(DIR.outfile, "a") as file:
            for line in in_lines:
                # Write the original line
                file.write(line)
        DIR.log('\n--------------------------------------------------------------------------\n\nBeginning simulation...\n')


class SOLclass:
    """
    Class to hold quantities relavant to the iterate.
    """
    def __init__(self, INFILE):
        self.x = None
        self.rho = None
        self.iter = None
        self.sanity = None
        self.freeEn = None
        self.dfdu = None
        self.device = INFILE.device
        self.dtype = torch.float64
        self.startTime = INFILE.startTime
        self.dumpFreq = INFILE.dump_freq
    
    def seed(self, INFILE, POLY, GRID):
        """
        Seed the SOL object with initial values.
        """
        if INFILE.SOL_file is not None:
            SOL_file = torch.load(INFILE.SOL_file, map_location=self.device, weights_only=False)
            self.x = SOL_file.x
            self.rho = SOL_file.rho
        elif INFILE.seed is not None:
            self.x = make_seed(INFILE.seed, POLY.nBlocks, 1, GRID.nGrid, amplitude=INFILE.epsilon, period=1/INFILE.period, noise_amp=1e-3).to(self.device)
        else:
            self.x = epsilon * (2 * torch.rand(POLY.nBlocks * (GRID.nGrid)**3, device=self.device, dtype=self.dtype) - 1)


class POLYclass:
    """
    Class to hold quantities relavant to the polymer.
    """
    def __init__(self, INFILE):
        # Chi parameter matrix
        self.chi = INFILE.chi
        # Block (equivalently volume) fractions
        self.fracs = torch.cat([INFILE.fracs, INFILE.fracs.new_tensor([1.0 - torch.sum(INFILE.fracs)])])
        # Cumulartive volume fractions
        self.cumFrac = torch.cat([torch.cumsum(INFILE.fracs, dim=0), INFILE.fracs.new_tensor([1.0])])
        # Reversed cumulative volume fractions
        self.cumFracRev = torch.cat([1 - torch.flip(torch.cumsum(INFILE.fracs, dim=0), dims=[0]), INFILE.fracs.new_tensor([1.0])])
        # Number of blocks in block copolymer
        self.nBlocks = self.chi.shape[0]
        # Chain step size
        self.ds = INFILE.ds
        # Number of steps in each block
        self.nS = torch.zeros(self.nBlocks, dtype=torch.int64, device=INFILE.device)
        self.nSRev = torch.zeros(self.nBlocks, dtype=torch.int64, device=INFILE.device)
        for i in range(self.nBlocks):
            self.nS[i] = (int(round((self.cumFrac[i] - self.cumFrac[i - 1]).item() / INFILE.ds)) if i > 0 else int(round(self.cumFrac[i].item() / INFILE.ds)))
            self.nSRev[i] = (int(round((self.cumFracRev[i] - self.cumFracRev[i - 1]).item() / INFILE.ds)) if i > 0 else int(round(self.cumFracRev[i].item() / INFILE.ds)))

class GRIDclass:
    """
    Class to hold quantities relevant to the grid.
    """
    def __init__(self, INFILE):
        self.nGrid  = INFILE.nGrid
        self.device = INFILE.device
        self.Lattice = INFILE.Lattice
        self.Lattice_stepsize = INFILE.Lattice_stepsize
        self.fix = INFILE.fix
        self.ucparams = self._get_ucparams()
        self._rebuild()

    def _rebuild(self):
        # Rebuild wavevectors (not necessary I think)
        d     = 1/self.nGrid
        kx    = 2*np.pi*torch.fft.fftfreq(self.nGrid, d=d, device=self.device, dtype=torch.float64)
        ky    = kx.clone()
        kz    = kx.clone()
        KX,KY,KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
        K = torch.stack((KX, KY, KZ), dim=0)
        # Rebuild the lattice vector matrix
        self.Lattice = self._get_Lattice()
        # Rebuild the metric and inverse metric
        self.g = self.Lattice.T @ self.Lattice
        self.g_inv = torch.linalg.inv(self.g)
        # Obtain metric derivatives with respect to the unit cell parameters and associated quantities
        self.dgdu = self._get_dgdu()
        self.dg_invdu = -torch.einsum('ij,jku,kl->ilu', self.g_inv, self.dgdu, self.g_inv)
        self.k_g_inv_k = torch.einsum('ixyz,ij,jxyz->xyz', K, self.g_inv, K)
        self.k_dg_invdu_k = torch.einsum('ixyz,iju,jxyz->uxyz', K, self.dg_invdu, K)
        # Obtain volume and volume per grid point
        self.V = torch.linalg.det(self.Lattice)
        self.dV = self.V / (self.nGrid**3)
    
    def _get_dgdu(self):
        """
        Calculate the derivative of the metric tensor with respect to the unit cell parameters, in this order:
            a: length of the first unit cell vector
            b: length of the second unit cell vector
            c: length of the third unit cell vector
            α: angle between the second and third unit cell vectors
            β: angle between the first and third unit cell vectors
            γ: angle between the first and second unit cell vectors
        """
        dgdL = torch.zeros((3, 3, 6), device=self.device, dtype=torch.float64)
        a = self.ucparams[0]
        b = self.ucparams[1]
        c = self.ucparams[2]
        α = self.ucparams[3]
        β = self.ucparams[4]
        γ = self.ucparams[5]
        dgdL[..., 0] = torch.tensor([
                                    [               2*a,    b*torch.cos(γ),    c*torch.cos(β)],
                                    [    b*torch.cos(γ),                 0,                 0],
                                    [    c*torch.cos(β),                 0,                 0]
                                    ], device=self.device, dtype=torch.float64)

        dgdL[..., 1] = torch.tensor([
                                    [                 0,    a*torch.cos(γ),                 0],
                                    [    a*torch.cos(γ),               2*b,    c*torch.cos(α)],
                                    [                 0,    c*torch.cos(α),                 0]
                                    ], device=self.device, dtype=torch.float64)

        dgdL[..., 2] = torch.tensor([
                                    [                 0,                 0,    a*torch.cos(β)],
                                    [                 0,                 0,    b*torch.cos(α)],
                                    [    a*torch.cos(β),    b*torch.cos(α),               2*c]
                                    ], device=self.device, dtype=torch.float64)

        dgdL[..., 3] = torch.tensor([
                                    [                 0,                 0,                 0],
                                    [                 0,                 0, -b*c*torch.sin(α)],
                                    [                 0, -b*c*torch.sin(α),                 0]
                                    ], device=self.device, dtype=torch.float64)

        dgdL[..., 4] = torch.tensor([
                                    [                 0,                 0, -a*c*torch.sin(β)],
                                    [                 0,                 0,                 0],
                                    [ -a*c*torch.sin(β),                 0,                 0]
                                    ], device=self.device, dtype=torch.float64)

        dgdL[..., 5] = torch.tensor([
                                    [                 0, -a*b*torch.sin(γ),                 0],
                                    [ -a*b*torch.sin(γ),                 0,                 0],
                                    [                 0,                 0,                  0]
                                    ], device=self.device, dtype=torch.float64)
        
        return dgdL

    def _get_Lattice(self):
        """
        Get the lattice vector matrix (columns are lattice vectors).
        """
        a = self.ucparams[0]
        b = self.ucparams[1]
        c = self.ucparams[2]
        α = self.ucparams[3]
        β = self.ucparams[4]
        γ = self.ucparams[5]

        e1 = torch.tensor([a, 0, 0], device=self.device, dtype=torch.float64)
        e2 = torch.tensor([b * torch.cos(γ), b * torch.sin(γ), 0], device=self.device, dtype=torch.float64)
        e3 = torch.zeros(3, device=self.device, dtype=torch.float64)
        e3[0] = c * torch.cos(β)
        e3[1] = c * (torch.cos(α) - torch.cos(β)*torch.cos(γ)) / torch.sin(γ)
        e3[2] = torch.sqrt(c**2 - e3[0]**2 - e3[1]**2)
        # Stack vectors so they form the columns of the lattice matrix
        Lattice = torch.stack((e1, e2, e3), dim=1)
        return Lattice 

    def _get_ucparams(self):
        """
        Get the unit cell parameters from the lattice vector matrix.
        """
        # Calculate lengths of the unit cell vectors
        a = torch.linalg.norm(self.Lattice[:, 0])
        b = torch.linalg.norm(self.Lattice[:, 1])
        c = torch.linalg.norm(self.Lattice[:, 2])
        # Calculate angles between the unit cell vectors
        α = torch.acos(torch.dot(self.Lattice[:, 1], self.Lattice[:, 2]) / (b * c))
        β = torch.acos(torch.dot(self.Lattice[:, 0], self.Lattice[:, 2]) / (a * c))
        γ = torch.acos(torch.dot(self.Lattice[:, 0], self.Lattice[:, 1]) / (a * b))
        return torch.tensor([a, b, c, α, β, γ], device=self.device, dtype=torch.float64)

    def update_ucparams(self, new_ucparams):
        self.ucparams = new_ucparams
        self._rebuild()


class DIRclass:
    """
    Class to hold directories.
    """
    def __init__(self, INFILE):
        self.isoDir = f"{INFILE.dir_name}/iso"
        self.projDir = f"{INFILE.dir_name}/proj"
        self.solDir = f"{INFILE.dir_name}/sol"
        self.outfile = f"{INFILE.dir_name}/outfile.txt"
        self.observables = f"{INFILE.dir_name}/observables.txt"
        # Clean up/create directories
        if os.path.exists(INFILE.dir_name):
            raise FileExistsError(
                f"Directory '{INFILE.dir_name}' already exists. Please choose a different name."
            )
        else:
            os.makedirs(INFILE.dir_name)
        for d in [self.isoDir, self.projDir, self.solDir]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        # Clean up/create outfile
        file_path = os.path.join(INFILE.dir_name, self.outfile)
        if os.path.exists(file_path):
            os.remove(file_path)
    
    def log(self, message):
        """
        Log a message to the output file.
        """
        with open(self.outfile, "a") as file:
            print(message, file=file)

    def dump(self, SOL, POLY, GRID):
        """
        Dump the current state of the simulation to a file.
        """
        # Density plots
        projection(self, SOL, POLY, GRID)
        isosurface(self, SOL, POLY, GRID, tau=0.8)
        # Save SOL
        path = os.path.join(self.solDir, f"SOL_iter{SOL.iter}.pt")
        torch.save(SOL, path)