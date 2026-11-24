import os
import shutil
import numpy as np
import torch
import time
from modules.dump import projection, isosurface
from modules.seed import make_seed
from scipy.special import comb

class INFILEclass:
    """
    Class to hold input file parameters.
    """
    def __init__(self, infile_path, out_dir_path):
        self.ds = None
        self.chi = None
        self.gridShape = None
        self.fracs_list = None
        self.species_list = None
        self.disp_weights = None
        self.N_list = None
        self.solvent = False
        self.solvent_weights = None
        self.solvent_species_list = None
        self.seed = None
        self.epsilon = None
        self.period = None
        self.x_file = None
        self.dump_freq = None
        self.nIter = None
        self.picard_stepsize = None
        self.picard_tol = None
        self.Lattice_tol = None
        self.Lattice_stepsize = None
        self.fix = "none"
        self.dir_name = out_dir_path
        self.infile_path = infile_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Lattice = torch.zeros((3, 3), device=self.device, dtype=torch.float64)
        self.startTime = time.time()

        with open(self.infile_path, 'r') as file:
            lines = file.readlines()
        
        # Enumerate capital letters
        def letter_to_number(letter):
            return ord(letter) - ord('A')

        for i, line in enumerate(lines):
            stripped_line = line.split('#', 1)[0].rstrip()
            if stripped_line:
                split_line = stripped_line.split()
                if split_line[0] == 'ds':
                    self.ds = float(split_line[1])
                elif split_line[0] == 'chi':
                    numPhases = int(split_line[1])
                    numChis = comb(numPhases, 2, exact=True) # number of chis is n choose 2
                    self.chi = torch.zeros((numPhases, numPhases), device=self.device, dtype=torch.float64)
                    for j in range(numChis):
                        split_line_chi = lines[i+j+1].strip().split()
                        species1 = letter_to_number(split_line_chi[0])
                        species2 = letter_to_number(split_line_chi[1])
                        chiVal = float(split_line_chi[2])
                        self.chi[species1, species2] = chiVal
                        self.chi[species2, species1] = chiVal # ensure symmetric chi
                elif split_line[0] == 'Lattice':
                    split_line1 = lines[i+1].strip().split()
                    split_line2 = lines[i+2].strip().split()
                    split_line3 = lines[i+3].strip().split()
                    self.Lattice[0, :] = torch.tensor([float(split_line1[0]), float(split_line1[1]), float(split_line1[2])], device=self.device, dtype=torch.float64)
                    self.Lattice[1, :] = torch.tensor([float(split_line2[0]), float(split_line2[1]), float(split_line2[2])], device=self.device, dtype=torch.float64)
                    self.Lattice[2, :] = torch.tensor([float(split_line3[0]), float(split_line3[1]), float(split_line3[2])], device=self.device, dtype=torch.float64)
                elif split_line[0] == 'fix':
                    self.fix = split_line[1]
                elif split_line[0] == 'gridShape':
                    self.gridShape = [int(split_line[1]), int(split_line[2]), int(split_line[3])]
                elif split_line[0] == 'initial_Lattice_stepsize':
                    self.Lattice_stepsize = float(split_line[1])
                elif split_line[0] == 'polymer-mixture':
                    mixSize = int(split_line[1])
                    self.N_list = []
                    self.fracs_list = []
                    self.species_list = []
                    self.disp_weights = []
                    for m in range(mixSize):
                        line_temp = lines[i+m+1].split('#', 1)[0].rstrip()
                        split_line_temp = line_temp.split()
                        numBlocks = int((len(split_line_temp) - 4) / 2) # number of blocks in block copolymer
                        self.N_list.append(int(split_line_temp[1]))
                        self.disp_weights.append(float(split_line_temp[3]))
                        fracs = torch.zeros(numBlocks, device=self.device, dtype=torch.float64)
                        species = torch.zeros(numBlocks, device=self.device, dtype=torch.int64)
                        for b in range(numBlocks):
                            species[b] = letter_to_number(split_line_temp[b*2 + 4])
                            fracs[b] = float(split_line_temp[b*2 + 5])
                        self.fracs_list.append(fracs)
                        self.species_list.append(species)
                elif split_line[0] == 'solvent-mixture':
                    self.solvent = True
                    mixSize = int(split_line[1])
                    self.solvent_weights = []
                    self.solvent_species_list = []
                    for m in range(mixSize):
                        split_line_temp = lines[i+m+1].strip().split()
                        self.solvent_weights.append(float(split_line_temp[1]))
                        self.solvent_species_list.append(letter_to_number(split_line_temp[2]))
                elif split_line[0] == 'seed':
                    self.seed = split_line[1]
                elif split_line[0] == 'epsilon':
                    self.epsilon = float(split_line[1])
                elif split_line[0] == 'period':
                    self.period = float(split_line[1])
                elif split_line[0] == 'x_file':
                    self.x_file = split_line[1]
                elif split_line[0] == 'dump_freq':
                    self.dump_freq = int(split_line[1])
                elif split_line[0] == 'nIter':
                    self.nIter = int(split_line[1])
                elif split_line[0] == 'picard_stepsize':
                    self.picard_stepsize = float(split_line[1])
                elif split_line[0] == 'picard_tol':
                    self.picard_tol = float(split_line[1])
                elif split_line[0] == 'Lattice_tol':
                    self.Lattice_tol = float(split_line[1])

        if self.chi is None:
            raise ValueError("chi parameters are missing")
        if self.ds is None:
            raise ValueError("ds is missing.")
        if self.Lattice is None:
            raise ValueError("Lattice is missing.")
        if self.gridShape is None:
            raise ValueError("gridShape is missing.")
        if self.Lattice_stepsize is None:
            raise ValueError("initial_Lattice_stepsize is missing.")
        if self.fracs_list is None:
            raise ValueError("fracs is missing.")
        if self.solvent == True:
            if abs(sum(self.disp_weights) + sum(self.solvent_weights) - 1.0) > 1e-10:
                raise ValueError("mixture weights must sum to 1.0.")
        else:
            if abs(sum(self.disp_weights) - 1.0) >= 1e-10:
                raise ValueError("mixture weights must sum to 1.0.")
        if self.epsilon is None and self.seed is not None:
            raise ValueError("epsilon is missing.")
        if self.dump_freq is None:
            raise ValueError("dump_freq is missing.")
        if self.nIter is None:
            raise ValueError("nIter is missing.")
        if self.picard_stepsize is None:
            raise ValueError("picard_stepsize is missing.")
        if self.picard_tol is None:
            raise ValueError("picard_tol is missing.")
        if self.Lattice_tol is None:
            raise ValueError("Lattice_tol is missing.")
        
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
        if INFILE.x_file is not None:
            self.x = torch.load(INFILE.x_file, map_location=self.device, weights_only=False)
            w_flat = self.x.view(-1, GRID.gridShape[0]*GRID.gridShape[1]*GRID.gridShape[2])
            nSpecies = w_flat.shape[0]
            # check if rho is the correct shape
            if nSpecies != POLY.nSpecies:
                # if not the correct shape, add extra fields which are small and close to zero
                extra_x = INFILE.epsilon * (2 * torch.rand((POLY.nSpecies - nSpecies, GRID.gridShape[0]*GRID.gridShape[1]*GRID.gridShape[2]), device=self.device, dtype=self.dtype) - 1)
                extra_x_flat = extra_x.view(-1)
                # stack the extra fields onto the existing rho and x
                self.x = torch.cat((self.x, extra_x_flat), dim=0)

        elif INFILE.seed is not None:
            self.x = make_seed(INFILE.seed, POLY.nSpecies, GRID.gridShape, amplitude=INFILE.epsilon, period=INFILE.period).to(self.device)
        else:
            self.x = INFILE.epsilon * (2 * torch.rand(POLY.nSpecies * GRID.gridShape[0]*GRID.gridShape[1]*GRID.gridShape[2], device=self.device, dtype=self.dtype) - 1)


class POLYclass:
    """
    Class to hold quantities relavant to the polymer and solvent.
    """
    def __init__(self, INFILE):
        # Account for polydispersity in volume fractions and polymer length
        self.N = INFILE.N_list
        self.disp_weights = INFILE.disp_weights
        polymer_probs = np.array(self.disp_weights) / sum(INFILE.disp_weights) # normalize polymer weights
        self.N_mean = sum(np.array(self.N) * polymer_probs) # multiply by weights to get expected N
        # Chi parameter matrix
        self.chi = INFILE.chi * self.N_mean # multiply by N_mean to scale properly
        # Number of species in system (number of distinct block types)
        self.nSpecies = self.chi.shape[0]
        self.mixSize = len(self.disp_weights)
        self.fracs = INFILE.fracs_list
        self.species = INFILE.species_list
        self.fracsRev = []
        self.speciesRev = []
        self.cumFrac = []
        self.cumFracRev = []
        # Loop over each polymer type in the system
        for i in range(self.mixSize):
            # Cumulative volume fractions
            self.cumFrac.append(torch.cumsum(self.fracs[i], dim=0))
            # Reversed species list
            self.speciesRev.append(torch.flip(self.species[i], dims=[0]))
            # Reversed volume fractions
            self.fracsRev.append(torch.flip(self.fracs[i], dims=[0]))
            # Reversed cumulative volume fractions
            self.cumFracRev.append(torch.cumsum(self.fracsRev[i], dim=0))
        # Chain step size
        self.ds = INFILE.ds
        # Number of steps in each block for each polymer type in system
        self.nS = []
        self.nSRev = []
        for i in range(self.mixSize):
            nS_temp = torch.zeros(len(self.species[i]), dtype=torch.int64, device=INFILE.device)
            nSRev_temp = torch.zeros(len(self.speciesRev[i]), dtype=torch.int64, device=INFILE.device)
            for j in range(len(self.species[i])):
                nS_temp[j] = (int(round((self.cumFrac[i][j] - self.cumFrac[i][j - 1]).item() / self.ds)) if j > 0 else int(round(self.cumFrac[i][j].item() / self.ds)))
                nSRev_temp[j] = (int(round((self.cumFracRev[i][j] - self.cumFracRev[i][j - 1]).item() / self.ds)) if j > 0 else int(round(self.cumFracRev[i][j].item() / self.ds)))
            self.nS.append(nS_temp)
            self.nSRev.append(nSRev_temp)
        
        # Solvent parameters
        self.solvent = INFILE.solvent
        if self.solvent == True:
            self.solvent_weights = INFILE.solvent_weights
            self.solventMixSize = len(self.solvent_weights)
            self.solvent_species = INFILE.solvent_species_list



class GRIDclass:
    """
    Class to hold quantities relevant to the grid.
    """
    def __init__(self, INFILE):
        self.gridShape  = INFILE.gridShape
        self.device = INFILE.device
        self.Lattice = INFILE.Lattice
        self.Lattice_stepsize = INFILE.Lattice_stepsize # Initial stepsize for lattice parameters
        self.fix = INFILE.fix
        self.ucparams = self._get_ucparams()
        self.init_ucparams = self.ucparams.clone()
        self._rebuild()

    def _rebuild(self):
        # Rebuild wavevectors (not necessary I think)
        dx, dy, dz = 1/self.gridShape[0], 1/self.gridShape[1], 1/self.gridShape[2]
        kx    = 2*np.pi*torch.fft.fftfreq(self.gridShape[0], d=dx, device=self.device, dtype=torch.float64)
        ky    = 2*np.pi*torch.fft.fftfreq(self.gridShape[1], d=dy, device=self.device, dtype=torch.float64)
        kz    = 2*np.pi*torch.fft.fftfreq(self.gridShape[2], d=dz, device=self.device, dtype=torch.float64)
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
        self.dV = self.V / (self.gridShape[0]*self.gridShape[1]*self.gridShape[2])
    
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
                                    [                 0,                 0,                 0]
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

    def update_ucparams(self, SOL):
        """ 
        Update the unit cell parameters and rebuild the grid.
        """
        if self.fix == 'none':
            self.ucparams = self.ucparams - self.Lattice_stepsize * SOL.dfdu
            err_f = torch.sqrt(torch.sum(SOL.dfdu**2) / torch.sum(self.ucparams**2)).item()
        elif self.fix == 'prop':
            theta = torch.mean(self.ucparams[:3] / self.init_ucparams[:3]) # proportionality constant
            dfdl = torch.sum(SOL.dfdu[:3] * self.init_ucparams[:3])
            theta -= self.Lattice_stepsize * dfdl
            self.ucparams[:3] = self.init_ucparams[:3] * theta
            err_f = abs((dfdl / theta).item())
        elif self.fix == 'angles':
            self.ucparams[:3] = self.ucparams[:3] - self.Lattice_stepsize * SOL.dfdu[:3]
            err_f = torch.sqrt(torch.sum(SOL.dfdu[:3]**2) / torch.sum(self.ucparams[:3]**2)).item()
        elif self.fix == 'Lattice':
            err_f = 0.0

        self._rebuild()

        return err_f


class DIRclass:
    """
    Class to hold directories.
    """
    def __init__(self, INFILE):
        self.mainDir = INFILE.dir_name
        self.isoDir = f"{INFILE.dir_name}/iso"
        self.projDir = f"{INFILE.dir_name}/proj"
        self.outfile = f"{INFILE.dir_name}/outfile.txt"
        # Clean up/create directories
        if os.path.exists(INFILE.dir_name):
            raise FileExistsError(
                f"Directory '{INFILE.dir_name}' already exists. Please choose a different name."
            )
        else:
            os.makedirs(INFILE.dir_name)
        for d in [self.isoDir, self.projDir]:
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

    def dumpIteration(self, iteration, SOL, POLY, GRID):
        """
        Dump the current state of the simulation.
        """
        if iteration % SOL.dumpFreq == 0:
            self.log("\nDumping plots...")
            projection(self, SOL, POLY, GRID)
            isosurface(self, SOL, POLY, GRID)
            Lattice = GRID.Lattice
            self.log(f"Lattice:\n{Lattice[0,0]:.3f} {Lattice[0,1]:.3f} {Lattice[0,2]:.3f}\n{Lattice[1,0]:.3f} {Lattice[1,1]:.3f} {Lattice[1,2]:.3f}\n{Lattice[2,0]:.3f} {Lattice[2,1]:.3f} {Lattice[2,2]:.3f}\n")
            self.log('Resuming Picard...\n')

    def dumpEnd(self, SOL, POLY, GRID):
        """
        Dump the end state of the simulation.
        """
        self.log("\nFinal volume fractions:")
        for i in range(POLY.nSpecies):
            letter = chr(ord('A') + i)            # 0→'A', 1→'B', …
            self.log(f"     {letter}: {SOL.rho[i].mean().item():.6f}")
        self.log("\nDumping plots...")
        projection(self, SOL, POLY, GRID)
        isosurface(self, SOL, POLY, GRID)
        self.log('Dumping density and x...')
        x_file_path = os.path.join(self.mainDir, "x_final.pt")
        density_file_path = os.path.join(self.mainDir, "density_final.pt")
        torch.save(SOL.x, x_file_path)
        torch.save(SOL.rho, density_file_path)
        self.log('Dumping free energy...')
        free_energy = SOL.freeEn.item()
        with open(os.path.join(self.mainDir, "free_energy.txt"), "w") as file:
            file.write(f"Free Energy: {free_energy:.6f}\n")
        self.log('Dumping Lattice...')
        with open(os.path.join(self.mainDir, "Lattice.txt"), "w") as file:
            Lattice = GRID.Lattice
            file.write(f"Lattice:\n{Lattice[0,0]:.3f} {Lattice[0,1]:.3f} {Lattice[0,2]:.3f}\n{Lattice[1,0]:.3f} {Lattice[1,1]:.3f} {Lattice[1,2]:.3f}\n{Lattice[2,0]:.3f} {Lattice[2,1]:.3f} {Lattice[2,2]:.3f}\n")
        