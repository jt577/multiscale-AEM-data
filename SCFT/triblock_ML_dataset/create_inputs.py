# generate_samples.py
import numpy as np
import os

def main():
    # --- 1) sampling setup ---
    num_samples = 5000
    chi_ranges = {
        "chiAB": (10, 80),
        "chiAC": (10, 80),
        "chiBC": (10, 40)
    }

    np.random.seed(1717722)
    samples = []
    for _ in range(num_samples):
        chiAB = int(np.round(np.random.uniform(*chi_ranges["chiAB"])))
        chiAC = int(np.round(np.random.uniform(*chi_ranges["chiAC"])))
        chiBC = int(np.round(np.random.uniform(*chi_ranges["chiBC"])))
        fC = float(np.round(np.random.uniform(0.3, 0.5), 2))
        fA = float(np.round(np.random.uniform(0.1, 1 - fC - 0.1), 2))
        fB = float(np.round(1 - fC - fA, 2))
        # Avoid regime where chiAB or chiAC is smaller than chiBC
        if chiAB < chiBC or chiAC < chiBC:
            continue
        else:
            samples.append((chiAB, chiAC, chiBC, fA, fC))

    # --- 2) directories and file templates ---
    base = os.getcwd()
    samples_dir = os.path.join(base, "samples_Sep12")
    os.makedirs(samples_dir, exist_ok=True)

    lattices = {
        "cylinders": 
"""1.5 3.0 0.0
2.6 0.0 0.0
0.0 0.0 3.0""",
        "spheres": 
"""3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0""",
        "gyroid":   
"""3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0""",
        "lamella":  
"""3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0""",
        "diamond": 
"""3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0""",
        "O70":   
"""3.0 0.0 0.0
0.0 6.0 0.0
0.0 0.0 10.39"""
    }

    morphologies = ["cylinders", "spheres", "gyroid", "lamella", "diamond", "O70"]
    slurm_template = """#!/bin/bash
#SBATCH -N 1 -n 1 -c 40 -p hi_mem2 -J SCFT -o slurm_out.o%j
#SBATCH --time=02:00:00

source /data2/jt577/COSMO-RS/myenv/bin/activate

python /data2/jt577/2025/September_2025/SCFT/official_SCFT_9.2.2025/main.py cylinders_in.txt out_cylinders
python /data2/jt577/2025/September_2025/SCFT/official_SCFT_9.2.2025/main.py gyroid_in.txt   out_gyroid
python /data2/jt577/2025/September_2025/SCFT/official_SCFT_9.2.2025/main.py lamella_in.txt  out_lamella
python /data2/jt577/2025/September_2025/SCFT/official_SCFT_9.2.2025/main.py diamond_in.txt  out_diamond
python /data2/jt577/2025/September_2025/SCFT/official_SCFT_9.2.2025/main.py O70_in.txt      out_O70
python /data2/jt577/2025/September_2025/SCFT/official_SCFT_9.2.2025/main.py spheres_in.txt  out_spheres
"""

    for chiAB, chiAC, chiBC, fA, fC in samples:
        fB = round(1 - fA - fC, 2)
        subname = f"{chiAB}_{chiAC}_{chiBC}_{fA:.2f}_{fC:.2f}"
        subdir = os.path.join(samples_dir, subname)
        os.makedirs(subdir, exist_ok=True)

        # write each morphology input file
        for morph in morphologies:
            txt = f"""# Input file for Polymer SCFT

# Polymer Properties
ds      0.005
chi 3
A B {chiAB}
A C {chiAC}
B C {chiBC}
polymer-mixture 1
N 1 weight 1.0  A {fA:.2f} B {fB:.2f} C {fC:.2f}

# Box Geometry and grid
Lattice
{lattices[morph]}
gridShape   32 32 32
fix     prop
Lattice_tol     3e-3
initial_Lattice_stepsize  0.10

# Nonlinear Solver
picard_stepsize  0.10
picard_tol       3e-3
nIter            2000

# Initial State
seed {morph}
period 1
epsilon 1

# Dumping
dump_freq   2000
"""
            with open(os.path.join(subdir, f"{morph}_in.txt"), "w") as f:
                f.write(txt)

        # write the slurm launcher
        with open(os.path.join(subdir, "run.sh"), "w") as f:
            f.write(slurm_template)

    print(f"Generated {len(samples)} sample folders under '{samples_dir}'")

if __name__ == "__main__":
    main()
