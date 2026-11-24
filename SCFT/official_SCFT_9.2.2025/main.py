import os
import argparse
import shutil
import numpy as np
# ask PyTorch to use expandable segments for its CUDA allocator
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import time
from modules.algorithms import Picard
from modules.core import SCFTeqns
from modules.classes import INFILEclass, SOLclass, POLYclass, GRIDclass, DIRclass

# Obtain paths of input file and output directory from command line arguments
parser = argparse.ArgumentParser(
    description="Run an SCFT calculation on an input file and dump results into a directory"
)
parser.add_argument(
    "infile_path",
    help="path to the input file"
)
parser.add_argument(
    "out_dir_path",
    help="directory where results will be written"
)
args = parser.parse_args()
infile_path  = args.infile_path
out_dir_path  = args.out_dir_path

# Initialize classes
INFILE = INFILEclass(infile_path, out_dir_path)
SOL = SOLclass(INFILE)
POLY = POLYclass(INFILE)
GRID = GRIDclass(INFILE)
DIR = DIRclass(INFILE)

# Write input to outfile
INFILE.write_input(DIR)

# Seed solution or initialize randomly
SOL.seed(INFILE, POLY, GRID)

# Run SCFT
Picard(SOL, POLY, GRID, DIR,
        func=SCFTeqns,
        tol_SCFT=INFILE.picard_tol,
        tol_UC=INFILE.Lattice_tol,
        maxIt=INFILE.nIter,
        beta=INFILE.picard_stepsize
)

DIR.dumpEnd(SOL, POLY, GRID)
DIR.log(f"All done!")