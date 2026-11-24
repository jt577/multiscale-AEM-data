import argparse
import numpy as np
import os
import torch
from modules.meso_macro_conductivity import meso_conductivity, macro_conductivity
from modules.meso_macro_elasticity  import meso_elasticity,   macro_elasticity

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--units",  type=int,   required=True)
    p.add_argument("--tiles",  type=int,   required=True)
    p.add_argument("--geom",   type=str,   choices=["lamellae","gyroid"], required=True)
    p.add_argument("--suff",   type=str,   default="_coarse")
    p.add_argument("--sample", type=int,   required=True)
    p.add_argument("--obs",    type=str,   choices=["conductivity","elasticity"], required=True)
    return p.parse_args()

args = parse_args()
units, tiles, geom, suff, sample, obs = \
    args.units, args.tiles, args.geom, args.suff, args.sample, args.obs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

outfile = f"./outputs/{geom}_{units}u{tiles}t{suff}_s{sample}_{obs}.txt"
if os.path.exists(outfile): os.remove(outfile)

field = torch.load(f'./fields/{geom}_{units}u{tiles}t{suff}_s{sample}.pt',
                    map_location=device)

if obs == "conductivity":
    conductivities = np.array([1e-3, 1.0])
    meso_sigma = meso_conductivity(field, conductivities, outfile)
    macro_sigma = macro_conductivity(meso_sigma, outfile)
else:
    bulk_moduli  = np.array([2.0, 2.0])
    shear_moduli = np.array([1.0, 1e-3])
    meso_C       = meso_elasticity(field, bulk_moduli, shear_moduli, outfile)
    macro_K, macro_Mu = macro_elasticity(meso_C, outfile)
