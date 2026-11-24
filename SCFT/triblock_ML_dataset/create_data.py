# parse_results_multi.py
import os
import re
import glob
import sys
import pandas as pd
import numpy as np
import torch

MORPHOLOGIES = ["cylinders", "spheres", "gyroid", "lamella", "O70", "diamond"]
SUBDIR_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)$")

def _kfreqs(shape, spacing):
    """
    Return a tuple of k-grid 1D arrays per axis that match rfftn output
    (nonnegative last-axis freqs), with physical spacing (dx,dy,dz,...).
    """
    dims = len(shape)
    if isinstance(spacing, (int, float)):
        spacing = (float(spacing),) * dims
    assert len(spacing) == dims
    ks = []
    for ax in range(dims):
        n = shape[ax]
        d = spacing[ax]
        if ax == dims - 1:
            # last axis is real-FFT, only nonnegative frequencies
            ks.append(torch.fft.rfftfreq(n, d=d))
        else:
            ks.append(torch.fft.fftfreq(n, d=d))
    return ks  # list of 1D tensors

def _meshgrid_vecs(ks, device):
    grids = torch.meshgrid(*[k.to(device) for k in ks], indexing='ij')
    return grids  # tuple of length dims, each same shape as rfftn output

def top_k_fourier_modes(
    rho: torch.Tensor,
    spacing=1.0,
    top_n: int = 10,
    exclude_dc: bool = True,
    center: bool = True,
):
    """
    rho: real density field, shape (Nx, Ny, [Nz]) or (…batch…, Nx, Ny, [Nz]).
         If batch dims are present, they’ll be flattened per-sample.
    spacing: voxel spacing (float or tuple per axis).
    Returns: dict with keys: 'kvecs' (top_n x D), 'power' (top_n,), 
                             'indices' (top_n x D int), 'total_power'
    Notes:
      - Uses rfftn so the last axis is nonredundant (k>=0).
      - Excludes DC if requested.
      - If center=True, subtracts mean before FFT (safer when exclude_dc=True).
    """
    # Ensure float & on one device
    device = rho.device
    # Collapse any leading batch dims; treat last D dims as spatial
    # Detect dimensionality D (2D/3D/…)
    # Assumes rho is [..., N1, N2, (N3)]
    # If you have channels, handle separately before calling this.
    D = 2 if rho.ndim == 2 else 3 if rho.ndim == 3 else None
    if D is None:
        D = 3 if rho.shape[-3] > 1 else 2  # heuristic but usually fine
    spatial_shape = rho.shape[-D:]
    if center:
        rho = rho - rho.mean()

    # rFFTs: nonredundant on the last axis only (k_last >= 0)
    F = torch.fft.rfftn(rho, dim=tuple(range(-D, 0)))
    P = (F.real**2 + F.imag**2)  # power spectrum

    # Build k-grid that matches rfftn layout
    ks = _kfreqs(spatial_shape, spacing)
    kgrids = _meshgrid_vecs(ks, device=device)

    # Optionally knock out the DC bin(s)
    if exclude_dc:
        # DC is (0,0,...,0 index)
        idxs = tuple(0 for _ in range(D-1)) + (0,)
        P[idxs] = 0.0

    # Flatten to rank bins
    P_flat = P.reshape(-1)
    # Get top-N indices
    top_vals, top_idx = torch.topk(P_flat, k=top_n, largest=True, sorted=True)
    # Convert flat indices to D-dimensional indices
    unraveled = torch.unravel_index(top_idx, P.shape)

    # Gather k vectors at those indices
    kvecs = []
    for ax in range(D):
        kvecs.append(kgrids[ax][unraveled].reshape(-1, 1))
    kvecs = torch.cat(kvecs, dim=1)  # (top_n, D)

    return {
        "kvecs": kvecs,                 # top-N wave vectors (units = 1/spacing)
        "power": top_vals/top_vals.sum(),              # corresponding powers
        "indices": torch.stack(unraveled, dim=1),  # integer indices in rfftn array
    }

def analyze_run_dict(density_dict, spacing=1.0, top_n=10):
    results = {}
    for name, rhos in density_dict.items():
        outs = []
        for rho in rhos:
            outs.append(top_k_fourier_modes(rho, spacing=spacing, top_n=top_n, exclude_dc=True, center=True))
        results[name] = outs
    return results

def get_similarity(A_list, B_list):
    """
    Compute similarity matrix between two lists of morphologies (list is indexing the phase)
    and return the max number of zero distances found in matrices.
    """
    num_zeros_list = []
    n = A_list[0]['kvecs'].shape[0]
    for A in A_list:
        for B in B_list:
            mtx = (A['kvecs'].unsqueeze(1) - B['kvecs'].unsqueeze(0)).norm(dim=2)
            # count number of zeros
            num_zeros = torch.sum(mtx < 1e-5).item()
            num_zeros_list.append(num_zeros)
    return max(num_zeros_list) / n

def get_similarity_mtx(A_dict, B_dict):
    """
    Compute similarity matrix between two dictionaries of morphologies (dict keys are indexing the phase)
    and return the similarity matrix.
    """
    morphs_A = list(A_dict.keys())
    morphs_B = list(B_dict.keys())
    sim_mtx = torch.zeros((len(morphs_A), len(morphs_B)))
    for i, morph_A in enumerate(morphs_A):
        for j, morph_B in enumerate(morphs_B):
            sim = get_similarity(A_dict[morph_A], B_dict[morph_B])
            sim_mtx[i, j] = sim
    return sim_mtx

def flag_morphs(A_dict, B_dict):
    """
    Flag morphologies in A_dict that have a similarity of 1 with any morphology in B_dict.
    """
    flagged = {}
    sim_mtx = get_similarity_mtx(A_dict, B_dict)
    max_indices_0 = torch.argmax(sim_mtx, dim=0)
    max_indices_1 = torch.argmax(sim_mtx, dim=1)
    indices = torch.linspace(0, len(A_dict)-1, steps=len(A_dict), dtype=torch.int64)
    ind_diffs_0 = max_indices_0 - indices
    ind_diffs_1 = max_indices_1 - indices
    flagged_indices_0 = torch.where(ind_diffs_0 != 0)
    flagged_indices_1 = torch.where(ind_diffs_1 != 0)
    flagged_morphs_0 = [list(A_dict.keys())[i] for i in flagged_indices_0[0].tolist()]
    flagged_morphs_1 = [list(B_dict.keys())[i] for i in flagged_indices_1[0].tolist()]
    for i, morph in enumerate(A_dict.keys()):
        if morph in flagged_morphs_0 or morph in flagged_morphs_1 or sim_mtx[i,i] < 0.5:
            flagged[morph] = True
        else:
            flagged[morph] = False
    return flagged




def parse_one_samples_dir(samples_dir, reference_density_dict):
    """Return list of record dicts from a single samples directory."""
    records = []
    if not os.path.isdir(samples_dir):
        print(f"[warn] Skipping missing directory: {samples_dir}")
        return records

    # each subfolder is named chiAB_chiAC_chiBC_fA_fC
    for sub_ind, subname in enumerate(sorted(os.listdir(samples_dir))):
        print(f"     Processing subdirectory {sub_ind} named {subname}")
        subdir = os.path.join(samples_dir, subname)
        if not os.path.isdir(subdir):
            continue

        m = SUBDIR_RE.match(subname)
        if not m:
            # skip folders that don't match expected naming
            continue

        # require all output dirs present
        out_dirs = [os.path.isdir(os.path.join(subdir, f"out_{morph}")) for morph in MORPHOLOGIES]
        if not all(out_dirs):
            # skip incomplete sample
            continue

        chiAB, chiAC, chiBC, fA_str, fC_str = m.groups()
        chiAB, chiAC, chiBC = map(int, (chiAB, chiAC, chiBC))
        fA = float(fA_str)
        fC = float(fC_str)

        rec = {"chiAB": chiAB, "chiAC": chiAC, "chiBC": chiBC, "fA": fA, "fC": fC}

        # Assemble density dict
        densities = {}
        for morph in MORPHOLOGIES:
            out_dir = os.path.join(subdir, f"out_{morph}")
            if not os.path.isfile(f'{out_dir}/density_final.pt'):
                densities[morph] = torch.randn(3,32,32,32)  # dummy random noise if missing
            else:
                densities[morph] = torch.load(f'{out_dir}/density_final.pt', map_location=torch.device('cpu'))
        density_dict = analyze_run_dict(densities, spacing=1, top_n=128)

        failed = {}
        # Check convergence
        for morph in MORPHOLOGIES:
            out_dir = os.path.join(subdir, f"out_{morph}")
            failed[morph] = False
            # 1) check for projection failures
            proj_dir = os.path.join(out_dir, "proj")
            if os.path.isdir(proj_dir):
                imgs = glob.glob(os.path.join(proj_dir, "projection_iter*.png"))
                if imgs:
                    iters = []
                    for img in imgs:
                        fname = os.path.basename(img)
                        mo = re.match(r"projection_iter(\d+)\.png", fname)
                        if mo:
                            iters.append(int(mo.group(1)))
                    if iters and max(iters) >= 1999:
                        failed[morph] = True
        
        # check similarity between reference morphologies
        flagged = flag_morphs(density_dict, reference_density_dict)
        for morph in MORPHOLOGIES:
            if flagged[morph] == True:
                failed[morph] = True
        # Record failure status
        print(f"            Failures: {failed}")
        
        # 2) Try to read free_energy.txt - fails if cannot
        for morph in MORPHOLOGIES:
            energy = np.nan
            out_dir = os.path.join(subdir, f"out_{morph}")
            if not failed[morph]:
                fe_path = os.path.join(out_dir, "free_energy.txt")
                if os.path.isfile(fe_path):
                    try:
                        txt = open(fe_path).read()
                        energy = float(re.search(r"Free Energy:\s*([0-9.+-eE]+)", txt).group(1))
                    except Exception:
                        energy = np.nan

            rec[f"FE_{morph}"] = energy

        records.append(rec)

    return records

def main():
    base = "/data2/jt577/2025/September_2025/SCFT/triblock_ML"
    base_curr = os.getcwd()

    # Use CLI args if provided, else default to your two dirs
    # Example: python parse_results_multi.py samples_Sep2 samples_Aug26
    if len(sys.argv) > 1:
        sample_dirs = [os.path.join(base, d) for d in sys.argv[1:]]
    else:
        sample_dirs = [
                        os.path.join(base, "samples_Aug26"),
                        os.path.join(base, "samples_Sep2"),
                    ]
    # Location of reference densities for similarity checking
    reference_densities = {}
    for morph in MORPHOLOGIES:
        reference_densities[morph] = torch.load(f'/data2/jt577/2025/September_2025/SCFT/distance_testing/good_runs_1/out_{morph}/density_final.pt', map_location=torch.device('cpu'))
    reference_density_dict = analyze_run_dict(reference_densities, spacing=1, top_n=128)

    print("Scanning sample directories (later dirs override duplicates):")
    for i, d in enumerate(sample_dirs, 1):
        print(f"  {i}. {d}")

    # Merge with de-duplication; last directory wins on duplicate keys
    # Key on the parameter tuple so the same case across dirs overwrites cleanly
    merged = {}
    counts = {"seen": 0, "kept": 0, "overwritten": 0}

    for d in sample_dirs:
        print(f"Processing directory: {d}")
        recs = parse_one_samples_dir(d, reference_density_dict)
        for i, rec in enumerate(recs):
            key = (rec["chiAB"], rec["chiAC"], rec["chiBC"], rec["fA"], rec["fC"])
            counts["seen"] += 1
            if key in merged:
                counts["overwritten"] += 1
            merged[key] = rec

    records = list(merged.values())
    counts["kept"] = len(records)

    df = pd.DataFrame(records)
    out_csv = os.path.join(base_curr, "dataset.csv")
    df.to_csv(out_csv, index=False)

    print(f"\nWrote dataset to '{out_csv}'")
    print(f"Total candidates seen: {counts['seen']}")
    print(f"Unique kept (after de-dup): {counts['kept']}")
    print(f"Overwritten due to duplicates: {counts['overwritten']}")

if __name__ == "__main__":
    main()
