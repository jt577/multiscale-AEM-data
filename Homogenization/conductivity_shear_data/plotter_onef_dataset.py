# ============================================================
# Use a collection of optimal translation matrices to assemble
# a SINGLE OVERLAY plot of ALL bounds (no duplication)
# WITH ZOOMED DATASET PANEL + POINT LABELS
# ============================================================

from scipy.optimize import brentq
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, argparse
import torch

from helper import TranslationHelper
from misc import phase_props_holder

# ============================================================
# ARGUMENT PARSING
# ============================================================

parser = argparse.ArgumentParser(
    description="Overlay all translation bounds into a single plot"
)
parser.add_argument("infile_path", help="path to the input file")
parser.add_argument("out_dir_path", help="directory where results will be written")
args = parser.parse_args()

infile_path   = args.infile_path
out_dir_path  = args.out_dir_path
os.makedirs(out_dir_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# READ INPUT FILE
# ============================================================

dataset_paths = None
special_dataset_paths = None
with open(infile_path, 'r') as openfile:
    lines = openfile.readlines()
    for line in lines:
        line = line.split('#', 1)[0].strip()
        if not line:
            continue

        parts = line.split()
        key = parts[0]

        if key == 'k1':
            k1 = float(parts[1])
        elif key == 'k2':
            k2 = float(parts[1])
        elif key == 'mu1':
            mu1 = float(parts[1])
        elif key == 'mu2':
            mu2 = float(parts[1])
        elif key == 'sig1':
            sig1 = float(parts[1])
        elif key == 'sig2':
            sig2 = float(parts[1])
        elif key == 'f':
            f = float(parts[1])
        elif key == 'bound_types':
            bound_types = parts[1:]
        elif key == 'calc_type':
            calc_type = parts[1]
        elif key == 'T_paths':
            T_paths = parts[1:]
        elif key == 'dataset_paths':
            dataset_paths = parts[1:]
        elif key == 'special_dataset_paths':
            special_dataset_paths = parts[1:]

if calc_type == 'bulk':
    n_shear = 1
elif calc_type == 'shear':
    n_shear = 5

# ============================================================
# ASSEMBLER CLASS
# ============================================================

class Assembler():
    def __init__(self, T_path, helper, bound_type):
        self.T = np.load(T_path)
        self.helper = helper
        self.bound_type = bound_type
        self.zopt = self._solve_nonlinear()

    def _solve_nonlinear(self):
        helper = self.helper
        zk = 100
        D_k = helper.D_k
        D_mu = helper.D_mu
        D_sig = helper.D_sig

        def nonlinfun(x, alpha, T):
            A = D_k * zk
            B = D_mu + alpha * D_sig
            eigvals = np.linalg.eigh(A + B * x + T)[0]
            return eigvals[0]

        N = 600
        alpha_min = 0.001
        alpha_max = 200.0
        t = np.linspace(0, 1, N)
        alphas = alpha_min * (alpha_max/alpha_min)**t

        zopt = []
        for alpha in alphas:
            z_sol = brentq(
                f=nonlinfun,
                args=(alpha, self.T),
                a=0.0,
                b=500.0,
                xtol=1e-10,
                maxiter=100
            )
            zopt.append([z_sol, alpha * z_sol])

        return np.array(zopt)

    def convert_to_physical(self, phase_props):
        mu1 = self.helper.phase_props.mu1
        mu2 = self.helper.phase_props.mu2
        sig1 = self.helper.phase_props.sig1
        sig2 = self.helper.phase_props.sig2

        if self.bound_type == 'upper_1':
            yopt = 1/self.zopt
        elif self.bound_type == 'upper_2':
            yopt = np.zeros(self.zopt.shape)
            yopt[:,0] = self.zopt[:,0]
            yopt[:,1] = 1/self.zopt[:,1]
        elif self.bound_type == 'lower_1':
            yopt = np.zeros(self.zopt.shape)
            yopt[:,0] = 1/self.zopt[:,0]
            yopt[:,1] = self.zopt[:,1]
        elif self.bound_type == 'lower_2':
            yopt = self.zopt

        mu_eff = self.helper.inv_y_transform(yopt[:,0], phase_props.f, mu1, mu2)
        sig_eff = self.helper.inv_y_transform(yopt[:,1], phase_props.f, sig1, sig2)

        return mu_eff, sig_eff

# ============================================================
# TWO-PANEL OVERLAY + ZOOM + LABELS
# ============================================================

def plot_all_bounds_overlay(assemblers, phase_props, datasets=None, special_datasets=None, out_path="translation_bounds_overlay.png"):
    k1, k2 = phase_props.k1, phase_props.k2
    mu1, mu2 = phase_props.mu1, phase_props.mu2
    sig1, sig2 = phase_props.sig1, phase_props.sig2
    f = phase_props.f

    klow  = 1/(f*1/k1  + (1-f)*1/k2)
    mulow = 1/(f*1/mu1 + (1-f)*1/mu2)
    siglow= 1/(f*1/sig1+ (1-f)*1/sig2)

    khigh  = f*k1  + (1-f)*k2
    muhigh = f*mu1 + (1-f)*mu2
    sighigh= f*sig1+ (1-f)*sig2

    mu_lower_HS, sig_lower_HS = phase_props.mu_lower_HS, phase_props.sig_lower_HS
    mu_upper_HS, sig_upper_HS = phase_props.mu_upper_HS, phase_props.sig_upper_HS

    midpt_mu_lower  = 0.5*(mulow + mu_lower_HS)
    midpt_mu_upper  = 0.5*(muhigh + mu_upper_HS)
    midpt_sig_lower = 0.5*(siglow + sig_lower_HS)
    midpt_sig_upper = 0.5*(sighigh + sig_upper_HS)

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 7))

    # ----------------------------
    # helper: plot only in-box segments
    # ----------------------------
    def extract_masked_segments(x, y, mask, min_len=2):
        """
        Return list of (xseg, yseg) contiguous segments where mask==True.
        """
        x = np.asarray(x); y = np.asarray(y)
        mask = np.asarray(mask, dtype=bool)

        idx = np.where(mask)[0]
        if idx.size == 0:
            return []

        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)

        segs = []
        for run in runs:
            if run.size >= min_len:
                segs.append((x[run], y[run]))
        return segs

    def which_edge(x, y, sigL, sigU, muL, muU, tol=1e-6):
        """
        Identify which HS edge a point lies on. Returns one of: 'bottom','right','top','left', or None.
        """
        # scale-aware tol
        tolx = tol * max(1.0, abs(sigU - sigL))
        toly = tol * max(1.0, abs(muU - muL))

        on_bottom = abs(y - muL) <= toly and (sigL - tolx <= x <= sigU + tolx)
        on_top    = abs(y - muU) <= toly and (sigL - tolx <= x <= sigU + tolx)
        on_left   = abs(x - sigL) <= tolx and (muL - toly <= y <= muU + toly)
        on_right  = abs(x - sigU) <= tolx and (muL - toly <= y <= muU + toly)

        # Prefer corners deterministically (bottom->right->top->left)
        if on_bottom: return "bottom"
        if on_right:  return "right"
        if on_top:    return "top"
        if on_left:   return "left"
        return None

    def boundary_param(x, y, sigL, sigU, muL, muU):
        """
        Perimeter parameter s in [0, P) going CCW starting at (sigL, muL).
        """
        W = sigU - sigL
        H = muU - muL
        P = 2*(W + H)

        e = which_edge(x, y, sigL, sigU, muL, muU, tol=1e-8)
        if e == "bottom":
            s = (x - sigL)
        elif e == "right":
            s = W + (y - muL)
        elif e == "top":
            s = W + H + (sigU - x)
        elif e == "left":
            s = W + H + W + (muU - y)
        else:
            # If slightly inside due to numerical issues, clamp to nearest edge by distance
            # (cheap fallback)
            db = abs(y - muL)
            dt = abs(y - muU)
            dl = abs(x - sigL)
            dr = abs(x - sigU)
            m = min(db, dr, dt, dl)
            if m == db:
                s = (np.clip(x, sigL, sigU) - sigL)
            elif m == dr:
                s = W + (np.clip(y, muL, muU) - muL)
            elif m == dt:
                s = W + H + (sigU - np.clip(x, sigL, sigU))
            else:
                s = W + H + W + (muU - np.clip(y, muL, muU))
        return float(s)

    def boundary_point(s, sigL, sigU, muL, muU):
        """
        Inverse of boundary_param: given s in [0,P), return (x,y) on the perimeter.
        CCW starting at (sigL, muL).
        """
        W = sigU - sigL
        H = muU - muL
        P = 2*(W + H)
        s = float(s % P)

        # bottom: (sigL,muL) -> (sigU,muL)
        if s <= W:
            return sigL + s, muL
        s -= W

        # right: (sigU,muL) -> (sigU,muU)
        if s <= H:
            return sigU, muL + s
        s -= H

        # top: (sigU,muU) -> (sigL,muU)
        if s <= W:
            return sigU - s, muU
        s -= W

        # left: (sigL,muU) -> (sigL,muL)
        return sigL, muU - s


    def boundary_path(p, q, sigL, sigU, muL, muU, n=120):
        """
        Return polyline along HS perimeter connecting p->q using the SHORTER of CCW/CW paths.
        """
        x1, y1 = p
        x2, y2 = q

        W = sigU - sigL
        H = muU - muL
        P = 2*(W + H)

        s1 = boundary_param(x1, y1, sigL, sigU, muL, muU)
        s2 = boundary_param(x2, y2, sigL, sigU, muL, muU)

        # CCW distance from s1 to s2
        d_ccw = (s2 - s1) % P
        d_cw  = (s1 - s2) % P

        if d_ccw <= d_cw:
            ss = np.linspace(s1, s1 + d_ccw, n)
        else:
            # go CW: build CCW from s2->s1 then reverse
            ss = np.linspace(s2, s2 + d_cw, n)[::-1]

        xs, ys = zip(*(boundary_point(s, sigL, sigU, muL, muU) for s in ss))
        return np.array(xs), np.array(ys)



    # ============================
    # TRANSLATION SEGMENTS (collect them)
    # ============================
    all_segs = []

    for assem in assemblers.values():
        mu_eff, sig_eff = assem.convert_to_physical(phase_props)

        in_box = (
            (mu_eff >= mu_lower_HS) & (mu_eff <= mu_upper_HS) &
            (sig_eff >= sig_lower_HS) & (sig_eff <= sig_upper_HS)
        )

        segs = extract_masked_segments(sig_eff, mu_eff, in_box, min_len=2)
        all_segs.extend(segs)

    # If nothing, fallback to plotting full rectangle or nothing
    if len(all_segs) == 0:
        print("WARNING: no translation segments inside HS box.")
    else:
        # Sort segments by perimeter position of their START point (helps with stitching)
        def seg_key(seg):
            xs, ys = seg
            return boundary_param(xs[0], ys[0], sig_lower_HS, sig_upper_HS, mu_lower_HS, mu_upper_HS)

        all_segs = sorted(all_segs, key=seg_key)

        # Plot segments and stitch gaps along HS perimeter
        lw = 4.0
        col = "blue"

        # plot first translation segment
        x0, y0 = all_segs[0]
        ax1.plot(x0, y0, color=col, linewidth=lw, label="Bounds")

        # stitch between consecutive segments
        for (x_prev, y_prev), (x_next, y_next) in zip(all_segs[:-1], all_segs[1:]):
            # end of prev, start of next
            p_end  = (x_prev[-1], y_prev[-1])
            p_next = (x_next[0],  y_next[0])

            # connect along HS boundary (shorter direction)
            bx, by = boundary_path(
                p_end, p_next,
                sig_lower_HS, sig_upper_HS, mu_lower_HS, mu_upper_HS,
                n=120
            )
            ax1.plot(bx, by, color=col, linewidth=lw)

            # now plot next translation segment
            ax1.plot(x_next, y_next, color=col, linewidth=lw)

        # OPTIONAL: close the loop (connect last -> first) if you want a closed boundary
        p_last = (all_segs[-1][0][-1], all_segs[-1][1][-1])
        p_first= (all_segs[0][0][0],   all_segs[0][1][0])
        bx, by = boundary_path(
            p_last, p_first,
            sig_lower_HS, sig_upper_HS, mu_lower_HS, mu_upper_HS,
            n=120
        )
        ax1.plot(bx, by, color=col, linewidth=lw)


    # ---- Dataset (unchanged) ----
    if datasets is not None:
        counter = 0
        for dataset in datasets:
            if counter == 0:
                ax1.scatter(dataset['conductivity'], dataset['shear_modulus'],
                            color='orange', marker='o', s=50, alpha=0.3,
                            label="Truss")
            else:
                ax1.scatter(dataset['conductivity'], dataset['shear_modulus'],
                            color='orange', marker='o', s=50, alpha=0.3)
            counter += 1
                    
    if special_datasets is not None:
        colors = ['purple', 'red', 'cyan', 'green', 'magenta']
        markers = ['P', 'D', 's', 'X', '*']
        labels = ["TPMS", "P-Bridged"]
        for idx_sd, special_dataset in enumerate(special_datasets):
            x = np.array(special_dataset['conductivity'])
            y = np.array(special_dataset['shear_modulus'])

            idx = np.argsort(x)  # sort by x
            x, y = x[idx], y[idx]
            ax1.scatter(x, y, color=colors[idx_sd], marker=markers[idx_sd], s=50, alpha=1.0, label=labels[idx_sd])
    
    # plot known morphologies
    # note these values are scaled by a small factor to ensure they corr exactly to volume fraction 0.5
    known_morphologies = {
        "Cylinders": (0.3973, 0.0514),
        "Lamellae": (0.281, 0.1716),
        "Gyroid": (0.34577, 0.2429),
        "Spheres": (0.3879, 0.12944),
        "Diamond": (0.34552, 0.2499),
        "O70": (0.3571, 0.16705),
        "Spheres Inverted": (0.1734, 0.30608),
        "Cylinders Inverted": (0.092177, 0.318627),
        "O70 Inverted": (0.26846, 0.2754)
    }
    morph_styles = {
        "Cylinders":          dict(marker='o', color='tab:blue',   inverted=False),
        "Lamellae":           dict(marker='s', color='tab:orange', inverted=False),
        "Gyroid":             dict(marker='D', color='tab:green',  inverted=False),
        "Spheres":            dict(marker='*', color='tab:red',    inverted=False),
        "Diamond":            dict(marker='d', color='tab:purple', inverted=False),
        "O70":                dict(marker='h', color='tab:brown',  inverted=False),

        "Spheres Inverted":   dict(marker='*', color='tab:red',    inverted=True),
        "Cylinders Inverted": dict(marker='o', color='tab:blue',   inverted=True),
        "O70 Inverted":       dict(marker='h', color='tab:brown',  inverted=True),
    }

    # for morph_name, (sig_val, mu_val) in known_morphologies.items():

    #     style = morph_styles[morph_name]

    #     if style["inverted"]:
    #         ax1.scatter(
    #             sig_val, mu_val,
    #             marker=style["marker"],
    #             facecolors='none',
    #             edgecolors=style["color"],
    #             linewidths=2.5,
    #             s=200,
    #             # label=morph_name
    #         )
    #     else:
    #         ax1.scatter(
    #             sig_val, mu_val,
    #             marker=style["marker"],
    #             color=style["color"],
    #             s=150,
    #             # label=morph_name
    #         )


    ax1.set_xlim(midpt_sig_lower, midpt_sig_upper)
    ax1.set_ylim(midpt_mu_lower, midpt_mu_upper)
    ax1.set_xlabel(r'$\sigma^{\star}$', fontsize=20)
    ax1.set_ylabel(r'$\mu^{\star}$', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=18)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    print(f"\n✅ Final overlay plot saved to: {out_path}")


# ============================================================
# BUILD ALL ASSEMBLERS
# ============================================================

phase_props = phase_props_holder(k1, k2, mu1, mu2, sig1, sig2, f)
datasets = []
if dataset_paths is not None:
    for dataset_path in dataset_paths:
        dataset = pd.read_csv(dataset_path) if dataset_path else None
        datasets.append(dataset)

if special_dataset_paths is not None:
    special_datasets = []
    for special_dataset_path in special_dataset_paths:
        special_dataset = pd.read_csv(special_dataset_path) if special_dataset_path else None
        special_datasets.append(special_dataset)
else:
    special_datasets = None

assemblers = {}

for bound_type, T_path in zip(bound_types, T_paths):

    helper = TranslationHelper(
        n_shear=n_shear,
        phase_props=phase_props,
        npts=10,
        upper=40,
        n_samples=50,
        alphas=(),
        seed=2387,
        calc_type=calc_type,
        bound_type=bound_type
    )

    assemblers[bound_type] = Assembler(T_path, helper, bound_type)

# ============================================================
# FINAL OVERLAY PLOT
# ============================================================

plot_all_bounds_overlay(
    assemblers,
    phase_props,
    datasets=datasets,
    special_datasets=special_datasets,
    out_path=f"{out_dir_path}/bounds_f={phase_props.f}_truss_datasets.png"
)
