# combined_train_and_plot_energy_differences.py
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# >>> TIMING ADDITIONS START
import time
from datetime import timedelta
from scipy.optimize import fmin_l_bfgs_b
# >>> TIMING ADDITIONS END


# ——— Plot defaults ———
plt.rcParams.update({
    'figure.dpi':       300,
    'font.family':      'serif',
    'font.size':        16,
    'axes.titlesize':   20,
    'axes.labelsize':   18,
    'xtick.labelsize':  16,
    'ytick.labelsize':  16,
    'legend.fontsize':  12,
    'axes.linewidth':   1,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})

# ——— Setup ———
base_dir = os.getcwd()
csv_path = os.path.join(base_dir, "dataset.csv")
plots_dir = os.path.join(base_dir, "plots_pred_vs_actual")
os.makedirs(plots_dir, exist_ok=True)

df = pd.read_csv(csv_path)
feature_cols = ["chiAB", "chiAC", "chiBC", "fA", "fC"]
morphologies = ["cylinders", "spheres", "gyroid", "lamella", "O70", "diamond"]

# Consistent random seed/split
rng_state = 88
all_rows = np.arange(len(df))
rows_tr, rows_te = train_test_split(all_rows, test_size=0.10, random_state=rng_state)
row_tr_mask = np.zeros(len(df), dtype=bool); row_tr_mask[rows_tr] = True
row_te_mask = ~row_tr_mask


# >>> TIMING ADDITIONS START
class _LBFGSTimer:
    """Tracks per-call (per restart) timing and iteration counts."""
    def __init__(self):
        self.restart_counter = 0  # counts how many times optimizer is invoked

    def __call__(self, obj_func, initial_theta, bounds):
        self.restart_counter += 1
        r = self.restart_counter
        start = time.time()
        last_print = start
        iters = {'n': 0}

        def _callback(xk):
            nonlocal last_print   # ← moved to the top
            iters['n'] += 1
            now = time.time()
            # throttle prints to about once per second
            if now - last_print >= 1.0 or iters['n'] <= 2:
                f, g = obj_func(xk)  # evaluate to display current objective
                elapsed = now - start
                avg_it = elapsed / max(1, iters['n'])
                with open('output.txt', 'a') as file:
                    print(f"[L-BFGS-B | restart {r:02d} | iter {iters['n']:04d}] "
                        f"f={f:.6g} | elapsed={timedelta(seconds=int(elapsed))} "
                        f"| avg/iter={avg_it:.3f}s", file=file)
                last_print = now


        # Run the same optimizer sklearn uses, but with our callback
        x_opt, f_min, info = fmin_l_bfgs_b(
            func=lambda th: obj_func(th),  # returns (f, grad)
            x0=initial_theta,
            fprime=None,                   # provided by obj_func
            bounds=bounds,
            callback=_callback,
        )
        total = time.time() - start
        with open('output.txt', 'a') as file:
            print(f"[L-BFGS-B | restart {r:02d}] done in {timedelta(seconds=int(total))}, "
              f"iters={iters['n']}, f_min={f_min:.6g}", file=file)
        # sklearn expects (theta_opt, func_min)
        return x_opt, f_min

# single shared instance so we can number restarts
_TIMED_OPTIMIZER = _LBFGSTimer()
# >>> TIMING ADDITIONS END


# ——— Train GPs per morphology on TRAIN rows only ———
def build_kernel(input_dim):
    # ARD RBF with constant scale
    return C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(input_dim),
                                     length_scale_bounds=(1e-2, 1e2))

gp_models = {}
with open('output.txt', 'a') as file:
    print("— Training Gaussian Processes on TRAIN rows only —", file=file)
for morph in morphologies:
    y_col = f"FE_{morph}"
    # Only rows that have this FE and belong to TRAIN split
    m_tr = row_tr_mask & df[y_col].notna()
    n_tr = int(m_tr.sum())
    if n_tr < 2:
        with open('output.txt', 'a') as file:
            print(f"Skipping {morph}: insufficient TRAIN data ({n_tr} samples)", file=file)
        continue
    with open('output.txt', 'a') as file:
        print(f"Training GP for {morph} with {n_tr} samples...", file=file)

    X_tr = df.loc[m_tr, feature_cols].values
    y_tr = df.loc[m_tr, y_col].values

    kernel = build_kernel(len(feature_cols))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,
        normalize_y=True,
        n_restarts_optimizer=20,
        random_state=rng_state,
        optimizer=_TIMED_OPTIMIZER  # <<< timing-aware optimizer
    )
    t0 = time.time()
    gp.fit(X_tr, y_tr)
    t1 = time.time()
    with open('output.txt', 'a') as file:
        print(f"[{morph}] total fit time: {timedelta(seconds=int(t1 - t0))}", file=file)

    gp_models[morph] = gp

    # Quick TRAIN/TEST MAE for this morphology's FE
    # TRAIN MAE
    mae_train = mean_absolute_error(y_tr, gp.predict(X_tr)) if len(y_tr) else np.nan
    # TEST MAE: only rows in TEST with FE available
    m_te = row_te_mask & df[y_col].notna()
    X_te = df.loc[m_te, feature_cols].values
    y_te = df.loc[m_te, y_col].values
    mae_test = mean_absolute_error(y_te, gp.predict(X_te)) if len(y_te) else np.nan

    with open('output.txt', 'a') as file:
        print(f"{morph:>9}: FE-MAE train={mae_train:.4f} | test={mae_test:.4f}", file=file)

    # Save model
    out_path = os.path.join(base_dir, f"gp_{morph}.pkl")
    joblib.dump(gp, out_path)

# ——————————————————————————————————————————————————————————————————
#           NEW: All-pairs ΔF, MAE matrices, unified plots
# ——————————————————————————————————————————————————————————————————

def compute_deltaF_for_pair(m1: str, m2: str, is_train: bool):
    """
    ΔF = FE_m1 - FE_m2 for rows where both FEs exist in the split.
    Returns (actual ΔF, predicted ΔF).
    """
    if (m1 not in gp_models) or (m2 not in gp_models):
        return np.array([]), np.array([])

    split_mask = row_tr_mask if is_train else row_te_mask
    col1, col2 = f"FE_{m1}", f"FE_{m2}"
    valid = split_mask & df[col1].notna() & df[col2].notna()
    if not np.any(valid):
        return np.array([]), np.array([])

    X = df.loc[valid, feature_cols].values
    actual = (df.loc[valid, col1].values - df.loc[valid, col2].values)
    pred = gp_models[m1].predict(X) - gp_models[m2].predict(X)
    return actual, pred

def mae_matrix(is_train: bool):
    """
    Build a symmetric matrix M where M[i,j] = MAE(ΔF_{mi}-ΔF_{mj})
    (i!=j), using rows that have both FEs available and both GPs trained.
    """
    n = len(morphologies)
    M = np.full((n, n), np.nan, dtype=float)
    for i, j in combinations(range(n), 2):
        m1, m2 = morphologies[i], morphologies[j]
        a, p = compute_deltaF_for_pair(m1, m2, is_train=is_train)
        if len(a) > 0:
            mae = mean_absolute_error(a, p)
            M[i, j] = mae
            M[j, i] = mae
        # else stays NaN
    np.fill_diagonal(M, 0.0)
    return M

def print_and_save_matrix(M: np.ndarray, split_name: str, fname: str):
    # Nice header + fixed-width formatting
    header = [""] + morphologies
    widths = [max(10, max(len(h) for h in header))]
    widths += [max(10, len(m)) for m in morphologies]

    def fmt_row(cells):
        return " ".join(str(c).rjust(w) for c, w in zip(cells, widths))

    # Build lines
    lines = []
    lines.append(f"— {split_name} ΔF MAE matrix (|ΔF_{'{mi}'}−ΔF_{'{mj}'}|) —")
    lines.append(fmt_row(header))
    for i, mi in enumerate(morphologies):
        row_vals = []
        for j in range(len(morphologies)):
            v = M[i, j]
            cell = f"{v:.4f}" if np.isfinite(v) else "nan"
            row_vals.append(cell)
        lines.append(fmt_row([mi] + row_vals))

    text = "\n".join(lines)
    with open('output.txt', 'a') as file:
        print("\n" + text + "\n", file=file)

    # Save to text file in CWD
    out_path = os.path.join(base_dir, fname)
    with open(out_path, "w") as f:
        f.write(text + "\n")
    with open('output.txt', 'a') as file:
        print(f"Saved MAE matrix -> {out_path}", file=file)

def average_offdiag_mae(M: np.ndarray):
    # Upper triangle without diagonal
    n = M.shape[0]
    mask = ~np.isnan(M)
    offdiag = []
    for i in range(n):
        for j in range(i+1, n):
            if mask[i, j] and i != j:
                offdiag.append(M[i, j])
    if len(offdiag) == 0:
        return np.nan
    return float(np.mean(offdiag))

def gather_all_points(is_train: bool):
    """
    Collect ALL per-pair points into single arrays for plotting.
    Returns (all_actual, all_pred).
    """
    all_a, all_p = [], []
    for i, j in combinations(range(len(morphologies)), 2):
        m1, m2 = morphologies[i], morphologies[j]
        a, p = compute_deltaF_for_pair(m1, m2, is_train=is_train)
        if len(a) > 0:
            all_a.append(a)
            all_p.append(p)
    if len(all_a) == 0:
        return np.array([]), np.array([])
    return np.concatenate(all_a), np.concatenate(all_p)

def style_ax(ax, mn, mx):
    ax.set_aspect('equal', 'box')
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('k')
        spine.set_linewidth(1)
    ax.plot([mn, mx], [mn, mx], '--', color='k', linewidth=1)

def unified_plot(all_actual, all_pred, title, out_path, overall_mae=None):
    if len(all_actual) == 0:
        with open('output.txt', 'a') as file:
            print(f"Skipping plot (no points): {title}", file=file)
        return
    gmin = min(all_actual.min(), all_pred.min())
    gmax = max(all_actual.max(), all_pred.max())
    buf = 0.02 * max(gmax - gmin, 1.0)
    mn, mx = gmin - buf, gmax + buf

    fig, ax = plt.subplots(figsize=(7,7))
    # — unified style: ONE color, ONE marker —
    ax.scatter(
        all_actual, all_pred,
        c='C0', marker='o', s=50, alpha=0.8,
        edgecolor='k', linewidth=0.5,
        label=None
    )
    style_ax(ax, mn, mx)
    ax.set_xlabel(r'$\Delta F_{\mathrm{actual}}$')
    ax.set_ylabel(r'$\Delta F_{\mathrm{predicted}}$')
    ttl = title if overall_mae is None else f"{title}\nOverall MAE = {overall_mae:.4f}"
    ax.set_title(ttl, loc='center', pad=20)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    with open('output.txt', 'a') as file:
        print(f"Saved -> {out_path}", file=file)

# ——— Build matrices ———
M_train = mae_matrix(is_train=True)
M_test  = mae_matrix(is_train=False)

# ——— Print + save matrices ———
print_and_save_matrix(M_train, "TRAIN", "mae_matrix_train.txt")
print_and_save_matrix(M_test,  "TEST",  "mae_matrix_test.txt")

# ——— Average MAE over all pairs (exclude diagonal) ———
avg_train = average_offdiag_mae(M_train)
avg_test  = average_offdiag_mae(M_test)
with open('output.txt', 'a') as file:
    print(f"Average off-diagonal ΔF MAE — TRAIN: {avg_train:.4f}" if np.isfinite(avg_train) else "Average off-diagonal ΔF MAE — TRAIN: nan", file=file)
    print(f"Average off-diagonal ΔF MAE — TEST:  {avg_test:.4f}"  if np.isfinite(avg_test)  else "Average off-diagonal ΔF MAE — TEST:  nan", file=file)

# ——— Unified plots (all pairs overlaid with same style) ———
a_tr, p_tr = gather_all_points(is_train=True)
a_te, p_te = gather_all_points(is_train=False)

unified_plot(
    a_tr, p_tr,
    title="Train: Energy Differences (ΔF) — All Pairings",
    out_path=os.path.join(plots_dir, 'energy_diff_train_ALL_PAIRS.png'),
    overall_mae=(mean_absolute_error(a_tr, p_tr) if len(a_tr) else None)
)

unified_plot(
    a_te, p_te,
    title="Test: Energy Differences (ΔF) — All Pairings",
    out_path=os.path.join(plots_dir, 'energy_diff_test_ALL_PAIRS.png'),
    overall_mae=(mean_absolute_error(a_te, p_te) if len(a_te) else None)
)

with open('output.txt', 'a') as file:
    print("\nDone.", file=file)
