# phase_slice_explorer.py
import os
import io
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgb

import torch
import torch.nn as nn
import streamlit as st

from sklearn.gaussian_process import GaussianProcessRegressor  # for type hints only

torch.set_grad_enabled(False)

# ---------- GLOBALS ----------
ALL_MORPHS = ["cylinders", "spheres", "gyroid", "lamella", "O70", "diamond"]
PHASE_COLORS = {
    "cylinders": "#1f77b4",
    "spheres":   "#2ca02c",
    "gyroid":    "#7f7f7f",
    "lamella":   "#ff7f0e",
    "O70":       "#9467bd",
    "diamond":   "#8c564b",
}

# ---------- MLP definition (must match training) ----------
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 512),    nn.ReLU(),
            nn.Linear(512, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        return self.net(x)

# --- shim so joblib can unpickle GP models trained with a custom optimizer ---
class _LBFGSTimer:
    """Stub for unpickling; optimizer isn't used at prediction time."""
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        # Not used by GaussianProcessRegressor.predict; safe stub.
        # If something ever tries to call it, fall back to SciPy's default.
        from scipy.optimize import fmin_l_bfgs_b
        return fmin_l_bfgs_b(*args, **kwargs)


# ---------- Colors and helpers ----------
def lighten(rgb_hex, amt=0.25):
    rgb = np.array(to_rgb(rgb_hex))
    return tuple(1.0 - (1.0 - rgb) * (1.0 - amt))

def darken(rgb_hex, amt=0.05):
    rgb = np.array(to_rgb(rgb_hex))
    return tuple(rgb * (1.0 - amt))

def make_legend_handles(morphs):
    return [
        mpatches.Patch(facecolor=PHASE_COLORS[m], edgecolor='k', label=m.capitalize())
        for m in morphs
    ]

def make_shaded_colormap(morphs):
    cmap_list = []
    for m in morphs:
        base = PHASE_COLORS[m]
        cmap_list.append(lighten(base))
        cmap_list.append(darken(base))
    cmap = ListedColormap(cmap_list)
    bounds = np.arange(2 * len(morphs) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, bounds, norm

def make_plain_colormap(morphs):
    cmap_list = [PHASE_COLORS[m] for m in morphs]
    cmap = ListedColormap(cmap_list)
    bounds = np.arange(len(morphs) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, bounds, norm

# ---------- Model discovery & loading ----------
def list_polymer_families(models_root: str = "models"):
    if not os.path.isdir(models_root):
        return []
    opts = [d for d in os.listdir(models_root)
            if os.path.isdir(os.path.join(models_root, d)) and not d.startswith(".")]
    # If there are no subdirs, allow selecting the root as a fallback
    return opts if opts else ["(root)"]

def _find_model_file(folder: str, morph: str):
    """Find a model file for a morphology inside folder, preferring GP (.pkl) over MLP (.joblib)."""
    candidates = os.listdir(folder) if os.path.isdir(folder) else []
    # Prefer exact conventional names
    preferred = [
        (f"gp_{morph}.pkl", "gp"),
        (f"torch_mlp_{morph}.joblib", "mlp"),
    ]
    for fname, kind in preferred:
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            return path, kind
    # Otherwise search by substring + extension
    gp_matches  = [f for f in candidates if f.endswith(".pkl")    and morph in f]
    mlp_matches = [f for f in candidates if f.endswith(".joblib") and morph in f]
    if gp_matches:
        return os.path.join(folder, gp_matches[0]), "gp"
    if mlp_matches:
        return os.path.join(folder, mlp_matches[0]), "mlp"
    return None, None

@st.cache_resource(show_spinner=False)
def load_bundles(models_root: str, polymer: str, morph_list):
    """Return dict: {morph: {'type':'gp','model':GP} or {'type':'mlp','state_dict':..., 'x_scaler':..., 'y_scaler':..., 'in_dim':...}}"""
    folder = models_root if polymer == "(root)" else os.path.join(models_root, polymer)
    bundles = {}
    notes = []
    for m in morph_list:
        path, kind = _find_model_file(folder, m)
        if path is None:
            continue
        if kind == "gp":
            model = joblib.load(path)  # sklearn GaussianProcessRegressor
            bundles[m] = {"type": "gp", "model": model}
            # check if MLP also exists to inform user about preference
            alt_path, alt_kind = None, None
            _ = _find_model_file(folder, m)
        elif kind == "mlp":
            bundle = joblib.load(path)  # contains scalers + state_dict + in_dim
            bundles[m] = {"type": "mlp", **bundle}
        else:
            continue

        # If both are present, prefer GP and note it
        if kind == "mlp":
            # check if a GP also exists
            gp_path, _ = _find_model_file(folder, m)
            # above returns first match; ensure it's GP:
            if gp_path and gp_path.endswith(".pkl"):
                notes.append(f"For '{m}', found both GP and MLP — using GP (preferred).")
                gp_model = joblib.load(gp_path)
                bundles[m] = {"type": "gp", "model": gp_model}
    return bundles, notes, folder

def predict_fe_for_morph(bundle_entry, X: np.ndarray, batch: int = 20000):
    """Predict FE for a single morphology using either GP or MLP bundle.
    Returns (mu, sigma) where sigma=0 for MLPs.
    """
    if bundle_entry["type"] == "gp":
        gp: GaussianProcessRegressor = bundle_entry["model"]
        mu, sigma = gp.predict(X.astype(np.float64), return_std=True)
        return mu, sigma
    else:
        # MLP path with scalers
        xsc = bundle_entry["x_scaler"]
        ysc = bundle_entry["y_scaler"]
        in_dim = int(bundle_entry["in_dim"])
        model = MLP(in_dim=in_dim)
        model.load_state_dict(bundle_entry["state_dict"])
        model.eval()

        Xs = xsc.transform(X).astype(np.float32)
        outs = []
        for i in range(0, Xs.shape[0], batch):
            xb = torch.from_numpy(Xs[i:i+batch])
            yb_std = model(xb).numpy()
            outs.append(yb_std)
        y_std = np.vstack(outs)
        mu = ysc.inverse_transform(y_std)[:, 0]
        sigma = np.zeros_like(mu)  # no epistemic σ for MLP
        return mu, sigma

@st.cache_data(show_spinner=False)
def predict_all_FE(_bundles, morphs, pts: np.ndarray):
    """Return (FE_mu, FE_std) each of shape (N_points, len(morphs))."""
    mu_cols, std_cols = [], []
    for m in morphs:
        mu, sigma = predict_fe_for_morph(_bundles[m], pts)
        mu_cols.append(mu)
        std_cols.append(sigma)
    FE_mu  = np.column_stack(mu_cols)
    FE_std = np.column_stack(std_cols)
    return FE_mu, FE_std


# ---------- UI ----------
st.set_page_config(page_title="Phase Slice Explorer", layout="wide")
st.title("Phase Slice Explorer (valid domain only)")

param_ranges = {
    "chiAB": (10.0, 80.0),
    "chiAC": (10.0, 80.0),
    "chiBC": (10.0, 40.0),
    "fA":    (0.10, 0.50),
    "fC":    (0.30, 0.50),
}
param_order = ["chiAB", "chiAC", "chiBC", "fA", "fC"]
# Display labels (plot only) — keep param_order/internal names unchanged
PARAM_LABEL = {
    "chiAB": r"$\chi_{AB}N$",
    "chiAC": r"$\chi_{AC}N$",
    "chiBC": r"$\chi_{BC}N$",
    "fA":    r"$f_A$",
    "fC":    r"$f_C$",
}


with st.sidebar:
    st.header("Controls")

    # --- NEW: Polymer family selector ---
    polymer_families = list_polymer_families("models")
    if not polymer_families:
        st.error("No models found. Please add models under a 'models' folder.")
        st.stop()
    polymer = st.selectbox("Polymer family", polymer_families, index=0,
                           help="Select a subfolder under 'models' (e.g., ABC, ABCBA).")

    axes_choice = st.multiselect(
        "Choose 2D axes (exactly two):",
        options=param_order,
        default=["chiAB", "fA"],
        max_selections=2
    )
    if len(axes_choice) != 2:
        st.warning("Please select exactly two axes.")
        st.stop()

    x_param, y_param = axes_choice
    other_params = [p for p in param_order if p not in axes_choice]

    def slider_for_range(name):
        lo, hi = param_ranges[name]
        return st.slider(f"{name} range", lo, hi, (lo, hi))

    x_range = slider_for_range(x_param)
    y_range = slider_for_range(y_param)

    nx = st.slider("X resolution (grid)", 50, 400, 200, step=10)
    ny = st.slider("Y resolution (grid)", 50, 400, 200, step=10)

    fixed_values = {}
    for p in other_params:
        lo, hi = param_ranges[p]
        fixed_values[p] = st.slider(f"{p} (fixed)", lo, hi, float(np.mean([lo, hi])))

    # --- Clamp to valid region ---
    if "chiBC" in fixed_values:
        bc = fixed_values["chiBC"]
        msgs = []
        for key in ("chiAB", "chiAC"):
            if key in fixed_values and fixed_values[key] < bc:
                fixed_values[key] = float(bc)
                msgs.append(f"Clamped fixed {key} to chiBC={bc:.3g} to enforce validity ({key} ≥ chiBC).")
        for m in msgs:
            st.info(m)

    shading_on = st.checkbox("Enable stability shading (dark = stable)", value=True)
    threshold = st.number_input("Stability threshold ΔE", value=0.05, step=0.01, min_value=0.0, format="%.3f")

    show_boundaries = st.checkbox("Show phase boundaries", value=True)
    title_on = st.checkbox("Show title", value=False)

# --- Load bundles for the selected polymer ---
bundles, notes, used_folder = load_bundles("models", polymer, ALL_MORPHS)
active_morphs = [m for m in ALL_MORPHS if m in bundles]

if len(active_morphs) < 2:
    st.error(f"No sufficient models found in '{used_folder}'. Need at least 2 morphologies.")
    st.stop()

if notes:
    for n in notes:
        st.info(n)

# --- Adjust axis range if chiBC is fixed and axis is chiAB/chiAC ---
def adjusted_range(param, rng, fixed_vals):
    lo, hi = rng
    if param in ("chiAB", "chiAC") and ("chiBC" in fixed_vals):
        bc = fixed_vals["chiBC"]
        lo = max(lo, bc)
    return (lo, hi)

x_range = adjusted_range(x_param, x_range, fixed_values)
y_range = adjusted_range(y_param, y_range, fixed_values)

# Build grid
x_vals = np.linspace(x_range[0], x_range[1], nx)
y_vals = np.linspace(y_range[0], y_range[1], ny)
X_mesh, Y_mesh = np.meshgrid(x_vals, y_vals, indexing='xy')

# Build points for model input
pts = np.zeros((X_mesh.size, len(param_order)), dtype=float)
for i, name in enumerate(param_order):
    if name == x_param:
        pts[:, i] = X_mesh.ravel()
    elif name == y_param:
        pts[:, i] = Y_mesh.ravel()
    else:
        pts[:, i] = fixed_values[name]

# Predict energies for available morphs (means & stds)
FE_mu, FE_std = predict_all_FE(bundles, active_morphs, pts)

# Phase decision uses means
phase_argmin = np.argmin(FE_mu, axis=1)

# (Kept for completeness; not used for shading now)
two_smallest = np.partition(FE_mu, 1, axis=1)[:, :2]
margins = two_smallest[:, 1] - two_smallest[:, 0]


if shading_on:
    cmap, bounds, norm = make_shaded_colormap(active_morphs)

    # Uncertainty-aware stability:
    # dark if (mu_winner + sigma_winner) + threshold <= min_{others} (mu_other - sigma_other)
    N, M = FE_mu.shape
    winner = phase_argmin
    rows = np.arange(N)

    mu_w = FE_mu[rows, winner]
    sd_w = FE_std[rows, winner]

    # competitor scores = mu - sigma; exclude the winner by setting +inf on its column
    comp_scores = (FE_mu - FE_std).copy()
    comp_scores[rows, winner] = np.inf
    min_comp = np.min(comp_scores, axis=1)

    stable_flag = ((min_comp - (mu_w + sd_w)) >= threshold).astype(int)

    class_idx = (2 * phase_argmin + stable_flag).reshape(X_mesh.shape)
    boundary_field = phase_argmin.reshape(X_mesh.shape)
else:
    cmap, bounds, norm = make_plain_colormap(active_morphs)
    class_idx = phase_argmin.reshape(X_mesh.shape)
    boundary_field = class_idx

def fixed_param_tag(fixed_values, precision=2):
    parts = []
    for k, v in fixed_values.items():
        parts.append(f"{k}={v:.{precision}f}")
    return "_".join(parts)


# ---------- Plot ----------
plt.rcParams.update({
    'figure.dpi':       400,
    'font.size':        12,
    'axes.titlesize':   18,
    'axes.labelsize':   26,
    'xtick.labelsize':  22,
    'ytick.labelsize':  22,
    'legend.fontsize':  20,
    'axes.linewidth':   1,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
})

fig, ax = plt.subplots(figsize=(8, 7))
ax.contourf(
    x_vals, y_vals, class_idx,
    levels=bounds, cmap=cmap, norm=norm, alpha=0.95
)

if show_boundaries:
    ax.contour(
        x_vals, y_vals, boundary_field,
        levels=np.arange(len(active_morphs)) - 0.5,
        colors='white', linewidths=0.7
    )

ax.set_xlabel(PARAM_LABEL.get(x_param, x_param))
ax.set_ylabel(PARAM_LABEL.get(y_param, y_param))

if title_on:
    others_txt = ", ".join([f"{p}={fixed_values[p]:.3g}" for p in param_order if p not in (x_param, y_param)])
    ax.set_title(f"{polymer} — {x_param} vs {y_param}\nFixed: {others_txt}")

leg_title = "Phases" + (f"\n(dark = ΔE ≥ {threshold:.3f})" if shading_on else "")
legend_handles = make_legend_handles(active_morphs)

def make_legend_figure(morphs, threshold, shading_on):
    handles = make_legend_handles(morphs)

    fig_leg, ax_leg = plt.subplots(figsize=(2.5, 3))
    ax_leg.axis("off")

    legend = ax_leg.legend(
        handles=handles,
        loc="center",
        frameon=True,
        title=None
    )

    return fig_leg


plt.tight_layout()
fig_leg = make_legend_figure(active_morphs, threshold, shading_on)

col1, col2 = st.columns([4, 1])

with col1:
    st.pyplot(fig, use_container_width=False)

with col2:
    st.pyplot(fig_leg, use_container_width=False)

tag = fixed_param_tag(fixed_values, precision=2)
base_name = f"{polymer}_{x_param}_vs_{y_param}_{tag}"

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
st.download_button("Download PNG", data=buf.getvalue(), file_name=f"{base_name}.png", mime="image/png")

buf_leg = io.BytesIO()
fig_leg.savefig(buf_leg, format="png", dpi=300, bbox_inches="tight")

st.download_button(
    "Download Legend PNG",
    data=buf_leg.getvalue(),
    file_name=f"legend.png",
    mime="image/png"
)


# Small footer about where models came from
st.caption(f"Models loaded from: {used_folder}. Active morphologies: {', '.join(active_morphs)}.")