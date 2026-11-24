import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os, numpy as np, pyvista as pv
from skimage import measure

# -----------------------------------------------------------------
# Projection Plotting routine (remains CPU‑based with matplotlib)
# -----------------------------------------------------------------
def projection(DIR, SOL, POLY, GRID):
    """
    rhos : list[Tensor] - each Tensor has shape gridShape
    Shows  xy-slice (z=0), xz-slice (y=0) and yz-slice (x=0) for every phase.
    For 2D data (len(gridShape)==2) it falls back to the old single-slice plot.
    """
    numPhases = POLY.nBlocks

    # ---------- basic styling ----------
    phase_colors = ['red', 'cyan', 'orange', 'lime', 'violet']
    if numPhases > len(phase_colors):
        raise ValueError("Too many phases for color palette.")

    colormaps = [
        mcolors.LinearSegmentedColormap.from_list(
            f"black_{color}", ["black", color]
        ) for color in phase_colors[:numPhases]
    ]

    # ---------- figure / axes ----------
    # 3 slices per phase → 3 rows × N columns
    fig, axes = plt.subplots(
        3, numPhases,
        figsize=(6*numPhases, 18),
        constrained_layout=True,
        sharex='col', sharey='row'
    )
    if numPhases == 1:
        axes = axes.reshape(3, 1)      # keep indexing uniform

    # ---------- plotting ----------
    x_min = y_min = z_min = 0
    x_max = y_max = z_max = 1

    for p in range(numPhases):
        data = SOL.rho[p].detach().cpu().numpy()
        cmap = colormaps[p]
        # (1) xy @ z=0
        ax_xy = axes[0, p]
        im_xy = ax_xy.imshow(
            data[:, :, 0].T,
            extent=[x_min, x_max, y_min, y_max],
            origin='lower',  cmap=cmap,  aspect='auto'
        )
        ax_xy.set_title(f'ρ{p}  (xy, z=0)')
        ax_xy.set_xlabel('x');  ax_xy.set_ylabel('y')
        fig.colorbar(im_xy, ax=ax_xy, shrink=0.7,
                        label=f'ρ{p}(x,y,0)')

        # (2) xz @ y=0
        ax_xz = axes[1, p]
        im_xz = ax_xz.imshow(
            data[:, 0, :].T,
            extent=[x_min, x_max, z_min, z_max],
            origin='lower',  cmap=cmap,  aspect='auto'
        )
        ax_xz.set_title(f'ρ{p}  (xz, y=0)')
        ax_xz.set_xlabel('x');  ax_xz.set_ylabel('z')
        fig.colorbar(im_xz, ax=ax_xz, shrink=0.7,
                        label=f'ρ{p}(x,0,z)')

        # (3) yz @ x=0
        ax_yz = axes[2, p]
        im_yz = ax_yz.imshow(
            data[0, :, :].T,
            extent=[y_min, y_max, z_min, z_max],
            origin='lower',  cmap=cmap,  aspect='auto'
        )
        ax_yz.set_title(f'ρ{p}  (yz, x=0)')
        ax_yz.set_xlabel('y');  ax_yz.set_ylabel('z')
        fig.colorbar(im_yz, ax=ax_yz, shrink=0.7,
                        label=f'ρ{p}(0,y,z)')

    # ---------- save ----------
    os.makedirs(DIR.projDir, exist_ok=True)
    out = os.path.join(DIR.projDir, f"projection_iter{SOL.iter}.png")
    plt.savefig(out, dpi=300)
    plt.close(fig)

def isosurface(DIR, SOL, POLY, GRID,
        tau=0.7, rel_tau=0.5,
        colour_cycle=('red', 'cyan', 'orange', 'lime', 'violet'),
        view='iso'):
    """
    Saves rhos_<iter>_phase<i>.png for every phase that yields a valid
    iso-surface.  τ used is printed on the picture.

    Parameters
    ----------
    tau      : preferred iso-level
    rel_tau  : fallback fraction if τ is outside [min,max] of that phase
    view     : camera alias ('iso', '+x', etc.) or PyVista tuple
    """
    # ---- headless handling -------------------------------------------------
    if os.environ.get("DISPLAY", "") == "":
        pv.start_xvfb()                             # virtual framebuffer

    spacing = float(1 / GRID.nGrid)              # assume cubic voxels
    os.makedirs(DIR.isoDir, exist_ok=True)

    for idx, rho in enumerate(SOL.rho):
        arr = rho.detach().cpu().numpy()
        vmin, vmax = float(arr.min()), float(arr.max())

        if vmax == vmin:
            print(f"[iso-surf] phase {idx}: uniform field - skipped")
            continue

        level = tau if vmin <= tau <= vmax else vmin + rel_tau * (vmax - vmin)

        # ---- extract surface ----------------------------------------------
        try:
            verts, faces, *_ = measure.marching_cubes(
                arr, level=level, spacing=(spacing,) * 3
            )
        except RuntimeError:
            print(f"[iso-surf] phase {idx}: no surface at τ={level:.3f} - skipped")
            continue

        if faces.size == 0:
            print(f"[iso-surf] phase {idx}: empty surface - skipped")
            continue

        faces_pv = np.hstack(
            [np.full((faces.shape[0], 1), 3), faces]
        ).astype(np.int32)
        mesh = pv.PolyData(verts, faces_pv)

        # ---- build off‑screen plot ----------------------------------------
        p = pv.Plotter(off_screen=True, window_size=(900, 900))
        p.set_background('white')
        p.add_mesh(mesh,
                   color=colour_cycle[idx % len(colour_cycle)],
                   opacity=0.65,
                   smooth_shading=True)
        p.show_axes()
        p.camera_position = view
        p.add_text(f"level = {level:.3f}", position='upper_left',
                   font_size=32, color='black')

        fname = os.path.join(DIR.isoDir, f"isosurface_iter{SOL.iter}_phase{idx}.png")
        p.screenshot(fname)
        p.close()