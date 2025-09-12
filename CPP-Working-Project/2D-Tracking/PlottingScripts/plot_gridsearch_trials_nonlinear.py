#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import h5py
import os
import yaml
from matplotlib import colors
import argparse

PLOTS_DIR = "../2D-Tracking/plots"
DEFAULT_H5 = "../2D-Tracking/Saved_Data/2D_gridsearch_nonlinear_trials_nis4.h5"
DEFAULT_SCENARIO = "../scenario_nonlinear.yaml"


def load_nonlinear_grid_trials(h5_path=DEFAULT_H5):
    if not os.path.exists(h5_path):
        print(f"Error: Grid-search trials file not found: {h5_path}")
        print("Please run GridSearch_Tracking_Nonlinear first to generate results.")
        return None, None, None
    with h5py.File(h5_path, 'r') as f:
        q = f['q_values'][:]
        r = f['r_values'][:]
        c = f['objective_values'][:]
    return q, r, c


def infer_grid(q, r, c):
    valid = np.isfinite(c) & (c < 1e5)
    q = q[valid]; r = r[valid]; c = c[valid]

    uq = np.unique(q)
    ur = np.unique(r)

    if uq.size * ur.size == q.size:
        q_idx = np.searchsorted(uq, q)
        r_idx = np.searchsorted(ur, r)
        grid = np.full((uq.size, ur.size), np.nan)
        grid[q_idx, r_idx] = c
        return uq, ur, grid
    else:
        return None, None, None

# Add log-normalization helpers

def _compute_log_norm_from_array(values):
    eps = 1e-12
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return colors.LogNorm(vmin=eps, vmax=1.0)
    pos = vals[vals > 0]
    if pos.size == 0:
        vmin = eps
    else:
        vmin = max(eps, np.nanmin(pos))
    vmax = max(vmin * (1.0 + 1e-6), np.nanmax(vals))
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def _compute_log_norm_from_grid(grid):
    eps = 1e-12
    g = np.asarray(grid)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return colors.LogNorm(vmin=eps, vmax=1.0)
    pos = g[g > 0]
    if pos.size == 0:
        vmin = eps
    else:
        vmin = max(eps, np.nanmin(pos))
    vmax = max(vmin * (1.0 + 1e-6), np.nanmax(g))
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def plot_3d_scatter(q, r, c, elev=30.0, azim=45.0, title_suffix=""):
    valid = np.isfinite(c) & (c < 1e5)
    q = q[valid]; r = r[valid]; c = c[valid]

    norm = _compute_log_norm_from_array(c)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(q, r, c, c=c, cmap='viridis', s=30, edgecolor='k', norm=norm, alpha=0.001)
    # Add a surface (sheet) over the points
    mappable = None
    try:
        uq, ur, grid = infer_grid(q, r, c)
        if grid is not None:
            X, Y = np.meshgrid(uq, ur, indexing='ij')
            G = np.ma.masked_invalid(grid)
            mappable = ax.plot_surface(X, Y, G, cmap='viridis', norm=norm, alpha=0.7, linewidth=0, antialiased=True)
        else:
            mappable = ax.plot_trisurf(q, r, c, cmap='viridis', norm=norm, alpha=0.7, linewidth=0.2, antialiased=True)
    except Exception as e:
        print(f"Surface plot failed: {e}")

    min_idx = np.argmin(c)
    ax.scatter([q[min_idx]], [r[min_idx]], [c[min_idx]], color='red', s=100, marker='*', label='Best')
    ax.set_xlabel('Q (Process Noise Intensity)')
    ax.set_ylabel('R (Measurement Noise Variance)')
    ax.set_zlabel('C (Consistency Metric)')
    ax.set_title(f'Nonlinear Grid Search Trials (3D){title_suffix}')
    ax.view_init(elev=elev, azim=azim)
    cbar = plt.colorbar(mappable if mappable is not None else sc, pad=0.1)
    cbar.set_label('C (Consistency Metric, log scale)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, 'nonlinear_gridsearch_trials_3d.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


def plot_heatmap(q, r, c, yaml_path=DEFAULT_SCENARIO, title_suffix=""):
    # Read YAML ranges
    q_min = q_max = R_min = R_max = None
    try:
        with open(yaml_path, 'r') as yf:
            cfg = yaml.safe_load(yf)
        if 'grid_search' in cfg and cfg['grid_search']:
            gs = cfg['grid_search']
            q_min = float(gs.get('q_min', np.min(q)))
            q_max = float(gs.get('q_max', np.max(q)))
            R_min = float(gs.get('R_min', np.min(r)))
            R_max = float(gs.get('R_max', np.max(r)))
        else:
            q_min = float(cfg['parameters'][0]['lower_bound'])
            q_max = float(cfg['parameters'][0]['upper_bound'])
            R_min = float(cfg['parameters'][1]['lower_bound'])
            R_max = float(cfg['parameters'][1]['upper_bound'])
    except Exception as e:
        print(f"YAML range read failed, using data min/max: {e}")
        q_min, q_max = float(np.min(q)), float(np.max(q))
        R_min, R_max = float(np.min(r)), float(np.max(r))

    uq, ur, grid = infer_grid(q, r, c)
    q_best = None; r_best = None; c_best = None
    if grid is None:
        print("Grid could not be reshaped; plotting scatter instead.")
        norm = _compute_log_norm_from_array(c)
        plt.figure(figsize=(8, 6))
        sc = plt.scatter(q, r, c=c, cmap='viridis', s=20, edgecolor='k', norm=norm)
        plt.colorbar(sc, label='C (Consistency Metric, log scale)')
        plt.xlim(R_min, R_max)
        plt.ylim(q_min, q_max)
        try:
            valid = np.isfinite(c) & (c < 1e5)
            qv = q[valid]; rv = r[valid]; cv = c[valid]
            midx = np.argmin(cv)
            q_best, r_best, c_best = float(qv[midx]), float(rv[midx]), float(cv[midx])
        except Exception:
            pass
    else:
        norm = _compute_log_norm_from_grid(grid)
        Q, R = np.meshgrid(ur, uq)
        plt.figure(figsize=(8, 6))
        im = plt.imshow(grid, origin='lower', aspect='auto',
                        extent=[R_min, R_max, q_min, q_max],
                        cmap='viridis', interpolation='nearest', norm=norm)
        plt.colorbar(im, label='C (Consistency Metric, log scale)')

        idx_min = np.nanargmin(grid)
        iq, ir = np.unravel_index(idx_min, grid.shape)
        plt.scatter([ur[ir]], [uq[iq]], color='red', s=60, marker='*', label='Best')
        plt.xlim(R_min, R_max)
        plt.ylim(q_min, q_max)
        q_best, r_best, c_best = float(uq[iq]), float(ur[ir]), float(grid[iq, ir])

    # Ground-truth overlay (if available)
    try:
        with open(yaml_path, 'r') as yf:
            cfg = yaml.safe_load(yf)
        q_true = float(cfg['Data_Generation']['q'])
        r_true = float(cfg['Data_Generation']['meas_noise_var'])
        plt.scatter([r_true], [q_true], color='white', edgecolor='k', s=100, marker='o', label='True (q,R)')
    except Exception as e:
        print(f"Ground-truth overlay skipped: {e}")

    plt.xlabel('R (Measurement Noise Variance)')
    plt.ylabel('Q (Process Noise Intensity)')
    if q_best is not None and r_best is not None and c_best is not None:
        best_text = f" - Best: Q={q_best:.6g}, R={r_best:.6g}, CNIS={c_best:.6g}"
    else:
        best_text = ""
    plt.title(f'Nonlinear Dense Grid{title_suffix}{best_text}')
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, 'nonlinear_gridsearch_heatmap.png')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot grid-search results for nonlinear tracking scenario.')
    parser.add_argument('--elev', type=float, default=45.0, help='Elevation angle for 3D view (degrees).')
    parser.add_argument('--azim', type=float, default=60.0, help='Azimuth angle for 3D view (degrees).')
    parser.add_argument('--prop', type=int, default=None, help='Proposition number to display (e.g., 3 or 4).')
    parser.add_argument('--nis', type=int, default=None, choices=[3, 4], help='NIS variant to display (3 or 4).')
    args = parser.parse_args()

    if args.prop is not None and args.nis is not None:
        title_suffix = f" - Proposition {args.prop} (NIS{args.nis})"
    elif args.prop is not None:
        title_suffix = f" - Proposition {args.prop}"
    elif args.nis is not None:
        title_suffix = f" - NIS{args.nis}"
    else:
        title_suffix = ""

    q, r, c = load_nonlinear_grid_trials()
    if q is None:
        return
    plot_3d_scatter(q, r, c, elev=args.elev, azim=args.azim, title_suffix=title_suffix)
    plot_heatmap(q, r, c, title_suffix=title_suffix)


if __name__ == "__main__":
    main() 