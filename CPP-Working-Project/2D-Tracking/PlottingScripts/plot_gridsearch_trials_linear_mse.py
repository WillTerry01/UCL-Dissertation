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
DEFAULT_H5 = "../2D-Tracking/Saved_Data/2D_gridsearch_linear_trials_mse.h5"
DEFAULT_SCENARIO = "../scenario_linear.yaml"


def load_linear_grid_trials_mse(h5_path=DEFAULT_H5):
    if not os.path.exists(h5_path):
        print(f"Error: Grid-search MSE file not found: {h5_path}")
        print("Please run GridSearch_Tracking_Linear_MSE first to generate results.")
        return None, None, None
    with h5py.File(h5_path, 'r') as f:
        q = f['q_values'][:]
        r = f['r_values'][:]
        c = f['objective_values'][:]
    return q, r, c


def infer_grid(q, r, c):
    valid = np.isfinite(c)
    q = q[valid]; r = r[valid]; c = c[valid]
    uq = np.unique(q); ur = np.unique(r)
    if uq.size * ur.size == q.size:
        q_idx = np.searchsorted(uq, q)
        r_idx = np.searchsorted(ur, r)
        grid = np.full((uq.size, ur.size), np.nan)
        grid[q_idx, r_idx] = c
        return uq, ur, grid
    return None, None, None


def _compute_log_norm_from_array(values):
    eps = 1e-12
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return colors.LogNorm(vmin=eps, vmax=1.0)
    pos = vals[vals > 0]
    vmin = max(eps, np.nanmin(pos)) if pos.size else eps
    vmax = max(vmin * (1.0 + 1e-6), np.nanmax(vals))
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def _compute_log_norm_from_grid(grid):
    eps = 1e-12
    g = np.asarray(grid)
    g = g[np.isfinite(g)]
    if g.size == 0:
        return colors.LogNorm(vmin=eps, vmax=1.0)
    pos = g[g > 0]
    vmin = max(eps, np.nanmin(pos)) if pos.size else eps
    vmax = max(vmin * (1.0 + 1e-6), np.nanmax(g))
    return colors.LogNorm(vmin=vmin, vmax=vmax)


def plot_3d_scatter(q, r, c, elev=30.0, azim=45.0):
    valid = np.isfinite(c)
    q = q[valid]; r = r[valid]; c = c[valid]
    norm = _compute_log_norm_from_array(c)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(q, r, c, c=c, cmap='viridis', s=30, edgecolor='k', norm=norm, alpha=0.001)
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
    ax.set_zlabel('Average Position MSE')
    ax.set_title('Linear Grid Search (Avg Position MSE)')
    ax.view_init(elev=elev, azim=azim)
    cbar = plt.colorbar(mappable if mappable is not None else sc, pad=0.1)
    cbar.set_label('Average Position MSE (log scale)')
    ax.legend(); ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, 'linear_gridsearch_trials_mse_3d.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); print(f"Saved: {out}")
    plt.show()


def plot_heatmap(q, r, c, yaml_path=DEFAULT_SCENARIO):
    try:
        with open(yaml_path, 'r') as yf:
            cfg = yaml.safe_load(yf)
        if 'mse_grid_search' in cfg and cfg['mse_grid_search']:
            gs = cfg['mse_grid_search']
            q_min = float(gs.get('q_min', np.min(q)))
            q_max = float(gs.get('q_max', np.max(q)))
            R_min = float(gs.get('R_min', np.min(r)))
            R_max = float(gs.get('R_max', np.max(r)))
        elif 'grid_search' in cfg and cfg['grid_search']:
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
    plt.figure(figsize=(8, 6))
    if grid is not None:
        norm = _compute_log_norm_from_grid(grid)
        im = plt.imshow(grid, origin='lower', aspect='auto', extent=[R_min, R_max, q_min, q_max], cmap='viridis', interpolation='nearest', norm=norm)
        plt.colorbar(im, label='Average Position MSE (log scale)')
        idx_min = np.nanargmin(grid)
        iq, ir = np.unravel_index(idx_min, grid.shape)
        plt.scatter([ur[ir]], [uq[iq]], color='red', s=60, marker='*', label='Best')
    else:
        norm = _compute_log_norm_from_array(c)
        sc = plt.scatter(r, q, c=c, cmap='viridis', s=20, edgecolor='k', norm=norm)
        plt.colorbar(sc, label='Average Position MSE (log scale)')
        valid = np.isfinite(c)
        qv = q[valid]; rv = r[valid]; cv = c[valid]
        midx = np.argmin(cv)
        plt.scatter([rv[midx]], [qv[midx]], color='red', s=60, marker='*', label='Best')
    plt.xlabel('R (Measurement Noise Variance)')
    plt.ylabel('Q (Process Noise Intensity)')
    plt.title('Linear Dense Grid (Avg Position MSE)')
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.legend(loc='best')
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, 'linear_gridsearch_mse_heatmap.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); print(f"Saved: {out}")
    plt.show()


def plot_heatmap_thresholded(q, r, c, yaml_path=DEFAULT_SCENARIO, threshold=1.0):
    try:
        with open(yaml_path, 'r') as yf:
            cfg = yaml.safe_load(yf)
        if 'mse_grid_search' in cfg and cfg['mse_grid_search']:
            gs = cfg['mse_grid_search']
            q_min = float(gs.get('q_min', np.min(q)))
            q_max = float(gs.get('q_max', np.max(q)))
            R_min = float(gs.get('R_min', np.min(r)))
            R_max = float(gs.get('R_max', np.max(r)))
        elif 'grid_search' in cfg and cfg['grid_search']:
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
    if grid is None:
        print("Grid could not be reshaped; skipping thresholded heatmap.")
        return
    vals = grid[np.isfinite(grid)]
    pos = vals[vals > 0]
    vmin = max(1e-12, float(np.nanmin(pos))) if pos.size else 1e-12
    norm = colors.LogNorm(vmin=vmin, vmax=threshold)
    cmap = plt.cm.viridis.copy(); cmap.set_over('crimson')
    plt.figure(figsize=(8, 6))
    im = plt.imshow(grid, origin='lower', aspect='auto', extent=[R_min, R_max, q_min, q_max], cmap=cmap, norm=norm, interpolation='nearest')
    plt.colorbar(im, label=f'Average Position MSE (log scale, > {threshold:g} single color)', extend='max')
    plt.xlabel('R (Measurement Noise Variance)'); plt.ylabel('Q (Process Noise Intensity)')
    plt.title(f'Thresholded Heatmap (Avg MSE; threshold={threshold:g})')
    plt.grid(True, linestyle='--', alpha=0.2); plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, 'linear_gridsearch_mse_heatmap_thresholded.png')
    plt.savefig(out, dpi=300, bbox_inches='tight'); print(f"Saved: {out}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot MSE grid-search results for linear tracking scenario.')
    parser.add_argument('--elev', type=float, default=45.0)
    parser.add_argument('--azim', type=float, default=60.0)
    parser.add_argument('--h5', type=str, default=DEFAULT_H5)
    parser.add_argument('--scenario', type=str, default=DEFAULT_SCENARIO)
    parser.add_argument('--threshold', type=float, default=1.0)
    args = parser.parse_args()
    q, r, c = load_linear_grid_trials_mse(args.h5)
    if q is None: return
    plot_3d_scatter(q, r, c, elev=args.elev, azim=args.azim)
    plot_heatmap(q, r, c, yaml_path=args.scenario)
    plot_heatmap_thresholded(q, r, c, yaml_path=args.scenario, threshold=args.threshold)

if __name__ == '__main__':
    main() 