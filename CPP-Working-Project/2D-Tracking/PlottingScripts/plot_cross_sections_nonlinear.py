#!/usr/bin/env python3
"""
Plot cross-sections of CNIS for the nonlinear tracking experiment.
Rows are methods, columns are cross-sections:
- Top row: NIS3 (first)
- Bottom row: NIS4 (second)
Columns:
- Left: lock Q=q_true, sweep R in [1.5, 2.5]
- Right: lock R=R_true, sweep Q in [0.75, 1.25]

Usage:
  python plot_cross_sections_nonlinear.py          # plots NIS3 then NIS4 if both exist (sequential windows)
  python plot_cross_sections_nonlinear.py nis3     # plots only NIS3
  python plot_cross_sections_nonlinear.py nis4     # plots only NIS4
"""

import os
import sys
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
SAVED_DIR = os.path.join(ROOT_DIR, 'Saved_Data')
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
SCENARIO = os.path.join(ROOT_DIR, '..', 'scenario_nonlinear.yaml')


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_cross_sections(method: str):
    h5_path = os.path.join(SAVED_DIR, f'2D_cross_sections_nonlinear_{method}.h5')
    if not os.path.exists(h5_path):
        return None
    with h5py.File(h5_path, 'r') as f:
        lockQ_r = f['lockQ_r_values'][:]
        lockQ_c = f['lockQ_c_values'][:]
        lockR_q = f['lockR_q_values'][:]
        lockR_c = f['lockR_c_values'][:]
        # Optional component datasets (present after tool update)
        lockQ_c_mean = f['lockQ_c_mean'][:] if 'lockQ_c_mean' in f else None
        lockQ_c_var = f['lockQ_c_var'][:] if 'lockQ_c_var' in f else None
        lockR_c_mean = f['lockR_c_mean'][:] if 'lockR_c_mean' in f else None
        lockR_c_var = f['lockR_c_var'][:] if 'lockR_c_var' in f else None
    return (lockQ_r, lockQ_c, lockQ_c_mean, lockQ_c_var), (lockR_q, lockR_c, lockR_c_mean, lockR_c_var)


def plot_method_row(ax_left, ax_right, lockQ, lockR, q_true, R_true, method_label, colors=('b','g')):
    (r_vals, c_lockQ, c_lockQ_mean, c_lockQ_var) = lockQ
    (q_vals, c_lockR, c_lockR_mean, c_lockR_var) = lockR

    idx_min_lockQ = int(np.argmin(c_lockQ))
    idx_min_lockR = int(np.argmin(c_lockR))

    # Left: lock Q, sweep R (scatter)
    ax = ax_left
    ax.scatter(r_vals, c_lockQ, s=12, color=colors[0], alpha=0.7, edgecolors='none', label=f'CNIS (Q locked)')
    if c_lockQ_mean is not None and c_lockQ_var is not None:
        ax.plot(r_vals, c_lockQ_mean, color=colors[0], lw=1.0, alpha=0.9, label='Mean term')
        ax.plot(r_vals, c_lockQ_var, color='tab:gray', lw=1.0, alpha=0.9, label='Variance term')
    ax.scatter([r_vals[idx_min_lockQ]], [c_lockQ[idx_min_lockQ]], color=colors[0], s=60, zorder=5, edgecolors='k')
    ax.axvline(R_true, color='k', ls='--', lw=1.2, label='R true')
    ax.set_xlabel('R (Measurement Noise Variance)')
    ax.set_ylabel('CNIS')
    ax.set_title(f'{method_label}: Lock Q = {q_true:.3f} (sweep R)')
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(loc='best')

    # Right: lock R, sweep Q (scatter)
    ax = ax_right
    ax.scatter(q_vals, c_lockR, s=12, color=colors[1], alpha=0.7, edgecolors='none', label=f'CNIS (R locked)')
    if c_lockR_mean is not None and c_lockR_var is not None:
        ax.plot(q_vals, c_lockR_mean, color=colors[1], lw=1.0, alpha=0.9, label='Mean term')
        ax.plot(q_vals, c_lockR_var, color='tab:gray', lw=1.0, alpha=0.9, label='Variance term')
    ax.scatter([q_vals[idx_min_lockR]], [c_lockR[idx_min_lockR]], color=colors[1], s=60, zorder=5, edgecolors='k')
    ax.axvline(q_true, color='k', ls='--', lw=1.2, label='Q true')
    ax.set_xlabel('Q (Process Noise Intensity)')
    ax.set_title(f'{method_label}: Lock R = {R_true:.3f} (sweep Q)')
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(loc='best')


def method_display_label(method: str) -> str:
    if method.lower() == 'nis3':
        return 'Proposition 3'
    if method.lower() == 'nis4':
        return 'Proposition 4'
    return method.upper()


def plot_single(method: str, lockQ, lockR, q_true: float, R_true: float):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    label = method_display_label(method)
    plot_method_row(axes[0], axes[1], lockQ, lockR, q_true, R_true, method_label=label)

    # Compute minima for both cross-sections
    r_vals, c_lockQ, c_lockQ_mean, c_lockQ_var = lockQ
    q_vals, c_lockR, c_lockR_mean, c_lockR_var = lockR
    idx_r = int(np.argmin(c_lockQ))
    idx_q = int(np.argmin(c_lockR))
    r_star = float(r_vals[idx_r])
    c_r_star = float(c_lockQ[idx_r])
    q_star = float(q_vals[idx_q])
    c_q_star = float(c_lockR[idx_q])

    # Component values at minima (may be None)
    m_r = float(c_lockQ_mean[idx_r]) if c_lockQ_mean is not None else None
    v_r = float(c_lockQ_var[idx_r]) if c_lockQ_var is not None else None
    m_q = float(c_lockR_mean[idx_q]) if c_lockR_mean is not None else None
    v_q = float(c_lockR_var[idx_q]) if c_lockR_var is not None else None

    def fmt(x):
        return f"{x:.6g}" if x is not None else "NA"

    # Print a console table with components
    print("\nMinimum points for", label)
    print("=" * 98)
    print(f"{'Section':<18}{'Fixed':<18}{'Argmin':<14}{'CNIS_min':<14}{'Mean term':<17}{'Variance term':<17}")
    print("-" * 98)
    print(f"{'Lock Q (sweep R)':<18}{f'Q={q_true:.6f}':<18}{f'R*={r_star:.6f}':<14}{f'{c_r_star:.6g}':<14}{fmt(m_r):<17}{fmt(v_r):<17}")
    print(f"{'Lock R (sweep Q)':<18}{f'R={R_true:.6f}':<18}{f'Q*={q_star:.6f}':<14}{f'{c_q_star:.6g}':<14}{fmt(m_q):<17}{fmt(v_q):<17}")
    print("=" * 98)

    # Save minima to CSV (with components)
    os.makedirs(SAVED_DIR, exist_ok=True)
    csv_path = os.path.join(SAVED_DIR, f'nonlinear_cross_sections_{method}_minima.csv')
    with open(csv_path, 'w') as f:
        f.write('method,section,fixed,argmin,cnis_min,mean_term,var_term\n')
        f.write(f"{method},lockQ,Q={q_true:.8f},R*={r_star:.8f},{c_r_star:.10g},{'' if m_r is None else f'{m_r:.10g}'},{'' if v_r is None else f'{v_r:.10g}'}\n")
        f.write(f"{method},lockR,R={R_true:.8f},Q*={q_star:.8f},{c_q_star:.10g},{'' if m_q is None else f'{m_q:.10g}'},{'' if v_q is None else f'{v_q:.10g}'}\n")
    print(f"Saved minima CSV to: {csv_path}")

    # Finalize figure
    fig.suptitle(f'CNIS Cross-Sections (Nonlinear) - {label}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(PLOTS_DIR, f'nonlinear_cross_sections_{method}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to: {out_path}")
    plt.show()
    plt.close(fig)


def plot_both_or_single(cfg, only_method: Optional[str]):
    q_true = float(cfg['Data_Generation']['q'])
    R_true = float(cfg['Data_Generation']['meas_noise_var'])

    # Decide which to plot
    methods = []
    if only_method in ('nis3', 'nis4'):
        methods = [only_method]
    else:
        # Prefer NIS3 first, then NIS4
        methods = ['nis3', 'nis4']

    # Load available
    data = []
    for m in methods:
        cs = load_cross_sections(m)
        if cs is not None:
            data.append((m, cs))
        else:
            print(f"Note: cross-section HDF5 missing for {m}; skipping.")

    if not data:
        raise FileNotFoundError("No cross-section files found. Run CrossSection_Tracking_Nonlinear nis3|nis4 first.")

    # If single method available/requested, plot it
    if len(data) == 1:
        method, (lockQ, lockR) = data[0]
        plot_single(method, lockQ, lockR, q_true, R_true)
        return

    # If both available and not explicitly requested, plot sequentially: NIS3 first, then NIS4
    data_dict = {m: cs for (m, cs) in data}
    for m in ('nis3', 'nis4'):
        if m in data_dict:
            (lockQ, lockR) = data_dict[m]
            plot_single(m, lockQ, lockR, q_true, R_true)


def main():
    cfg = load_config(SCENARIO)
    # If an argument is provided, plot only that method; otherwise plot both (NIS3 first, then NIS4 sequentially)
    only_method = sys.argv[1].lower() if len(sys.argv) > 1 else None
    if only_method not in (None, 'nis3', 'nis4'):
        print(f"Warning: unknown method '{only_method}', plotting both if available.")
        only_method = None
    plot_both_or_single(cfg, only_method)


if __name__ == '__main__':
    main() 