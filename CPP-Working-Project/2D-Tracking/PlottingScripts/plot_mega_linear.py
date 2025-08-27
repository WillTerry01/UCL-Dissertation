#!/usr/bin/env python3
import argparse
import os
import h5py
import yaml
import numpy as np
import matplotlib.pyplot as plt

try:
    import scipy.ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def load_trials_h5(path):
    with h5py.File(path, 'r') as f:
        data = f['trials'][:]
    valid = np.isfinite(data[:, 2]) & (data[:, 2] < 1e5)
    return data[valid, 0], data[valid, 1], data[valid, 2]


def filter_rel_window(Q, R, C, q_true, R_true, tol_rel):
    q_min, q_max = q_true * (1 - tol_rel), q_true * (1 + tol_rel)
    R_min, R_max = R_true * (1 - tol_rel), R_true * (1 + tol_rel)
    mask = (Q >= q_min) & (Q <= q_max) & (R >= R_min) & (R <= R_max)
    return Q[mask], R[mask], C[mask], (q_min, q_max, R_min, R_max)


def overlay_plot(Q1, R1, C1, Q2, R2, C2, q_true, R_true, title, outfile):
    plt.figure(figsize=(8, 6))
    plt.scatter(Q1, R1, c=C1, cmap='Blues', s=50, edgecolor='k', alpha=0.8, label='Constant dt', marker='s')
    plt.scatter(Q2, R2, c=C2, cmap='Oranges', s=50, edgecolor='k', alpha=0.8, label='Variable dt', marker='o')
    plt.scatter([q_true], [R_true], color='red', s=120, marker='*', label='True (q, R)')
    cb = plt.colorbar(label='C (Consistency Metric)')
    cb.set_alpha(1)
    cb.draw_all()
    plt.xlabel('Q (Process Noise Diagonal)')
    plt.ylabel('R (Measurement Noise Diagonal)')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved: {outfile}")
    plt.show()


def _mean_heat(Q, R, C, bounds, bins):
    q_min, q_max, R_min, R_max = bounds
    H_sum, q_edges, r_edges = np.histogram2d(Q, R, bins=bins, range=[[q_min, q_max], [R_min, R_max]], weights=C)
    H_cnt, _, _ = np.histogram2d(Q, R, bins=bins, range=[[q_min, q_max], [R_min, R_max]])
    return H_sum, H_cnt, (q_min, q_max, R_min, R_max)


def _smooth_ratio(H_sum, H_cnt, sigma_bins):
    # Smooth weighted sum and counts, then divide
    if _HAS_SCIPY and sigma_bins > 0:
        H_sum_s = ndi.gaussian_filter(H_sum, sigma=sigma_bins, mode='nearest')
        H_cnt_s = ndi.gaussian_filter(H_cnt, sigma=sigma_bins, mode='nearest')
    else:
        # No scipy: fallback to no data smoothing (visual interpolation will be used)
        H_sum_s = H_sum
        H_cnt_s = H_cnt
    with np.errstate(invalid='ignore', divide='ignore'):
        H_mean = H_sum_s / H_cnt_s
    return np.ma.masked_invalid(H_mean)


def heatmap_pair_plot(Qc, Rc, Cc, Qv, Rv, Cv, bounds, q_true, R_true, title_const, title_var, outfile, bins=100, sigma_bins=1.0):
    q_min, q_max, R_min, R_max = bounds
    if q_max <= q_min or R_max <= R_min:
        print("Skipped heatmap pair: invalid bounds")
        return
    Hs_c, Hn_c, _ = _mean_heat(Qc, Rc, Cc, bounds, bins)
    Hs_v, Hn_v, _ = _mean_heat(Qv, Rv, Cv, bounds, bins)
    Hc = _smooth_ratio(Hs_c, Hn_c, sigma_bins)
    Hv = _smooth_ratio(Hs_v, Hn_v, sigma_bins)

    vmin = np.nanmin([Hc.min(), Hv.min()])
    vmax = np.nanmax([Hc.max(), Hv.max()])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    extent = [q_min, q_max, R_min, R_max]

    im0 = axes[0].imshow(Hc.T, origin='lower', extent=extent, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0].scatter([q_true], [R_true], color='red', s=100, marker='*')
    axes[0].set_xlabel('Q (Process Noise Diagonal)')
    axes[0].set_ylabel('R (Measurement Noise Diagonal)')
    axes[0].set_title(title_const)

    im1 = axes[1].imshow(Hv.T, origin='lower', extent=extent, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1].scatter([q_true], [R_true], color='red', s=100, marker='*')
    axes[1].set_xlabel('Q (Process Noise Diagonal)')
    axes[1].set_ylabel('R (Measurement Noise Diagonal)')
    axes[1].set_title(title_var)

    cbar = fig.colorbar(im1, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label('Mean C in bin (smoothed)')

    if outfile:
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        print(f"Saved: {outfile}")
    plt.show()


def main():
    ap = argparse.ArgumentParser(description='Overlay linear BO Q–R for constant vs variable dt and generate side-by-side heatmaps within a window around true q,R.')
    ap.add_argument('--const', required=True, help='H5 file for constant-dt BO trials (trials dataset Nx3).')
    ap.add_argument('--var', required=True, help='H5 file for variable-dt BO trials (trials dataset Nx3).')
    ap.add_argument('--const_nis4', help='Optional H5 for constant-dt NIS4 run.')
    ap.add_argument('--var_nis4', help='Optional H5 for variable-dt NIS4 run.')
    ap.add_argument('--yaml', default='../scenario_linear.yaml', help='YAML with Data_Generation.q and meas_noise_var.')
    ap.add_argument('--tol_rel', type=float, default=0.2, help='Relative window around true q and R (e.g., 0.2 = 20%).')
    ap.add_argument('--out_prefix', default='../2D-Tracking/plots/linear_mega', help='Output prefix for saved figures.')
    ap.add_argument('--bins', type=int, default=100, help='Number of bins per axis for heatmaps.')
    ap.add_argument('--sigma_bins', type=float, default=1.0, help='Gaussian smoothing sigma in bins (requires scipy; else visual interpolation only).')
    args = ap.parse_args()

    with open(args.yaml, 'r') as yf:
        cfg = yaml.safe_load(yf)
    q_true = float(cfg['Data_Generation']['q'])
    R_true = float(cfg['Data_Generation']['meas_noise_var'])

    # NIS3 (or whichever BO produced these files)
    Qc, Rc, Cc = load_trials_h5(args.const)
    Qv, Rv, Cv = load_trials_h5(args.var)
    Qc_f, Rc_f, Cc_f, bounds = filter_rel_window(Qc, Rc, Cc, q_true, R_true, args.tol_rel)
    Qv_f, Rv_f, Cv_f, _ = filter_rel_window(Qv, Rv, Cv, q_true, R_true, args.tol_rel)

    # Overlay scatter (optional visual)
    out1 = f"{args.out_prefix}_nis3.png"
    overlay_plot(Qc_f, Rc_f, Cc_f, Qv_f, Rv_f, Cv_f, q_true, R_true,
                 title=f'Linear BO (NIS3) filtered ±{int(args.tol_rel*100)}% around true q,R',
                 outfile=out1)

    # Side-by-side heatmaps for constant and variable
    heatmap_pair_plot(
        Qc_f, Rc_f, Cc_f, Qv_f, Rv_f, Cv_f, bounds, q_true, R_true,
        title_const='Linear BO (NIS3) Heatmap (Constant dt)',
        title_var='Linear BO (NIS3) Heatmap (Variable dt)',
        outfile=f"{args.out_prefix}_nis3_heat_pair.png",
        bins=args.bins,
        sigma_bins=args.sigma_bins,
    )

    # NIS4, if provided
    if args.const_nis4 and args.var_nis4:
        Qc4, Rc4, Cc4 = load_trials_h5(args.const_nis4)
        Qv4, Rv4, Cv4 = load_trials_h5(args.var_nis4)
        Qc4_f, Rc4_f, Cc4_f, bounds4 = filter_rel_window(Qc4, Rc4, Cc4, q_true, R_true, args.tol_rel)
        Qv4_f, Rv4_f, Cv4_f, _ = filter_rel_window(Qv4, Rv4, Cv4, q_true, R_true, args.tol_rel)

        out2 = f"{args.out_prefix}_nis4.png"
        overlay_plot(Qc4_f, Rc4_f, Cc4_f, Qv4_f, Rv4_f, Cv4_f, q_true, R_true,
                     title=f'Linear BO (NIS4) filtered ±{int(args.tol_rel*100)}% around true q,R',
                     outfile=out2)

        heatmap_pair_plot(
            Qc4_f, Rc4_f, Cc4_f, Qv4_f, Rv4_f, Cv4_f, bounds4, q_true, R_true,
            title_const='Linear BO (NIS4) Heatmap (Constant dt)',
            title_var='Linear BO (NIS4) Heatmap (Variable dt)',
            outfile=f"{args.out_prefix}_nis4_heat_pair.png",
            bins=args.bins,
            sigma_bins=args.sigma_bins,
        )


if __name__ == '__main__':
    main() 