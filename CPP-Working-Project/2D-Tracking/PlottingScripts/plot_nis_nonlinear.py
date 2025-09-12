#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# Optional SciPy for chi-square pdf/cdf and normal pdf
try:
    from scipy.stats import chi2 as sp_chi2, norm as sp_norm
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

H5 = "../2D-Tracking/Saved_Data/2D_nis_nonlinear.h5"

with h5py.File(H5, 'r') as f:
    q = f['q_values'][:]
    r = f['r_values'][:]
    dof = f['dof_values'][:]
    chi2 = f['chi2_values'][:]  # shape (num_tests, num_runs)

num_tests, num_runs = chi2.shape

# Compute CNIS per test
cnis = np.zeros(num_tests)
for i in range(num_tests):
    y = chi2[i, :]
    total_dof = float(dof[i]) if float(dof[i]) > 0 else 1.0
    mean_nis = float(np.mean(y))
    var_nis = float(np.var(y, ddof=1)) if num_runs > 1 else 0.0
    log_mean = np.log(max(mean_nis / total_dof, 1e-12))
    log_var_term = var_nis / (2.0 * total_dof) if total_dof > 0 else 0.0
    log_variance = np.log(max(log_var_term, 1e-12)) if var_nis > 0 else 0.0
    cnis[i] = abs(log_mean) + abs(log_variance)

os.makedirs("../2D-Tracking/plots", exist_ok=True)

# Boxplot per test (include CNIS)
plt.figure(figsize=(12, 6))
plt.boxplot([chi2[i, :] for i in range(num_tests)], labels=[f"q={q[i]:.3f}, R={r[i]:.3f}\nDOF={int(dof[i])}, CNIS={cnis[i]:.6f}" for i in range(num_tests)])
plt.ylabel("NIS (chi2)")
plt.title("Nonlinear NIS distributions across runs per (q,R)")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("../2D-Tracking/plots/nonlinear_nis_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()

# Histogram for best test (by CNIS) with theoretical overlays
best_idx = int(np.argmin(cnis))
y_best = chi2[best_idx, :]
df_best = float(dof[best_idx]) if float(dof[best_idx]) > 0 else 1.0
mu_theory = df_best
sigma_theory = np.sqrt(2.0 * df_best)

# Compute histogram once to reuse for GOF
counts, edges = np.histogram(y_best, bins=30)
bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
centers = 0.5 * (edges[:-1] + edges[1:])

plt.figure(figsize=(8, 5))
plt.hist(y_best, bins=edges, alpha=0.8, color='indianred', edgecolor='k', label='Observed')
plt.axvline(np.mean(y_best), color='red', linestyle='--', label=f"mean={np.mean(y_best):.1f}")
plt.axvline(df_best, color='green', linestyle='-', label=f"DOF={int(df_best)}")

# Overlay chi-square PDF and normal approximation scaled to counts
x = np.linspace(edges[0], edges[-1], 400)
scale = len(y_best) * bin_width
if SCIPY_AVAILABLE:
    chi2_curve = sp_chi2.pdf(x, df_best) * scale
    plt.plot(x, chi2_curve, 'b-', lw=2, label='Chi-square PDF (theory)')
else:
    # Fallback: skip chi2 overlay if SciPy not available
    pass

# Normal approximation overlay (always available)
normal_curve = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_theory)) * np.exp(-0.5 * ((x - mu_theory) / sigma_theory) ** 2) * scale
plt.plot(x, normal_curve, 'k--', lw=2, label='Normal approx PDF')

plt.xlabel("NIS (chi2)")
plt.ylabel("Count")
plt.title(f"Nonlinear NIS histogram for q={q[best_idx]:.3f}, R={r[best_idx]:.3f}\n(CNIS={cnis[best_idx]:.6f})")
plt.legend()
plt.tight_layout()
plt.savefig("../2D-Tracking/plots/nonlinear_nis_hist_best.png", dpi=300, bbox_inches='tight')
plt.show()

# Pearson chi-square goodness-of-fit to chi-square(df_best)
if SCIPY_AVAILABLE:
    expected = len(y_best) * (sp_chi2.cdf(edges[1:], df_best) - sp_chi2.cdf(edges[:-1], df_best))
    # Use only bins with expected >= 5 for validity
    mask = expected >= 5.0
    obs = counts[mask]
    exp = expected[mask]
    if exp.size >= 2 and np.all(exp > 0):
        stat = float(np.sum((obs - exp) ** 2 / exp))
        dof_gof = int(max(1, exp.size - 1))
        pval = float(1.0 - sp_chi2.cdf(stat, dof_gof))

        plt.figure(figsize=(9, 5))
        width = 0.45 * bin_width
        plt.bar(centers[mask] - 0.5 * width, obs, width=width, color='tab:blue', alpha=0.7, label='Observed')
        plt.bar(centers[mask] + 0.5 * width, exp, width=width, color='tab:orange', alpha=0.7, label='Expected (chi-square)')
        plt.xlabel("NIS (chi2)")
        plt.ylabel("Count per bin")
        plt.title(f"Pearson GOF (best test): stat={stat:.2f}, dof={dof_gof}, p={pval:.3g}")
        plt.legend()
        plt.tight_layout()
        plt.savefig("../2D-Tracking/plots/nonlinear_nis_gof_best.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("[WARN] Pearson GOF skipped: insufficient expected counts per bin.")
else:
    print("[INFO] SciPy not available: skipping chi-square overlay and Pearson GOF.")

# Per-test MC run index vs chi2 with mean/variance and CNIS in title
for i in range(num_tests):
    x_idx = np.arange(num_runs)
    y = chi2[i, :]
    mu = float(np.mean(y))
    var = float(np.var(y, ddof=1))
    plt.figure(figsize=(8, 6))
    plt.plot(x_idx, y, 'o-', alpha=0.7)
    plt.axhline(mu, color='red', linestyle='--', label=f"mean={mu:.2f}")
    plt.xlabel("Monte Carlo run index")
    plt.ylabel("NIS (chi2)")
    plt.title(f"Nonlinear NIS per run: q={q[i]:.3f}, R={r[i]:.3f}, DOF={int(dof[i])}, CNIS={cnis[i]:.6f}\nvariance={var:.2f}, mean={mu:.2f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = f"../2D-Tracking/plots/nonlinear_nis_runs_test_{i+1}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show() 