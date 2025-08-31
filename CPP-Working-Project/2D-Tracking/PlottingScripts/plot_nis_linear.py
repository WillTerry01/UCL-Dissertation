#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

H5 = "../2D-Tracking/Saved_Data/2D_nis_linear.h5"

with h5py.File(H5, 'r') as f:
    q = f['q_values'][:]
    r = f['r_values'][:]
    dof = f['dof_values'][:]
    chi2 = f['chi2_values'][:]  # shape (num_tests, num_runs)

num_tests, num_runs = chi2.shape

# Compute CNIS per test (consistent with grid search definition)
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

# Boxplot per test (add CNIS to labels)
plt.figure(figsize=(12, 6))
plt.boxplot([chi2[i, :] for i in range(num_tests)], labels=[f"q={q[i]:.3f}, R={r[i]:.3f}\nDOF={int(dof[i])}, CNIS={cnis[i]:.6f}" for i in range(num_tests)])
plt.ylabel("NIS (chi2)")
plt.title("Linear NIS distributions across runs per (q,R)")
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("../2D-Tracking/plots/linear_nis_boxplots.png", dpi=300, bbox_inches='tight')
plt.show()

# Histogram for best test
best_idx = int(np.argmin(cnis))
plt.figure(figsize=(8, 5))
plt.hist(chi2[best_idx, :], bins=30, alpha=0.8, color='steelblue', edgecolor='k')
plt.axvline(np.mean(chi2[best_idx, :]), color='red', linestyle='--', label=f"mean={np.mean(chi2[best_idx, :]):.1f}")
plt.axvline(dof[best_idx], color='green', linestyle='-', label=f"DOF={int(dof[best_idx])}")
plt.xlabel("NIS (chi2)")
plt.ylabel("Count")
plt.title(f"Linear NIS histogram for q={q[best_idx]:.3f}, R={r[best_idx]:.3f} (CNIS={cnis[best_idx]:.6f})")
plt.legend()
plt.tight_layout()
plt.savefig("../2D-Tracking/plots/linear_nis_hist_best.png", dpi=300, bbox_inches='tight')
plt.show()

# Per-test MC run index (x) vs chi2 (y) with mean and variance annotation (include CNIS)
for i in range(num_tests):
    x = np.arange(num_runs)
    y = chi2[i, :]
    mu = float(np.mean(y))
    var = float(np.var(y, ddof=1))
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, 'o-', alpha=0.7)
    plt.axhline(mu, color='red', linestyle='--', label=f"mean={mu:.2f}")
    plt.xlabel("Monte Carlo run index")
    plt.ylabel("NIS (chi2)")
    plt.title(f"Linear NIS per run: q={q[i]:.3f}, R={r[i]:.3f}, DOF={int(dof[i])}, CNIS={cnis[i]:.6f}\nvariance={var:.2f}, mean={mu:.2f}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = f"../2D-Tracking/plots/linear_nis_runs_test_{i+1}.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.show() 