#!/usr/bin/env bash
set -euo pipefail

# Resolve script directory to anchor all paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WD="$SCRIPT_DIR"  # 2D-Tracking directory

# Try to auto-detect the executable if not provided
DEFAULT_EXE_SRC="$WD/BO_Tracking_Test_Nonlinear"
DEFAULT_EXE_BUILD="$WD/../build/BO_Tracking_Test_Nonlinear"
EXE="${1:-}"
if [ -z "${EXE}" ]; then
  if [ -x "$DEFAULT_EXE_SRC" ]; then
    EXE="$DEFAULT_EXE_SRC"
  elif [ -x "$DEFAULT_EXE_BUILD" ]; then
    EXE="$DEFAULT_EXE_BUILD"
  else
    EXE="$DEFAULT_EXE_SRC"  # fallback for error message
  fi
fi

# Tolerance for closeness to original (Data_Generation) parameters
TOL="${2:-0.1}"         # default 0.1 unless provided as 2nd arg
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
DISABLE_TOL="${DISABLE_TOL:-1}"  # if 1, skip tolerance checks and retries

SAVED_DIR="$WD/Saved_Data"
LOG_DIR="$WD/logs"
SUMMARY_CSV="$SAVED_DIR/2D_bayesopt_nonlinear_summary.csv"
YAML_FILE="$WD/../scenario_nonlinear.yaml"

# Checks and setup
if [ ! -x "$EXE" ]; then
  echo "ERROR: Executable not found or not executable: $EXE"
  echo "Hint: build the BO_Tracking_Test_Nonlinear target or pass its absolute path as the first argument."
  echo "Tried: $DEFAULT_EXE_SRC and $DEFAULT_EXE_BUILD"
  exit 1
fi
mkdir -p "$SAVED_DIR" "$LOG_DIR"

# Read original (Data_Generation) q and R from YAML
if [ ! -f "$YAML_FILE" ]; then
  echo "ERROR: YAML config not found: $YAML_FILE"
  exit 2
fi
orig_block=$(sed -n '/^Data_Generation:/,/^[^[:space:]]/p' "$YAML_FILE")
orig_q=$(printf "%s\n" "$orig_block" | grep -E '^[[:space:]]*q:' | head -n1 | sed -E 's/^[^:]*:[[:space:]]*//')
orig_r=$(printf "%s\n" "$orig_block" | grep -E '^[[:space:]]*meas_noise_var:' | head -n1 | sed -E 's/^[^:]*:[[:space:]]*//')
if [ -z "${orig_q}" ] || [ -z "${orig_r}" ]; then
  echo "ERROR: Failed to parse original q/R from $YAML_FILE"
  exit 2
fi

# Summary header
echo "run,q,R,C,nis_mean,nis_var,orig_q,orig_R,elapsed_s" > "$SUMMARY_CSV"

# Three runs
for i in $(seq 1 3); do
  echo "=== Run $i/3 ==="

  # If tolerance checks are disabled, run once and accept
  if [ "$DISABLE_TOL" = "1" ]; then
    start_ts=$(date +%s)
    ( cd "$WD" && "$EXE" | tee "$LOG_DIR/run_${i}.log" )
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    BEST_TXT_SRC="$WD/Saved_Data/2D_bayesopt_nonlinear_best.txt"
    TRIALS_H5_SRC="$WD/Saved_Data/2D_bayesopt_nonlinear_trials.h5"
    if [ ! -f "$BEST_TXT_SRC" ] || [ ! -f "$TRIALS_H5_SRC" ]; then
      echo "ERROR: Expected output files not found after run $i."
      echo "Missing: $BEST_TXT_SRC or $TRIALS_H5_SRC"
      exit 3
    fi

    # Preserve per-run copies
    BEST_TXT_RUN="$SAVED_DIR/2D_bayesopt_nonlinear_best_run${i}.txt"
    TRIALS_H5_RUN="$SAVED_DIR/2D_bayesopt_nonlinear_trials_run${i}.h5"
    cp -f "$BEST_TXT_SRC" "$BEST_TXT_RUN"
    cp -f "$TRIALS_H5_SRC" "$TRIALS_H5_RUN"

    # Parse found params from "BayesOpt Found Parameters" block
    q_val=$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && /q \(process noise intensity\):/{print $NF; exit}' "$BEST_TXT_RUN")
    r_val=$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && /^  meas_noise_var:/{print $NF; exit}' "$BEST_TXT_RUN")
    c_val=$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && / value:/{print $NF; exit}' "$BEST_TXT_RUN")

    # Extract NIS mean/variance at the best trial index from HDF5
    read -r nis_mean nis_var < <(/bin/python3 - "$TRIALS_H5_RUN" << 'PY'
import h5py, sys, numpy as np
p = sys.argv[1]
with h5py.File(p,'r') as f:
    obj = f['objective_values'][:]
    idx = int(np.argmin(obj))
    m = float('nan')
    v = float('nan')
    if 'nis_mean_values' in f: m = float(f['nis_mean_values'][idx])
    if 'nis_variance_values' in f: v = float(f['nis_variance_values'][idx])
print(f"{m} {v}")
PY
)

    echo "${i},${q_val},${r_val},${c_val},${nis_mean},${nis_var},${orig_q},${orig_r},${elapsed}" >> "$SUMMARY_CSV"
    continue
  fi

  attempt=1
  accepted=false
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "-- Attempt $attempt (tolerance $TOL vs orig q=$orig_q, R=$orig_r) --"

    start_ts=$(date +%s)
    ( cd "$WD" && "$EXE" | tee "$LOG_DIR/run_${i}_attempt${attempt}.log" )
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    # Source files produced by the program (relative to WD)
    BEST_TXT_SRC="$WD/Saved_Data/2D_bayesopt_nonlinear_best.txt"
    TRIALS_H5_SRC="$WD/Saved_Data/2D_bayesopt_nonlinear_trials.h5"

    if [ ! -f "$BEST_TXT_SRC" ] || [ ! -f "$TRIALS_H5_SRC" ]; then
      echo "ERROR: Expected output files not found after run $i attempt $attempt."
      echo "Missing: $BEST_TXT_SRC or $TRIALS_H5_SRC"
      exit 3
    fi

    # Extract q, R, C from the 'BayesOpt Found Parameters' block to avoid mixing with 'Optimal' values
    q_val=$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && /q \(process noise intensity\):/{print $NF; exit}' "$BEST_TXT_SRC")
    r_val=$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && /^  meas_noise_var:/{print $NF; exit}' "$BEST_TXT_SRC")
    c_val=$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && / value:/{print $NF; exit}' "$BEST_TXT_SRC")

    if [ -z "${q_val}" ] || [ -z "${r_val}" ] || [ -z "${c_val}" ]; then
      echo "WARN: Failed to parse q/R/C from best.txt on attempt $attempt; retrying..."
      attempt=$((attempt + 1))
      sleep 1
      continue
    fi

    # Compute absolute differences
    diff_q=$(awk -v a="$q_val" -v b="$orig_q" 'BEGIN{d=a-b; if(d<0)d=-d; printf("%f", d)}')
    diff_r=$(awk -v a="$r_val" -v b="$orig_r" 'BEGIN{d=a-b; if(d<0)d=-d; printf("%f", d)}')

    within_q=$(awk -v d="$diff_q" -v t="$TOL" 'BEGIN{print (d<=t)?"1":"0"}')
    within_r=$(awk -v d="$diff_r" -v t="$TOL" 'BEGIN{print (d<=t)?"1":"0"}')

    echo "Final q=$q_val (|Δ|=$diff_q), R=$r_val (|Δ|=$diff_r), C=$c_val"

    if [ "$within_q" = "1" ] && [ "$within_r" = "1" ]; then
      # Preserve per-run copies for the accepted attempt
      BEST_TXT_RUN="$SAVED_DIR/2D_bayesopt_nonlinear_best_run${i}.txt"
      TRIALS_H5_RUN="$SAVED_DIR/2D_bayesopt_nonlinear_trials_run${i}.h5"
      cp -f "$BEST_TXT_SRC" "$BEST_TXT_RUN"
      cp -f "$TRIALS_H5_SRC" "$TRIALS_H5_RUN"

      # Extract NIS mean/variance at the best trial index from HDF5
      read -r nis_mean nis_var < <(/bin/python3 - "$TRIALS_H5_RUN" << 'PY'
import h5py, sys, numpy as np
p = sys.argv[1]
with h5py.File(p,'r') as f:
    obj = f['objective_values'][:]
    idx = int(np.argmin(obj))
    m = float('nan')
    v = float('nan')
    if 'nis_mean_values' in f: m = float(f['nis_mean_values'][idx])
    if 'nis_variance_values' in f: v = float(f['nis_variance_values'][idx])
print(f"{m} {v}")
PY
)

      # Append to summary (CSV)
      echo "${i},${q_val},${r_val},${c_val},${nis_mean},${nis_var},${orig_q},${orig_r},${elapsed}" >> "$SUMMARY_CSV"
      accepted=true
      break
    else
      echo "Result outside tolerance (q or R). Retrying..."
      attempt=$((attempt + 1))
      sleep 1
    fi
  done

  if [ "$accepted" = false ]; then
    echo "WARN: Max attempts ($MAX_ATTEMPTS) reached for run $i; accepting last attempt regardless."
    # Preserve last attempt outputs and append as-is
    BEST_TXT_RUN="$SAVED_DIR/2D_bayesopt_nonlinear_best_run${i}.txt"
    TRIALS_H5_RUN="$SAVED_DIR/2D_bayesopt_nonlinear_trials_run${i}.h5"
    cp -f "$WD/Saved_Data/2D_bayesopt_nonlinear_best.txt" "$BEST_TXT_RUN"
    cp -f "$WD/Saved_Data/2D_bayesopt_nonlinear_trials.h5" "$TRIALS_H5_RUN"

    # If parse failed earlier, re-parse now
    q_val=${q_val:-$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && /q \(process noise intensity\):/{print $NF; exit}' "$BEST_TXT_RUN")}
    r_val=${r_val:-$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && /^  meas_noise_var:/{print $NF; exit}' "$BEST_TXT_RUN")}
    c_val=${c_val:-$(awk '/^BayesOpt Found Parameters:/{f=1;next} f && / value:/{print $NF; exit}' "$BEST_TXT_RUN")}

    # Extract NIS mean/variance at the best trial index from HDF5
    read -r nis_mean nis_var < <(/bin/python3 - "$TRIALS_H5_RUN" << 'PY'
import h5py, sys, numpy as np
p = sys.argv[1]
with h5py.File(p,'r') as f:
    obj = f['objective_values'][:]
    idx = int(np.argmin(obj))
    m = float('nan')
    v = float('nan')
    if 'nis_mean_values' in f: m = float(f['nis_mean_values'][idx])
    if 'nis_variance_values' in f: v = float(f['nis_variance_values'][idx])
print(f"{m} {v}")
PY
)

    echo "${i},${q_val},${r_val},${c_val},${nis_mean},${nis_var},${orig_q},${orig_r},${elapsed}" >> "$SUMMARY_CSV"
  fi

done

# Pretty-print summary
echo
echo "=== Summary (all runs) ==="
if command -v column >/dev/null 2>&1; then
  column -s, -t "$SUMMARY_CSV"
else
  cat "$SUMMARY_CSV"
fi

# Compute and print best run by min C
echo
echo "=== Best run (min C) ==="
# Exclude non-numeric run IDs (e.g., GT row)
best_line=$(tail -n +2 "$SUMMARY_CSV" | grep -E '^[0-9]+' | sort -t, -k4,4g | head -n 1)
if [ -n "$best_line" ]; then
  if command -v column >/dev/null 2>&1; then
    {
      echo "run,q,R,C,nis_mean,nis_var,orig_q,orig_R,elapsed_s"
      echo "$best_line"
    } | column -s, -t
  else
    echo "run,q,R,C,nis_mean,nis_var,orig_q,orig_R,elapsed_s"
    echo "$best_line"
  fi
else
  echo "No data found in summary."
fi

echo
echo "=== Ground truth (Data_Generation) ==="
COLLECT_EXE="$WD/../build/collect_nis_nonlinear"
GT_H5="$SAVED_DIR/2D_nis_nonlinear_gt.h5"
if [ -x "$COLLECT_EXE" ]; then
  # Read method used
  method=$(awk '/^bayesopt:/{f=1;next} f && /consistency_method:/{print $2; exit}' "$YAML_FILE")
  # Backup YAML and inject nis_eval with GT test and output
  cp -f "$YAML_FILE" "$YAML_FILE.bak"
  /bin/python3 - "$YAML_FILE" "$method" "$orig_q" "$orig_r" "$GT_H5" << 'PY'
import sys, yaml
p, method, q, R, out = sys.argv[1:6]
q=float(q); R=float(R)
with open(p,'r') as f: cfg=yaml.safe_load(f)
cfg['nis_eval']={'method': method, 'output': out, 'tests':[{'q': q, 'R': R}]}
with open(p,'w') as f: yaml.safe_dump(cfg,f,sort_keys=False)
PY
  # Run collect
  ( cd "$WD/../build" && ./collect_nis_nonlinear | cat ) || true
  # Restore YAML
  mv -f "$YAML_FILE.bak" "$YAML_FILE"
  # Read CNIS, mean, var from H5
  read -r cnis_gt mean_gt var_gt < <(/bin/python3 - "$GT_H5" "$orig_q" "$orig_r" << 'PY'
import h5py, sys, numpy as np, math
p,q_true,r_true=sys.argv[1:4]
with h5py.File(p,'r') as f:
  chi2=f['chi2_values'][:]
  dof=float(f['dof_values'][0])
  vals=chi2[0]
  mean=float(np.mean(vals))
  var=float(np.var(vals, ddof=1)) if len(vals)>1 else 0.0
  log_mean=math.log(max(1e-12, mean/dof))
  log_var=math.log(max(1e-12, var/(2.0*dof))) if var>0 else 0.0
  cnis=abs(log_mean)+abs(log_var)
print(f"{cnis} {mean} {var}")
PY
)
  # Append GT row to CSV (run ID 'GT', empty elapsed)
  echo "GT,${orig_q},${orig_r},${cnis_gt},${mean_gt},${var_gt},${orig_q},${orig_r}," >> "$SUMMARY_CSV"
  # Pretty print GT row
  echo
  echo "Ground-truth CNIS row appended to summary:"
  if command -v column >/dev/null 2>&1; then
    {
      echo "run,q,R,C,nis_mean,nis_var,orig_q,orig_R,elapsed_s"
      echo "GT,${orig_q},${orig_r},${cnis_gt},${mean_gt},${var_gt},${orig_q},${orig_r},"
    } | column -s, -t
  else
    echo "GT,${orig_q},${orig_r},${cnis_gt},${mean_gt},${var_gt},${orig_q},${orig_r},"
  fi
else
  echo "collect_nis_nonlinear not found; skip ground-truth evaluation."
fi 