"""
Compare mean Delta-V (dV) usage between the baseline and reduced-maneuverability
policies across ALL Monte Carlo runs -- not just successful (docked) ones.

For each run, dV used is computed as:
    dV_used = dV_remaining[first timestep] - dV_remaining[last timestep]

This captures total propellant consumption over the full episode regardless of
outcome (docked, timeout, collision, out-of-bounds, etc.), directly addressing
whether the "successful runs only" dV comparison in the paper is subject to a
selection effect (i.e., whether excluded failed runs are systematically more or
less fuel-expensive than the runs that succeeded).

Usage:
    Set BASELINE_CSV_PATH and REDUCED_CSV_PATH below, then run:
        python compare_dv_all_runs.py

Expects (at minimum) these columns in each CSV:
    run_id, sim_time, dV_remaining, docked_state
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
BASELINE_CSV_PATH = r"D:\tanne\Desktop\results\mc_all_runs_data.csv"   # <-- set to your baseline (0.25 m/s^2) file
REDUCED_CSV_PATH  = "results\mc_all_runs_data.csv"    # <-- set to your reduced (0.05 m/s^2) file
ALPHA = 0.05

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def load_and_prepare(path):
    df = pd.read_csv(path)
    required = ["run_id", "sim_time", "dV_remaining", "docked_state"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    # docked_state may be read as the literal string "True"/"False"
    if df["docked_state"].dtype == object:
        df["docked_state"] = df["docked_state"].astype(str).str.strip().str.lower() == "true"

    return df


def per_run_dv_used(df):
    """
    For each run_id, compute dV used = dV_remaining at first timestep
    minus dV_remaining at last timestep. Also carries through whether
    that run ultimately docked, for the success-conditioned breakdown.
    """
    df_sorted = df.sort_values(["run_id", "sim_time"])

    def _summarize(g):
        first_dv = g["dV_remaining"].iloc[0]
        last_dv = g["dV_remaining"].iloc[-1]
        docked = bool(g["docked_state"].any())  # docked if it ever registers True
        return pd.Series({
            "dV_used": first_dv - last_dv,
            "docked": docked,
            "n_timesteps": len(g),
            "final_sim_time": g["sim_time"].iloc[-1],
        })

    per_run = df_sorted.groupby("run_id", group_keys=False).apply(_summarize, include_groups=False)
    return per_run


def summarize(label, series):
    print(f"{label:28s} n={len(series):4d}  mean={series.mean():7.3f}  "
          f"median={series.median():7.3f}  std={series.std():6.3f}")


# ---------------------------------------------------------------
# Load both campaigns
# ---------------------------------------------------------------
baseline_raw = load_and_prepare(BASELINE_CSV_PATH)
reduced_raw = load_and_prepare(REDUCED_CSV_PATH)

baseline_runs = per_run_dv_used(baseline_raw)
reduced_runs = per_run_dv_used(reduced_raw)

print("=" * 78)
print("DELTA-V USAGE COMPARISON: ALL RUNS vs. SUCCESSFUL-ONLY")
print("=" * 78)
print(f"Baseline campaign: {len(baseline_runs)} runs "
      f"({baseline_runs['docked'].sum()} docked, {(~baseline_runs['docked']).sum()} not docked)")
print(f"Reduced campaign:  {len(reduced_runs)} runs "
      f"({reduced_runs['docked'].sum()} docked, {(~reduced_runs['docked']).sum()} not docked)")
print()

# ---------------------------------------------------------------
# Summary: all runs vs. successful-only vs. failed-only, both campaigns
# ---------------------------------------------------------------
print("-" * 78)
print("Mean dV used (m/s), by campaign and outcome subset")
print("-" * 78)
summarize("Baseline - ALL runs", baseline_runs["dV_used"])
summarize("Baseline - docked only", baseline_runs.loc[baseline_runs["docked"], "dV_used"])
summarize("Baseline - NOT docked", baseline_runs.loc[~baseline_runs["docked"], "dV_used"])
print()
summarize("Reduced - ALL runs", reduced_runs["dV_used"])
summarize("Reduced - docked only", reduced_runs.loc[reduced_runs["docked"], "dV_used"])
summarize("Reduced - NOT docked", reduced_runs.loc[~reduced_runs["docked"], "dV_used"])
print()

# ---------------------------------------------------------------
# Statistical test: baseline ALL-runs vs. reduced ALL-runs
# ---------------------------------------------------------------
print("-" * 78)
print("Mann-Whitney U test: Baseline (all runs) vs. Reduced (all runs)")
print("-" * 78)
u_stat, u_p = stats.mannwhitneyu(
    baseline_runs["dV_used"], reduced_runs["dV_used"], alternative="two-sided"
)
print(f"U = {u_stat:.1f}, p = {u_p:.4g}")
if u_p < ALPHA:
    direction = "LOWER" if reduced_runs["dV_used"].mean() < baseline_runs["dV_used"].mean() else "HIGHER"
    print(f"=> Significant difference (p < {ALPHA}): reduced-policy dV usage is {direction}, "
          f"across ALL runs, not just successful ones.")
else:
    print(f"=> No significant difference detected (p >= {ALPHA}) across ALL runs.")
print()

print("-" * 78)
print("Mann-Whitney U test: Baseline (docked only) vs. Reduced (docked only)")
print("-" * 78)
u_stat2, u_p2 = stats.mannwhitneyu(
    baseline_runs.loc[baseline_runs["docked"], "dV_used"],
    reduced_runs.loc[reduced_runs["docked"], "dV_used"],
    alternative="two-sided",
)
print(f"U = {u_stat2:.1f}, p = {u_p2:.4g}")
print("(This reproduces the successful-runs-only comparison from the paper for reference.)")
print()

# ---------------------------------------------------------------
# Figure: dV distribution comparison
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

# Overlaid histograms, ALL runs, baseline vs reduced
ax.hist(baseline_runs["dV_used"], bins=30, alpha=0.55, color="tab:blue",
        label=f"Baseline, ALL runs (n={len(baseline_runs)})", density=True)
ax.hist(reduced_runs["dV_used"], bins=30, alpha=0.55, color="tab:orange",
        label=f"Reduced, ALL runs (n={len(reduced_runs)})", density=True)
ax.axvline(baseline_runs["dV_used"].mean(), color="tab:blue", ls="--", lw=1.5)
ax.axvline(reduced_runs["dV_used"].mean(), color="tab:orange", ls="--", lw=1.5)
ax.set_xlabel(r"$\Delta V$ used per run (m/s)")
ax.set_ylabel("Density")
ax.set_title(r"$\Delta V$ Usage Distribution: ALL Runs")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("dv_all_runs_comparison.png", dpi=200, bbox_inches="tight")
print("Saved figure: dv_all_runs_comparison.png")