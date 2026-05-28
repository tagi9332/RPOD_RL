import os
import numpy as np
import matplotlib.pyplot as plt

_COLORS = {
    'best':  '#1565A7',
    'worst': '#0A2342',
    'mean':  '#2E86C1',
}


def _compute_dv_impulses(run_df):
    """Return (times, impulses) for one run.

    impulses[i] is the dV (m/s) consumed at the timestep whose sim_time is
    times[i].  Negative diffs (numerical noise / model state resets) are
    clipped to zero.
    """
    if 'dV_remaining' not in run_df.columns:
        return None, None
    dv = run_df['dV_remaining'].values.astype(float)
    impulses = np.clip(-np.diff(dv), 0.0, None)
    if 'sim_time' in run_df.columns:
        times = run_df['sim_time'].values[1:]
    else:
        times = np.arange(1, len(impulses) + 1, dtype=float)
    return times, impulses


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Histogram of per-timestep ΔV impulse — pooled across all MC runs
# ─────────────────────────────────────────────────────────────────────────────
def plot_impulse_histogram(all_runs_data, summary_df, output_folder):
    """Histogram of every per-timestep ΔV impulse across all Monte Carlo runs.

    Zero-thrust steps are excluded from the histogram body but their fraction
    is annotated so the reader can judge how often the agent coasts.
    """
    print("Generating ΔV Impulse Histogram...")

    all_impulses = []
    total_steps  = 0

    for run_df in all_runs_data:
        _, impulses = _compute_dv_impulses(run_df)
        if impulses is None:
            continue
        total_steps  += len(impulses)
        all_impulses.extend(impulses.tolist())

    if not all_impulses:
        print("  No dV_remaining data — skipping.")
        return

    all_arr   = np.array(all_impulses)
    nonzero   = all_arr[all_arr > 1e-10]
    zero_frac = 1.0 - len(nonzero) / len(all_arr) if len(all_arr) > 0 else 0.0

    if len(nonzero) == 0:
        print("  All impulses are zero — skipping histogram.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(nonzero, bins=60, color=_COLORS['mean'], edgecolor='k', linewidth=0.4,
            alpha=0.75, zorder=3)

    mean_val   = np.mean(nonzero)
    median_val = np.median(nonzero)

    ax.axvline(mean_val,   color='#0A2342', linestyle='--', linewidth=1.8,
               label=f'Mean   = {mean_val:.4g} m/s', zorder=4)
    ax.axvline(median_val, color='#5DADE2', linestyle=':',  linewidth=1.8,
               label=f'Median = {median_val:.4g} m/s', zorder=4)

    stats_text = (
        f"Non-zero steps:  {len(nonzero):,} / {total_steps:,}  "
        f"({100*(1-zero_frac):.1f}% thrusting)\n"
        f"Coasting (zero): {zero_frac*100:.1f}% of all steps\n\n"
        f"mean   = {mean_val:.4g} m/s\n"
        f"median = {median_val:.4g} m/s\n"
        f"std    = {nonzero.std():.4g} m/s\n"
        f"max    = {nonzero.max():.4g} m/s"
    )
    ax.text(0.97, 0.97, stats_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.88))

    ax.set_xlabel('ΔV Impulse per Timestep (m/s)', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.set_title(
        f'Commanded ΔV Impulse Distribution  '
        f'(all MC runs, non-zero steps only,  n = {len(nonzero):,})',
        fontsize=15, fontweight='bold'
    )
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'dv_impulse_histogram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Per-timestep ΔV impulse time series — best and worst runs
# ─────────────────────────────────────────────────────────────────────────────
def plot_impulse_timeseries(best_run_df, worst_run_df, output_folder):
    """Bar-chart style per-timestep ΔV impulse for the best and worst runs."""
    print("Generating ΔV Impulse Time Series (Best / Worst)...")

    entries = [
        (best_run_df,  'Best Run',  _COLORS['best']),
        (worst_run_df, 'Worst Run', _COLORS['worst']),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))

    for ax, (run_df, label, color) in zip(axes, entries):
        times, impulses = _compute_dv_impulses(run_df)
        if times is None or len(times) == 0:
            ax.text(0.5, 0.5, 'No dV data', transform=ax.transAxes, ha='center')
            ax.set_title(label, fontsize=14, fontweight='bold')
            continue

        dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
        ax.bar(times, impulses, width=dt * 0.85,
               color=color, alpha=0.70, edgecolor='none', zorder=3)

        mean_imp  = float(np.mean(impulses))
        total_dv  = float(np.sum(impulses))
        thrust_frac = float(np.mean(impulses > 1e-10)) * 100.0

        ax.axhline(mean_imp, color='black', linestyle='--', linewidth=1.4,
                   label=f'Mean = {mean_imp:.4g} m/s', zorder=4)

        info_text = (
            f"Total ΔV = {total_dv:.3f} m/s\n"
            f"Thrusting = {thrust_frac:.1f}% of steps"
        )
        ax.text(0.01, 0.97, info_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='gray', alpha=0.85))

        ax.set_ylabel('ΔV Impulse (m/s)', fontsize=13)
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Simulation Time (s)', fontsize=14)
    fig.suptitle('Per-Timestep ΔV Impulse: Best vs. Worst Run',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'dv_impulse_timeseries.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Mean per-timestep ΔV impulse vs. simulation time  (all runs)
# ─────────────────────────────────────────────────────────────────────────────
def plot_mean_impulse_history(all_runs_data, summary_df, output_folder, n_bins=80):
    """Mean commanded ΔV impulse at each point in sim time, averaged over all runs."""
    print("Generating Mean ΔV Impulse History...")

    if not any('dV_remaining' in df.columns for df in all_runs_data):
        print("  No dV_remaining data — skipping.")
        return

    all_times = []
    all_imps  = []

    for run_df in all_runs_data:
        times, impulses = _compute_dv_impulses(run_df)
        if times is None:
            continue
        all_times.extend(times.tolist())
        all_imps.extend(impulses.tolist())

    if not all_times:
        print("  No data to plot.")
        return

    all_times = np.array(all_times)
    all_imps  = np.array(all_imps)

    t_max      = all_times.max()
    bin_edges  = np.linspace(0.0, t_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    bin_means, bin_stds, bin_counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (all_times >= lo) & (all_times < hi)
        pts  = all_imps[mask]
        bin_counts.append(int(mask.sum()))
        if len(pts) > 0:
            bin_means.append(float(np.mean(pts)))
            bin_stds.append(float(np.std(pts)))
        else:
            bin_means.append(np.nan)
            bin_stds.append(np.nan)

    bm    = np.array(bin_means)
    bs    = np.array(bin_stds)
    valid = ~np.isnan(bm)

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.fill_between(
        bin_centers[valid],
        np.maximum(0.0, (bm - bs)[valid]),
        (bm + bs)[valid],
        alpha=0.25, color=_COLORS['mean'], zorder=2, label='±1σ band'
    )
    ax.plot(bin_centers[valid], bm[valid],
            color=_COLORS['mean'], linewidth=2.2,
            label='Mean impulse per step', zorder=3)

    overall_mean = float(np.mean(all_imps))
    ax.axhline(overall_mean, color='black', linestyle='--', linewidth=1.4,
               label=f'Grand mean = {overall_mean:.4g} m/s', zorder=4)

    ax.set_xlabel('Simulation Time (s)', fontsize=14)
    ax.set_ylabel('Mean ΔV Impulse (m/s)', fontsize=14)
    ax.set_title(
        'Mean Commanded ΔV Impulse per Timestep over Simulation Time  (All MC Runs)',
        fontsize=15, fontweight='bold'
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'dv_impulse_mean_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
