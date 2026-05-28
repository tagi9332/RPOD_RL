import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from resources.sim_parameters import (
    sun_illumination_cone_angle_deg as _SUN_CONE_DEG,
    illumination_cutoff_range as _ILLUM_CUTOFF,
)

# Mission-geometry constants (mirrors resources/sim_parameters.py)
_STANDOFF_DIST = 30.0
_CONJUNCTION_R = 5.0
_WAYPOINT_R    = 10.0
_CORRIDOR_DEG  = 30.0

# Abyss blue shades
_ML = {
    'darkest':  '#0A2342',
    'dark':     '#0D5C8B',
    'mid_dark': '#1565A7',
    'medium':   '#2E86C1',
    'light':    '#5DADE2',
    'lightest': '#85C1E9',
}

# Outcome colours — success/fail kept as-is per user preference
_C = {
    'success':   '#55A868',
    'collision': '#EF553B',
    'timeout':   '#9E9E9E',
    'fuel':      '#0D5C8B',
    'other':     '#EF553B',
}
_CATEGORY_ORDER  = ['Docked', 'Collision', 'Timeout', 'Fuel / Bounds']
_CATEGORY_COLORS = [_C['success'], _C['collision'], _C['timeout'], _C['fuel']]


def _status_color(end_status, success):
    if success:
        return _C['success']
    s = str(end_status).lower()
    if 'collision' in s:
        return _C['collision']
    if 'timeout' in s:
        return _C['timeout']
    if 'fuel' in s or 'bound' in s:
        return _C['fuel']
    return _C['other']


def _simplify_status(s):
    s = str(s)
    if s.startswith('Docked'):
        return 'Docked'
    if s.startswith('Collision'):
        return 'Collision'
    if 'Timeout' in s:
        return 'Timeout'
    return 'Fuel / Bounds'


def _separation(run_df):
    return np.sqrt(run_df['hill_x']**2 + run_df['hill_y']**2 + run_df['hill_z']**2)


def _has_pos(run_df):
    return all(c in run_df.columns for c in ['hill_x', 'hill_y', 'hill_z'])


def _has_vel(run_df):
    return all(c in run_df.columns for c in ['v_DC_C_x', 'v_DC_C_y', 'v_DC_C_z'])


# ─────────────────────────────────────────────────────────────────────────────
# 1. Velocity–Range (V-R) diagram
# ─────────────────────────────────────────────────────────────────────────────
def plot_vr_diagram(all_runs_data, summary_df, output_folder):
    print("Generating V-R Diagram...")

    fig, ax = plt.subplots(figsize=(10, 7))
    seen_labels = set()

    for idx, run_df in enumerate(all_runs_data):
        row       = summary_df.iloc[idx]
        success   = bool(row['success'])
        end_status = str(row['end_status'])

        if not (_has_pos(run_df) and _has_vel(run_df)):
            continue

        sep   = _separation(run_df).values
        speed = np.sqrt(
            run_df['v_DC_C_x']**2 + run_df['v_DC_C_y']**2 + run_df['v_DC_C_z']**2
        ).values
        color = _status_color(end_status, success)
        alpha = 0.65 if success else 0.20

        label_key = 'Docked' if success else _simplify_status(end_status)
        lbl = label_key if label_key not in seen_labels else None
        if lbl:
            seen_labels.add(label_key)

        ax.plot(sep, speed, color=color, alpha=alpha, linewidth=0.8, label=lbl, zorder=2)

    # Reference velocity gates (v = r / T_stop)
    r_ref = np.linspace(1.0, 2200.0, 500)
    ax.plot(r_ref, r_ref / 1800, color=_ML['medium'], linestyle='--', linewidth=1.8,
            label='Gate  v = r / 1800 s  (30-min stop)', zorder=4)
    ax.plot(r_ref, r_ref / 600,  color=_ML['darkest'], linestyle=':', linewidth=1.8,
            label='Gate  v = r / 600 s  (10-min stop)',  zorder=4)

    ax.set_xlabel('Separation Distance (m)', fontsize=14)
    ax.set_ylabel('Relative Speed (m/s)', fontsize=14)
    ax.set_title('Velocity–Range (V-R) Diagram', fontsize=16, fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'vr_diagram.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Approach angle time history (all runs)
# ─────────────────────────────────────────────────────────────────────────────
def plot_approach_angle_history(all_runs_data, summary_df, output_folder,
                                corridor_angle_deg=_CORRIDOR_DEG):
    print("Generating Approach Angle Time History...")

    fig, ax = plt.subplots(figsize=(12, 6))
    seen_labels = set()

    for idx, run_df in enumerate(all_runs_data):
        if 'approach_angle_deg' not in run_df.columns:
            continue

        row        = summary_df.iloc[idx]
        success    = bool(row['success'])
        end_status = str(row['end_status'])
        color      = _status_color(end_status, success)
        alpha      = 0.65 if success else 0.20

        n      = len(run_df)
        t_norm = np.linspace(0.0, 1.0, n)
        angles = run_df['approach_angle_deg'].values

        label_key = 'Docked' if success else _simplify_status(end_status)
        lbl = label_key if label_key not in seen_labels else None
        if lbl:
            seen_labels.add(label_key)

        ax.plot(t_norm, angles, color=color, alpha=alpha, linewidth=0.8, label=lbl, zorder=2)

    # Corridor limit
    ax.axhline(corridor_angle_deg, color='black', linestyle='--', linewidth=1.8,
               label=f'Corridor limit ({corridor_angle_deg:.0f}°)', zorder=5)
    ax.axhspan(corridor_angle_deg, 180, alpha=0.07, color='red', zorder=1)

    ax.set_xlabel('Normalised Episode Time', fontsize=14)
    ax.set_ylabel('Approach Angle (deg)', fontsize=14)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, 185)
    ax.set_title('Approach Angle History  (All MC Runs)', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'approach_angle_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Initial condition scatter (Hill frame)
# ─────────────────────────────────────────────────────────────────────────────
def plot_initial_condition_scatter(all_runs_data, summary_df, output_folder):
    print("Generating Initial Condition Scatter...")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax_xy, ax_xz, ax_yz = axes

    seen_labels = set()
    legend_handles = []

    for idx, run_df in enumerate(all_runs_data):
        if not _has_pos(run_df):
            continue

        row        = summary_df.iloc[idx]
        success    = bool(row['success'])
        end_status = str(row['end_status'])
        color      = _status_color(end_status, success)
        label_key  = 'Docked' if success else _simplify_status(end_status)

        x0, y0, z0 = (
            float(run_df['hill_x'].iloc[0]),
            float(run_df['hill_y'].iloc[0]),
            float(run_df['hill_z'].iloc[0]),
        )

        kw = dict(color=color, s=45, alpha=0.80, edgecolors='k', linewidths=0.3, zorder=3)
        ax_xy.scatter(x0, y0, **kw)
        ax_xz.scatter(x0, z0, **kw)
        ax_yz.scatter(y0, z0, **kw)

        if label_key not in seen_labels:
            seen_labels.add(label_key)
            legend_handles.append(
                Patch(facecolor=color, edgecolor='k', label=label_key)
            )

    for ax in axes:
        ax.scatter(0, 0, color='black', marker='*', s=220, zorder=5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')

    ax_xy.set_xlabel('Hill X / Radial (m)',    fontsize=13)
    ax_xy.set_ylabel('Hill Y / In-Track (m)',   fontsize=13)
    ax_xy.set_title('R–I Plane (X-Y)',          fontsize=14, fontweight='bold')

    ax_xz.set_xlabel('Hill X / Radial (m)',     fontsize=13)
    ax_xz.set_ylabel('Hill Z / Cross-Track (m)', fontsize=13)
    ax_xz.set_title('R–C Plane (X-Z)',           fontsize=14, fontweight='bold')

    ax_yz.set_xlabel('Hill Y / In-Track (m)',    fontsize=13)
    ax_yz.set_ylabel('Hill Z / Cross-Track (m)', fontsize=13)
    ax_yz.set_title('I–C Plane (Y-Z)',           fontsize=14, fontweight='bold')

    legend_handles.append(
        Line2D([0], [0], marker='*', color='black', markersize=11,
               linestyle='None', label='RSO (Origin)')
    )
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles),
               fontsize=12, bbox_to_anchor=(0.5, -0.04), frameon=True)
    fig.suptitle('Initial Condition Coverage  (Hill Frame)', fontsize=17, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    save_path = os.path.join(output_folder, 'initial_condition_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Failure mode breakdown bar chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_failure_mode_breakdown(summary_df, output_folder):
    print("Generating Failure Mode Breakdown...")

    simplified = summary_df['end_status'].apply(_simplify_status)
    n_total    = len(summary_df)

    present    = [c for c in _CATEGORY_ORDER if (simplified == c).any()]
    counts     = [int((simplified == c).sum()) for c in present]
    colors     = [_CATEGORY_COLORS[_CATEGORY_ORDER.index(c)] for c in present]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(present, counts, color=colors, edgecolor='k', zorder=3)

    for bar, cnt in zip(bars, counts):
        pct = 100.0 * cnt / n_total
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{cnt}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=13, fontweight='bold',
        )

    ax.set_ylabel('Count', fontsize=14)
    ax.set_ylim(0, (max(counts) if counts else 1) * 1.30)
    ax.set_title(f'Episode Outcome Breakdown  (n = {n_total})', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'failure_mode_breakdown.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Separation distance time history (all runs)
# ─────────────────────────────────────────────────────────────────────────────
def plot_distance_history(all_runs_data, summary_df, output_folder,
                          standoff_dist=_STANDOFF_DIST,
                          conjunction_r=_CONJUNCTION_R):
    print("Generating Distance Time History...")

    fig, ax = plt.subplots(figsize=(12, 6))
    seen_labels = set()

    for idx, run_df in enumerate(all_runs_data):
        if not (_has_pos(run_df) and 'sim_time' in run_df.columns):
            continue

        row        = summary_df.iloc[idx]
        success    = bool(row['success'])
        end_status = str(row['end_status'])
        color      = _status_color(end_status, success)
        alpha      = 0.65 if success else 0.20

        label_key = 'Docked' if success else _simplify_status(end_status)
        lbl = label_key if label_key not in seen_labels else None
        if lbl:
            seen_labels.add(label_key)

        ax.plot(run_df['sim_time'].values, _separation(run_df).values,
                color=color, alpha=alpha, linewidth=0.8, label=lbl, zorder=2)

    ax.axhline(standoff_dist, color=_ML['medium'], linestyle='--', linewidth=1.8,
               label=f'Standoff / Waypoint  ({standoff_dist:.0f} m)', zorder=4)

    ax.set_xlabel('Simulation Time (s)', fontsize=16)
    ax.set_ylabel('Separation Distance (m)', fontsize=16)
    ax.set_title('Separation Distance History  (All MC Runs)', fontsize=18, fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=13, loc='upper right')
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'distance_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. ΔV consumption vs. separation distance
# ─────────────────────────────────────────────────────────────────────────────
def plot_dv_vs_distance(all_runs_data, summary_df, output_folder, n_bins=60):
    print("Generating ΔV vs Distance Profile...")

    if not any('dV_remaining' in df.columns for df in all_runs_data):
        print("  No dV_remaining data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    all_dist_pts: list = []
    all_dv_pts:   list = []

    for idx, run_df in enumerate(all_runs_data):
        if 'dV_remaining' not in run_df.columns or not _has_pos(run_df):
            continue

        row        = summary_df.iloc[idx]
        success    = bool(row['success'])
        end_status = str(row['end_status'])
        color      = _status_color(end_status, success)

        sep         = _separation(run_df).values
        dv_init     = float(run_df['dV_remaining'].iloc[0])
        dv_consumed = dv_init - run_df['dV_remaining'].values

        ax.plot(sep, dv_consumed, color=color, alpha=0.18, linewidth=0.7, zorder=2)
        all_dist_pts.extend(sep.tolist())
        all_dv_pts.extend(dv_consumed.tolist())

    # Binned mean ± 1σ across all runs
    all_dist_arr = np.array(all_dist_pts)
    all_dv_arr   = np.array(all_dv_pts)

    if len(all_dist_arr) > 0:
        bin_edges   = np.linspace(0.0, all_dist_arr.max(), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_means, bin_stds = [], []

        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (all_dist_arr >= lo) & (all_dist_arr < hi)
            pts  = all_dv_arr[mask]
            if len(pts) >= 3:
                bin_means.append(float(np.mean(pts)))
                bin_stds.append(float(np.std(pts)))
            else:
                bin_means.append(np.nan)
                bin_stds.append(np.nan)

        bm   = np.array(bin_means)
        bs   = np.array(bin_stds)
        valid = ~np.isnan(bm)

        ax.fill_between(bin_centers[valid], (bm - bs)[valid], (bm + bs)[valid],
                        alpha=0.22, color='black', zorder=3)
        ax.plot(bin_centers[valid], bm[valid], 'k-', linewidth=2.2,
                label='Mean ± 1σ  (all runs)', zorder=4)

    # Outcome legend proxies
    outcome_handles = [
        Line2D([0], [0], color=_C['success'],   linewidth=1.5, label='Docked'),
        Line2D([0], [0], color=_C['collision'],  linewidth=1.5, label='Collision'),
        Line2D([0], [0], color=_C['timeout'],    linewidth=1.5, label='Timeout'),
        Line2D([0], [0], color=_C['fuel'],       linewidth=1.5, label='Fuel / Bounds'),
        Line2D([0], [0], color='black',          linewidth=2.2, label='Mean ± 1σ'),
    ]
    ax.legend(handles=outcome_handles, fontsize=11)

    ax.set_xlabel('Separation Distance (m)', fontsize=14)
    ax.set_ylabel('Cumulative ΔV Consumed (m/s)', fontsize=14)
    ax.set_title('ΔV Consumption vs. Separation Distance', fontsize=16, fontweight='bold')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'dv_vs_distance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Reward component heatmap + success-vs-failure comparison bar
# ─────────────────────────────────────────────────────────────────────────────
def plot_reward_heatmap(all_runs_data, summary_df, output_folder):
    print("Generating Reward Component Heatmap...")

    # Discover rew_* columns across all runs (preserve insertion order)
    rew_cols: list = []
    for df in all_runs_data:
        for c in df.columns:
            if c.startswith('rew_') and c not in rew_cols:
                rew_cols.append(c)

    if not rew_cols:
        print("  No rew_* columns found — skipping.")
        return

    # Build matrix: (n_runs × n_components) of cumulative episode reward per component
    n_runs = len(all_runs_data)
    matrix = np.full((n_runs, len(rew_cols)), np.nan)
    for i, run_df in enumerate(all_runs_data):
        for j, col in enumerate(rew_cols):
            if col in run_df.columns:
                matrix[i, j] = float(run_df[col].sum())

    # Sort rows by ascending total_reward so best runs appear at the top
    sort_order     = np.argsort(summary_df['total_reward'].values)
    matrix_sorted  = matrix[sort_order]
    success_sorted = summary_df['success'].values[sort_order].astype(bool)

    # Per-column normalisation to [0, 1] for the heatmap
    mat_norm = matrix_sorted.astype(float).copy()
    for j in range(mat_norm.shape[1]):
        col_vals = mat_norm[:, j]
        finite   = col_vals[np.isfinite(col_vals)]
        if len(finite) == 0:
            continue
        vmin, vmax = finite.min(), finite.max()
        mat_norm[:, j] = (col_vals - vmin) / (vmax - vmin) if vmax > vmin else 0.5

    # Readable column labels  (strip rew_ prefix, split CamelCase)
    def _label(col):
        name = col.replace('rew_', '')
        return re.sub(r'(?<=[a-z])(?=[A-Z])', '\n', name)

    col_labels = [_label(c) for c in rew_cols]

    # ── Layout: heatmap | success strip | comparison bar ─────────────────────
    fig = plt.figure(figsize=(max(12, len(rew_cols) * 2 + 5), 9))
    gs  = gridspec.GridSpec(1, 3, width_ratios=(3, 0.12, 1.2), wspace=0.06)

    # ── Left: Heatmap ────────────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0, 0])
    im = ax_heat.imshow(mat_norm, aspect='auto', cmap='RdYlGn',
                        vmin=0.0, vmax=1.0, interpolation='nearest')

    ax_heat.set_xticks(range(len(col_labels)))
    ax_heat.set_xticklabels(col_labels, rotation=0, ha='center', fontsize=11)
    ax_heat.set_yticks([])
    ax_heat.set_ylabel('Runs  (sorted by total reward  ↑  best)', fontsize=13)
    ax_heat.set_title('Reward Components per Run\n(per-column min–max normalised)',
                      fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02,
                 label='Normalised component value')

    # ── Middle: Success / fail strip ─────────────────────────────────────────
    ax_strip = fig.add_subplot(gs[0, 1])
    strip_data = np.where(success_sorted, 1.0, 0.0).reshape(-1, 1)
    ax_strip.imshow(strip_data, aspect='auto', cmap='RdYlGn',
                    vmin=0.0, vmax=1.0, interpolation='nearest')
    ax_strip.set_xticks([0])
    ax_strip.set_xticklabels(['S/F'], fontsize=10)
    ax_strip.set_yticks([])
    ax_strip.set_title('', fontsize=1)

    # ── Right: Mean component comparison (success vs fail) ───────────────────
    ax_bar = fig.add_subplot(gs[0, 2])

    success_mask = summary_df['success'].values.astype(bool)
    fail_mask    = ~success_mask
    y_pos        = np.arange(len(rew_cols))
    bar_h        = 0.38

    means_succ = (np.nanmean(matrix[success_mask], axis=0)
                  if success_mask.any() else np.zeros(len(rew_cols)))
    means_fail = (np.nanmean(matrix[fail_mask],    axis=0)
                  if fail_mask.any()    else np.zeros(len(rew_cols)))

    ax_bar.barh(y_pos + bar_h / 2, means_succ, bar_h,
                color=_ML['medium'], edgecolor='k', label='Docked',    alpha=0.85)
    ax_bar.barh(y_pos - bar_h / 2, means_fail, bar_h,
                color=_ML['darkest'], edgecolor='k', label='Not Docked', alpha=0.85)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([c.replace('\n', ' ') for c in col_labels], fontsize=11)
    ax_bar.axvline(0, color='black', linewidth=0.8)
    ax_bar.set_xlabel('Mean Cumulative Reward', fontsize=12)
    ax_bar.set_title('Success vs Fail\nMean per Component', fontsize=13, fontweight='bold')
    ax_bar.legend(fontsize=11, loc='lower right')
    ax_bar.grid(True, linestyle='--', alpha=0.5, axis='x')

    fig.suptitle('Reward Component Analysis  (Monte Carlo)', fontsize=17,
                 fontweight='bold', y=1.01)

    save_path = os.path.join(output_folder, 'reward_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Sun angle vs. separation distance (all runs)
# ─────────────────────────────────────────────────────────────────────────────
def plot_sun_angle_vs_distance(all_runs_data, summary_df, output_folder,
                               cone_angle_deg=_SUN_CONE_DEG,
                               cutoff_range=_ILLUM_CUTOFF):
    print("Generating Sun Angle vs. Separation Distance...")

    if not any('sun_angle_deg' in df.columns for df in all_runs_data):
        print("  No sun_angle_deg data — skipping.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    seen_labels = set()

    for idx, run_df in enumerate(all_runs_data):
        if 'sun_angle_deg' not in run_df.columns or not _has_pos(run_df):
            continue

        row        = summary_df.iloc[idx]
        success    = bool(row['success'])
        end_status = str(row['end_status'])
        color      = _status_color(end_status, success)
        alpha      = 0.65 if success else 0.20

        sep    = _separation(run_df).values
        angles = run_df['sun_angle_deg'].values

        label_key = 'Docked' if success else _simplify_status(end_status)
        lbl = label_key if label_key not in seen_labels else None
        if lbl:
            seen_labels.add(label_key)

        ax.plot(sep, angles, color=color, alpha=alpha, linewidth=0.8, label=lbl, zorder=2)

    # Illumination cone limit (horizontal)
    ax.axhline(cone_angle_deg, color='darkorange', linestyle='--', linewidth=1.8,
               label=f'Illumination cone limit  ({cone_angle_deg:.0f}°)', zorder=5)

    # Illumination rewarder cutoff range (vertical — reward disabled below this)
    ax.axvline(cutoff_range, color='royalblue', linestyle=':', linewidth=1.8,
               label=f'Illumination reward cutoff  ({cutoff_range:.0f} m)', zorder=5)

    ax.set_xlabel('Separation Distance (m)', fontsize=14)
    ax.set_ylabel('Sun Angle (deg)', fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 185)
    ax.set_title('Sun Angle vs. Separation Distance  (All MC Runs)', fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=11, loc='upper right')
    plt.tight_layout()

    save_path = os.path.join(output_folder, 'sun_angle_vs_distance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
