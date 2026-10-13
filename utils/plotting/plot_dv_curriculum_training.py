"""
Plot reward and delta-V usage against the dv-penalty-weight curriculum from an SB3 progress.csv log.

Usage:
    python utils/plotting/plot_dv_curriculum_training.py logs/training_run_2026-10-01_15-36-21
    python utils/plotting/plot_dv_curriculum_training.py path/to/progress.csv --smooth 10 --dv-smooth 30 --show

The argument may be a progress.csv file or a run directory containing one.
Panels (shared x-axis):
    top    - mean episode reward
    middle - mean episode delta-V
    bottom - curriculum: delta-V penalty weight
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

X_COL = "time/total_timesteps"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"

# Run to plot when no path is given on the command line (e.g. VS Code "Run Python File").
# Run directory or progress.csv file, relative to the repo root or absolute. None -> newest training_run_* in logs/.
RUN_PATH = "logs/training_run_2026-10-01_15-36-21"

# Optional figure title. None -> auto-generated from the run name. The --title CLI flag overrides this.
TITLE = "Delta-V Penalty Weight Curriculum Training"

# Drop log rows before this many timesteps to hide the early-training transient (dv spike while
# the policy is still random), which otherwise sets the y-axis scale. 0 -> plot everything.
# The --start-step CLI flag overrides this.
START_STEP = 300_000

# Rolling-mean windows, in log rows. Delta-V is noisier relative to its trend than reward, so it gets a
# heavier window. The --smooth / --dv-smooth CLI flags override these; 1 disables smoothing.
SMOOTH = 10
DV_SMOOTH = 30

EP_REW_COL = "rollout/ep_rew_mean"
DV_COL = "rollout/ep_dv_mean"
DV_WEIGHT_COL = "curriculum/dv_penalty_weight"

COLOR_BLUE = "#2a78d6"
COLOR_DV = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
AXIS = "#b5b4ae"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})


def resolve_csv(path_str):
    path = Path(path_str)
    if not path.exists() and not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_dir():
        path = path / "progress.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No progress.csv found at {path}")
    return path


def load_progress(csv_path):
    df = pd.read_csv(csv_path)
    if X_COL not in df.columns:
        raise KeyError(f"{csv_path} has no '{X_COL}' column")
    return df.sort_values(X_COL)


def latest_run_dir():
    runs = sorted(p for p in LOGS_DIR.glob("training_run_*") if (p / "progress.csv").is_file())
    if not runs:
        raise FileNotFoundError(f"No training_run_* directories with a progress.csv in {LOGS_DIR}")
    return runs[-1]


def run_label(dir_name):
    """training_run_2026-09-15_08-28-15 -> 'Run 2026-09-15 08:28'."""
    m = re.match(r"training_run_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-\d{2}$", dir_name)
    return f"Run {m.group(1)} {m.group(2)}:{m.group(3)}" if m else dir_name


def series(df, col, smooth):
    """Return (x, raw, smoothed) for a column, dropping rows where it wasn't logged."""
    s = df[[X_COL, col]].dropna()
    raw = s[col]
    smoothed = raw.rolling(smooth, min_periods=1).mean() if smooth > 1 else raw
    return s[X_COL], raw, smoothed


def format_steps(x, _pos):
    if x >= 1e6:
        return f"{x / 1e6:g}M"
    if x >= 1e3:
        return f"{x / 1e3:g}k"
    return f"{x:g}"


def style_axis(ax, title, ylabel):
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontweight="semibold", pad=8)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=TEXT_SECONDARY, length=3)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(format_steps))
    ax.margins(x=0.01)


def plot_line(ax, df, col, smooth, color):
    """Faint raw trace under a bold smoothed line (or just the raw line when unsmoothed)."""
    x, raw, sm = series(df, col, smooth)
    if smooth > 1:
        ax.plot(x, raw, color=color, linewidth=0.8, alpha=0.4)
    ax.plot(x, sm, color=color, linewidth=2)


def plot_run(name, df, smooth, dv_smooth, out_path, show, title):
    for col in (EP_REW_COL, DV_COL, DV_WEIGHT_COL):
        if col not in df.columns:
            raise KeyError(f"progress.csv has no '{col}' column")

    fig, (ax_rew, ax_dv, ax_cur) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.patch.set_facecolor("white")

    # Top: mean episode reward
    style_axis(ax_rew, "Mean Episode Reward", "Reward")
    plot_line(ax_rew, df, EP_REW_COL, smooth, COLOR_BLUE)

    # Middle: mean episode delta-V
    style_axis(ax_dv, "Mean Episode Delta-V".format(dv_smooth), "Delta-V Used [m/s]")
    plot_line(ax_dv, df, DV_COL, dv_smooth, COLOR_DV)

    # Bottom: curriculum schedule (deterministic, so never smoothed)
    style_axis(ax_cur, "Curriculum: Delta-V Penalty Weight", "Penalty Weight")
    x, raw, _ = series(df, DV_WEIGHT_COL, 1)
    ax_cur.plot(x, raw, color=COLOR_BLUE, linewidth=2)
    ax_cur.set_xlabel("Training Timesteps", color=TEXT_SECONDARY)

    # Header: title + subtitle describing what's shown
    if title is None:
        title = f"Training Progress — {name}"
    subtitle = f"Reward, delta-V usage and curriculum over {df[X_COL].max() / 1e6:.1f}M timesteps"
    if df[X_COL].min() > 0:
        subtitle += f"  ·  first {format_steps(df[X_COL].min(), None)} steps omitted"
    windows = [f"{w}-iteration rolling mean for {what}" for w, what in ((smooth, "reward"), (dv_smooth, "delta-V"))
               if w > 1]
    if windows:
        subtitle += "  ·  " + ", ".join(windows)
    fig.text(0.012, 0.995, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT_PRIMARY)
    fig.text(0.012, 0.995 - 0.34 / fig.get_figheight(), subtitle, ha="left", va="top",
             fontsize=10, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()), h_pad=1.6)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot reward, delta-V usage and dv-weight curriculum "
                                                 "from progress.csv")
    parser.add_argument("path", nargs="?",
                        help="progress.csv file or run directory "
                             "(default: RUN_PATH in this file, else the newest run in logs/)")
    parser.add_argument("-o", "--out", help="Output path; extension sets format, e.g. .png/.pdf/.svg "
                                            "(default: dv_curriculum_progress.png next to the CSV)")
    parser.add_argument("-s", "--smooth", type=int, default=SMOOTH,
                        help=f"Rolling-mean window for reward, in log rows (default: {SMOOTH}; 1 disables)")
    parser.add_argument("--dv-smooth", type=int, default=DV_SMOOTH,
                        help=f"Rolling-mean window for delta-V, in log rows (default: {DV_SMOOTH}; 1 disables)")
    parser.add_argument("-t", "--title", default=TITLE,
                        help="Custom figure title (default: TITLE in this file, else auto-generated)")
    parser.add_argument("--start-step", type=int, default=START_STEP,
                        help="Omit log rows before this timestep to hide the early transient "
                             f"(default: START_STEP in this file, {START_STEP:,}; 0 plots everything)")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    csv_path = resolve_csv(args.path or RUN_PATH or latest_run_dir())
    df = load_progress(csv_path)
    df = df[df[X_COL] >= args.start_step]
    out_path = Path(args.out) if args.out else csv_path.parent / "dv_curriculum_progress.png"
    plot_run(run_label(csv_path.parent.name), df, args.smooth, args.dv_smooth, out_path, args.show, args.title)


if __name__ == "__main__":
    main()
