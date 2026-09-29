"""
Plot rollout and curriculum metrics over training steps from SB3 progress.csv logs.

Usage:
    python src/analysis/plot_training_progress.py logs/training_run_2026-09-15_08-28-15
    python src/analysis/plot_training_progress.py path/to/progress.csv --smooth 10 --show
    python src/analysis/plot_training_progress.py logs/run_A logs/run_B -o compare.pdf   # overlay runs

Each argument may be a progress.csv file or a run directory containing one.
Panels: mean episode reward, mean episode length, and one panel per
curriculum/* column found.
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

# Run(s) to plot when no paths are given on the command line (e.g. VS Code "Run Python File").
# Run directories or progress.csv files, relative to the repo root or absolute; list several to overlay.
# Empty -> newest training_run_* in logs/.
RUN_PATHS = ["logs/training_run_2026-09-15_08-28-15"]

# Optional figure title. None -> auto-generated from the run name(s). The --title CLI flag overrides this.
TITLE = "Maximum Delta-V Bound Curriculum Training"

# Fixed categorical order, one color per run (never cycled past 8)
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
AXIS = "#b5b4ae"

# column -> (panel title, y-axis label)
ROLLOUT_PANELS = {
    "rollout/ep_rew_mean": ("Mean Episode Reward", "Reward"),
    "rollout/ep_len_mean": ("Mean Episode Length", "Length [steps]"),
}
CURRICULUM_LABELS = {
    "curriculum/max_dv": ("Curriculum: Maximum Delta-V per Step", "Max Delta-V Bound [m/s]"),
    "curriculum/conjunction_radius": ("Curriculum: Conjunction Radius", "Radius [m]"),
    "curriculum/corridor_angle_deg": ("Curriculum: Approach Corridor Half-Angle", "Angle [deg]"),
}
# column -> multiplier applied before plotting, so logged values match what the sim actually applies
COLUMN_SCALE = {
    "curriculum/max_dv": 0.5,
}

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


def curriculum_labels(col):
    if col in CURRICULUM_LABELS:
        return CURRICULUM_LABELS[col]
    name = col.split("/", 1)[1].replace("_", " ").title()
    return f"Curriculum: {name}", name


def series(df, col, smooth):
    """Return (x, raw, smoothed) for a column, dropping rows where it wasn't logged."""
    s = df[[X_COL, col]].dropna()
    raw = s[col] * COLUMN_SCALE.get(col, 1.0)
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


def plot_runs(runs, smooth, out_path, show, title):
    if len(runs) > len(RUN_COLORS):
        raise ValueError(f"At most {len(RUN_COLORS)} runs can be overlaid at once")

    panels = [(col, *labels) for col, labels in ROLLOUT_PANELS.items()
              if any(col in df.columns for _, df in runs)]
    curriculum_cols = sorted({c for _, df in runs for c in df.columns if c.startswith("curriculum/")})
    panels += [(col, *curriculum_labels(col)) for col in curriculum_cols]
    if not panels:
        raise ValueError("No rollout/* or curriculum/* columns found to plot")

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.6 * len(panels) + 1.0),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    fig.patch.set_facecolor("white")

    for ax, (col, panel_title, ylabel) in zip(axes, panels):
        style_axis(ax, panel_title, ylabel)
        for (name, df), color in zip(runs, RUN_COLORS):
            if col not in df.columns:
                continue
            x, raw, sm = series(df, col, smooth)
            # Curriculum values are deterministic schedules; smoothing them adds nothing
            if smooth > 1 and col.startswith("rollout/"):
                ax.plot(x, raw, color=color, linewidth=0.8, alpha=0.2)
                ax.plot(x, sm, color=color, linewidth=2, label=name)
            else:
                ax.plot(x, raw, color=color, linewidth=2, label=name)

    axes[-1].set_xlabel("Training Timesteps", color=TEXT_SECONDARY)

    # Header: title + subtitle describing what's shown
    if title is None:
        title = "Training Progress" + (f" — {runs[0][0]}" if len(runs) == 1 else " — Run Comparison")
    max_steps = max(df[X_COL].max() for _, df in runs)
    subtitle = f"Rollout and curriculum metrics over {max_steps / 1e6:.1f}M timesteps"
    if smooth > 1:
        subtitle += f"  ·  rollout curves smoothed with a {smooth}-iteration rolling mean"
    fig.text(0.012, 0.995, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT_PRIMARY)
    fig.text(0.012, 0.995 - 0.34 / fig.get_figheight(), subtitle, ha="left", va="top",
             fontsize=10, color=TEXT_SECONDARY)

    header_in = 0.6 if len(runs) == 1 else 0.9
    fig.tight_layout(rect=(0, 0, 1, 1 - header_in / fig.get_figheight()), h_pad=1.6)

    if len(runs) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", ncol=min(len(runs), 4), frameon=False,
                   fontsize=9, labelcolor=TEXT_SECONDARY, handlelength=1.6,
                   bbox_to_anchor=(0.005, 1 - 0.62 / fig.get_figheight()))

    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved plot to {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot rollout and curriculum metrics from progress.csv")
    parser.add_argument("paths", nargs="*",
                        help="progress.csv file(s) or run directory(ies) "
                             "(default: RUN_PATHS in this file, else the newest run in logs/)")
    parser.add_argument("-o", "--out", help="Output path; extension sets format, e.g. .png/.pdf/.svg "
                                            "(default: training_progress.png next to the first CSV)")
    parser.add_argument("-s", "--smooth", type=int, default=1,
                        help="Rolling-mean window for rollout curves, in log rows (default: 1, no smoothing)")
    parser.add_argument("-t", "--title", default=TITLE,
                        help="Custom figure title (default: TITLE in this file, else auto-generated)")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    paths = args.paths or RUN_PATHS or [latest_run_dir()]
    csv_paths = [resolve_csv(p) for p in paths]
    runs = [(run_label(p.parent.name), load_progress(p)) for p in csv_paths]
    out_path = Path(args.out) if args.out else csv_paths[0].parent / "training_progress.png"
    plot_runs(runs, args.smooth, out_path, args.show, args.title)


if __name__ == "__main__":
    main()
