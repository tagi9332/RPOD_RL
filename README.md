# RPOD_RL
**Reinforcement Learning for Rendezvous, Proximity Operations, and Docking**

RPOD_RL trains an autonomous **Inspector** satellite to rendezvous with and dock to a **Resident Space Object (RSO)** using deep reinforcement learning. It combines the high-fidelity **[Basilisk](http://hanspeterschaub.info/basilisk/)** astrodynamics simulator with the **[bsk-rl](https://avslab.github.io/bsk_rl/)** Gymnasium wrapper and **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** (PPO) to learn a full approach → waypoint-capture → docking maneuver under realistic orbital dynamics, attitude control, fuel constraints, and illumination geometry.

Everything in the environment — orbital propagation, attitude pointing, sensor/actuator behavior, and mission geometry — is simulated in Basilisk. The RL agent only chooses **when and how hard to thrust** (an impulsive Δv in the RSO's Hill frame) and **how long to drift** before its next decision.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
  - [Satellites](#satellites)
  - [Action Space](#action-space)
  - [Observation Space](#observation-space)
  - [Reward Structure](#reward-structure)
  - [Episode Randomization](#episode-randomization)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Monitoring Training](#monitoring-training)
  - [Curriculum Learning](#curriculum-learning)
  - [Evaluation & Monte Carlo Analysis](#evaluation--monte-carlo-analysis)
  - [HPC / Slurm Submission](#hpc--slurm-submission)
- [Configuration Reference](#configuration-reference)
- [Output Artifacts](#output-artifacts)
- [License](#license)

---

## Overview

The task: an **Inspector** satellite starts 1000–1200 m from a tumbling/oriented **RSO** in a geostationary-altitude orbit, with a limited Δv budget (150 m/s). It must:

1. Navigate toward the RSO's docking-port boresight, preferring the sunlit approach side.
2. Brake and capture a **30 m body-fixed standoff waypoint** in front of the docking port.
3. Perform a controlled final ingress and achieve a **conjunction** (within the docking radius) while inside a **30° approach corridor** around the docking-port boresight.

Success requires balancing four competing objectives — fuel economy, time, approach-corridor alignment, and terminal docking velocity — which the reward function encodes as a hierarchical, phase-aware shaping signal (see [Reward Structure](#reward-structure)).

## How It Works

### Satellites

Two `bsk_rl.sats.Satellite` subclasses are defined in [src/train/docking_sim_training.py](src/train/docking_sim_training.py):

| Satellite | Role | Behavior |
|---|---|---|
| `RSOSat` | Target / chief | Passive — never retasked by the RL agent. Spawns with a randomized GEO-like orbit and attitude each episode. |
| `InspectorSat` | Agent / deputy | The learned policy controls this satellite via impulsive Δv burns and continuously points its instrument boresight at the RSO (custom FSW task, [src/fsw_modules/pointing_fsw.py](src/fsw_modules/pointing_fsw.py)). |

### Action Space

A single custom `bsk_rl` action, [`ImpulsiveThrustHill`](src/actions/impulsive_thrust_hill.py), exposes a 4-dimensional continuous action in `[-1, 1]^4`:

| Index | Meaning | Decoding |
|---|---|---|
| 0–2 | Δv direction/magnitude in the RSO's **Hill frame** | scaled by `MAX_DV` (0.5 m/s), clamped to the sphere of radius `MAX_DV` |
| 3 | Drift duration before the next decision | mapped from `[-1, 1]` to `[2·SIM_DT, MAX_DRIFT_DURATION]` (2–60 s) |

Keeping the raw action space fixed to `[-1, 1]^4` (rather than scaling it directly by physical units) avoids saturating SB3's Gaussian policy at initialization and lets the same policy architecture survive curriculum changes to `MAX_DV`/`MAX_DRIFT_DURATION` mid-training.

### Observation Space

Defined on `InspectorSat.observation_spec` ([src/train/docking_sim_training.py](src/train/docking_sim_training.py)), flattened via `gymnasium.wrappers.FlattenObservation`:

- Remaining Δv budget (normalized)
- Current active Δv reward weight (for weight-randomized training)
- Relative position/velocity to the RSO in the **Hill frame** (global navigation context)
- Relative position/velocity to the RSO in the **RSO body frame** (docking alignment context)
- Signed distance to the 30 m body-fixed standoff waypoint
- Inspector boresight direction expressed in the Hill frame
- Sun direction expressed in the RSO body frame (for illumination-aware approach)
- Normalized mission elapsed time

### Reward Structure

The reward is a hierarchical sum of six components, built by [`get_rewarders()`](src/rewarders/rewarders.py) and weighted by [resources/weights.py](resources/weights.py):

**Phase 0 — long-range approach** (Inspector outside the waypoint capture radius):
- `IlluminationReward` — rewards approaching from the RSO's sunlit side (active beyond 500 m).
- `WaypointPhaseReward` (dense) — log-MSE shaping toward the 30 m standoff waypoint, with a proximity-scaled velocity-braking term.
- `DockingCorridorReward` — rewards boresight-corridor alignment once inside 120 m of the RSO.
- `DeltaVReward` — quadratic penalty on Δv magnitude per burn (discourages wasteful maneuvers).
- `QuadraticTimePenalty` — an integral, step-size-independent time penalty that ramps steeply near the episode's 3-hour limit.

**Phase transition:** a one-time sparse bonus fires the instant the Inspector first enters the waypoint capture sphere (10 m radius); the episode latches into Phase 1 from then on.

**Phase 1 — terminal ingress** (after waypoint capture):
- `WaypointPhaseReward` (dense) — log-MSE shaping toward the docking port itself, with an always-on, stronger velocity penalty to enforce a controlled final approach.

**Terminal events** (`SparseEventReward`, fire once and end the episode):
- **Docking success** — conjunction within the approach-corridor angle of the docking-port boresight → large positive reward, scaled by alignment quality, plus a fuel-efficiency bonus for Δv remaining.
- **Collision** — conjunction outside the corridor → large penalty.
- **Max-range violation** — Inspector exceeds the allowed operating radius → penalty.

An optional **waypoint gate** (`WAYPOINT_GATE_ENABLED` in [src/rewarders/rewarders.py](src/rewarders/rewarders.py)) can require physical waypoint capture before the docking bonus is awarded at all.

### Episode Randomization

[`SatArgRandomizer`](src/randomizers/sat_arg_randomizer_rso_random_inertial.py) randomizes, each episode:
- RSO orbital elements (near-GEO altitude, near-circular).
- RSO attitude — several modes are supported (`random`, `velocity`-aligned, `radial`, `normal`, noisy variants, etc.), selected via `rso_att_type`.
- Inspector's initial relative position/velocity in the Hill frame.
- Optionally, the active Δv-penalty weight (sampled per episode from a truncated Gaussian) so a single policy generalizes across a range of fuel-cost tradeoffs.

`mode="train"` re-randomizes the RSO every reset; `mode="test"` persists the first-generated RSO orbit/attitude across resets, giving a stable scenario for comparing checkpoints during evaluation.

## Repository Layout

```
rpod_rl/
├── resources/                  # All tunable constants (single source of truth)
│   ├── constants.py            #   physical/astrodynamic constants
│   ├── sim_parameters.py       #   sim timing, Δv limits, randomization bounds, sat_args
│   ├── weights.py              #   reward-term weights
│   └── hyperparameters.py      #   PPO hyperparameters
│
├── src/
│   ├── train/                  # Environment definitions + training entry points
│   │   ├── docking_sim_training.py           # RSOSat/InspectorSat/Sb3BksEnv + single-core smoke test
│   │   ├── docking_sim_multi_process.py       # main multi-core PPO training script
│   │   ├── docking_sim_curriculum_multi_process.py  # 3-stage reverse-curriculum training
│   │   └── docking_sim_multi_process_slurm.py # HPC-oriented variant used by slurm/
│   │
│   ├── test/                   # Evaluation / inference scripts
│   │   ├── docking_sim_ref_traj.py           # Monte Carlo rollout + full plotting suite
│   │   └── docking_sim_eval_fixed_state.py   # navigation-error sensitivity study (pinned scenario)
│   │
│   ├── rewarders/               # Modular bsk_rl Reward subclasses (see Reward Structure)
│   ├── randomizers/              # Episode/scenario randomizers
│   ├── actions/                  # Custom ImpulsiveThrust(Hill) continuous action
│   ├── fsw_modules/               # Custom flight-software (pointing) task
│   ├── curriculum/                 # Curriculum-learning parameter schedulers + stage callback
│   └── conjunction_radius_scheduler.py, manual_reward_computation.py
│
├── utils/
│   ├── observations/            # Custom bsk_rl observation functions (Hill/body-frame transforms)
│   ├── plotting/                 # Post-run analysis & visualization (trajectories, Vizard export, MC stats)
│   ├── frame_conversions/         # MRP <-> DCM helpers
│   └── misc/time_est.py           # SB3 callback estimating remaining training time
│
├── slurm/rl_training_submit.sh   # Slurm batch job for CU Boulder's Alpine HPC cluster
├── logs/                          # TensorBoard + CSV training logs (per run, timestamped)
├── models/                        # Saved policy checkpoints (.zip) and best models
├── results/                        # Evaluation plots, Monte Carlo CSVs, Vizard playback data
├── requirements.txt                # Windows/dev pinned dependencies
└── requirements_linux.txt          # Linux/HPC pinned dependencies (uv-compiled)
```

## Installation

### Prerequisites

- **Python 3.12** (pinned: `3.12.12`)
- A C++ toolchain is *not* required — `bsk` (Basilisk) and `bsk-rl` are installed as prebuilt PyPI packages here.
- Windows, Linux, or macOS. Multi-core training spawns one Basilisk simulation per CPU core, so more cores materially speeds up training.

### 1. Clone the repository

```bash
git clone https://github.com/tagi9332/RPOD_RL.git
cd RPOD_RL
```

### 2. Create a virtual environment and install dependencies

Using `uv` (recommended — this is what [slurm/rl_training_submit.sh](slurm/rl_training_submit.sh) uses on HPC):

```bash
uv venv .venv --python 3.12
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

uv pip install -r requirements.txt        # Windows/dev
# or, on Linux:
uv pip install -r requirements_linux.txt
```

Using plain `pip`:

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

> **Note:** [requirements.txt](requirements.txt) was frozen from a Windows environment as UTF-16; if your tooling chokes on it, use [requirements_linux.txt](requirements_linux.txt) as a reference or regenerate with `pip freeze > requirements.txt` (UTF-8) locally.

### 3. Verify the install

```bash
python -c "import Basilisk, bsk_rl, stable_baselines3; print('OK')"
```

## Usage

All commands below are run from the repository root with the virtual environment active. Scripts add the repo root to `PYTHONPATH` implicitly when run as `python -m` or via the `slurm` script's explicit `export PYTHONPATH`; if you hit `ModuleNotFoundError: src`, run from the repo root or set `PYTHONPATH` yourself.

### Training

**Multi-core PPO training (primary entry point):**

```bash
python -m src.train.docking_sim_multi_process
```

This is the main training loop. It:
- Detects available CPU cores and reserves 4 for the OS (`num_cpu = cpu_count - 4`), launching one `SubprocVecEnv` worker per remaining core.
- Trains PPO (`n_steps=512` per env, `batch_size=1024`, `gamma=1.0`, `device="cpu"` — Basilisk simulation, not the network, is the bottleneck) for `total_timesteps=4_000_000` by default.
- Can warm-start from an existing checkpoint — set `LOAD_MODEL = True` and `LOAD_PATH` near the top of `__main__` in [src/train/docking_sim_multi_process.py](src/train/docking_sim_multi_process.py); set `LOAD_MODEL = False` to train from scratch.
- Evaluates against a fixed-orbit eval env every `eval_freq` steps (`EvalCallback`) and checkpoints periodically (`CheckpointCallback`), saving to `models/training_run_<timestamp>/`.
- Logs to `logs/training_run_<timestamp>/` (stdout, CSV, and TensorBoard).
- Optionally randomizes the Δv-penalty weight per episode — set `DV_WEIGHT_MEAN` near the top of the file to a float to enable, or leave `None` to use the fixed weight from `resources/weights.py`.
- Supports optional curriculum schedules (conjunction radius, corridor angle, attitude error, Δv penalty, drift duration, max Δv) — see [Curriculum Learning](#curriculum-learning).
- Ctrl+C safely saves the current model and logs before exiting.

**Single-core smoke test** (sanity-checks the environment/reward wiring in a couple minutes, not intended for real training — `total_timesteps=100`):

```bash
python -m src.train.docking_sim_training
```

### Monitoring Training

```bash
tensorboard --logdir ./logs/
```

Then open the printed `localhost` URL. Each training run's `progress.csv` in `logs/training_run_<timestamp>/` can also be inspected directly (e.g. with `utils/plotting/plot_training_data.py`) or loaded in pandas.

### Curriculum Learning

Two independent curriculum mechanisms are available:

1. **Continuous parameter schedulers** ([src/curriculum/parameter_schedulers.py](src/curriculum/parameter_schedulers.py)) — linearly anneal a single parameter (conjunction radius, corridor angle, attitude pointing error, Δv penalty weight, max drift duration, or max Δv) over training, evaluated against `eval_env` performance. Enable any of these by setting the corresponding `*_SCHEDULE = (initial, final)` tuple near the bottom of `docking_sim_multi_process.py`'s `__main__`; leave `None` to disable.

2. **Discrete 3-stage reverse curriculum** ([src/train/docking_sim_curriculum_multi_process.py](src/train/docking_sim_curriculum_multi_process.py)) — trains "backwards" from the easiest sub-task to the full mission:
   - **Stage 0 — Terminal approach:** spawns 20–24 m from the docking port, already inside the capture sphere; trains the final ingress maneuver in isolation.
   - **Stage 1 — Correct-side capture:** spawns 65–615 m out on the boresight side; trains braking and waypoint capture, chaining into Stage 0 behavior.
   - **Stage 2 — Full mission:** spawns 1800–2000 m out in any direction; trains the complete navigate → capture → dock chain, including wrong-side recovery.
   
   Advancement is automatic and performance-gated: Stage 0→1 requires a 70% conjunction rate over 50 episodes; Stage 1→2 requires 50% over 100 episodes (see `CurriculumStageCallback`). Run with:

   ```bash
   python -m src.train.docking_sim_curriculum_multi_process
   ```

### Evaluation & Monte Carlo Analysis

**Full Monte Carlo rollout with the complete plotting suite** ([src/test/docking_sim_ref_traj.py](src/test/docking_sim_ref_traj.py)):

```bash
python -m src.test.docking_sim_ref_traj
```

Set `model_path`, `num_runs`, and `num_workers` in its `__main__` block. This loads a trained PPO checkpoint, runs `num_runs` episodes in parallel (deterministic policy, randomized scenarios), and writes to `results/`:

- Per-run trajectory, control, reward-breakdown, and impulse plots for the best and worst runs (`results/best_run/`, `results/worst_run/`)
- Aggregate Monte Carlo distributions, summary table, Pareto front (time vs. Δv), failure-mode breakdown, waypoint-capture and approach-angle analysis
- An interactive HTML trajectory viewer (`results/interactive_trajectories.html`)
- Per-run [Vizard](http://hanspeterschaub.info/basilisk/Vizard/Vizard.html) playback files (`results/vizard_data/run_N_vizard.bin`) for 3D visualization in Basilisk's Vizard viewer
- `results/mc_summary_stats.csv` / `results/mc_all_runs_data.csv` — raw tabular results for further analysis

**Navigation-error sensitivity study** ([src/test/docking_sim_eval_fixed_state.py](src/test/docking_sim_eval_fixed_state.py)):

```bash
python -m src.test.docking_sim_eval_fixed_state
```

Pins every scenario variable (epoch, RSO orbit/attitude, Inspector initial state, policy determinism) and varies **only** a per-step Gaussian navigation error injected into the observed relative state (`nav_pos_std`, `nav_vel_std` in `__main__`). Useful for characterizing policy robustness to imperfect relative navigation, independent of scenario geometry. Produces the same plotting suite as above plus per-run navigation-error telemetry.

### HPC / Slurm Submission

[slurm/rl_training_submit.sh](slurm/rl_training_submit.sh) is configured for CU Boulder's Alpine cluster (`amilan` partition, 14 CPUs, 32 GB RAM, 4-hour wall time). It provisions a fresh `uv` virtual environment on scratch storage, installs `requirements_linux.txt`, and runs `src/train/docking_sim_multi_process_slurm.py`. Submit from the `slurm/` directory:

```bash
cd slurm
sbatch rl_training_submit.sh
```

Adjust `--time`, `--cpus-per-task`, `--mem`, and `--mail-user` for your allocation before submitting.

## Configuration Reference

All tunables live in [resources/](resources/) and are re-exported through `resources/__init__.py`, so any script can do `from resources import MAX_DV, dv_reward_weight, ...`. There are no CLI flags — change training behavior by editing these files (or the `__main__` blocks of the training scripts for run-level settings like model paths and timestep counts).

**[resources/sim_parameters.py](resources/sim_parameters.py)** — simulation timing, action limits, and episode geometry:

| Constant | Value | Meaning |
|---|---|---|
| `SIM_TIME` | 10800 s | Episode time limit (3 hours) |
| `SIM_DT` | 1.0 s | Basilisk simulation step |
| `MAX_DV` | 0.5 m/s | Max Δv magnitude per burn |
| `DV_AVAILABLE_INIT` | 150 m/s | Initial fuel budget |
| `MAX_DRIFT_DURATION` | 60 s | Max coast time between decisions |
| `MIN_REL_POS` / `MAX_REL_POS` | 1000 / 1200 m | Inspector spawn range from RSO |
| `MAX_REL_VEL` | 1.0 m/s | Inspector spawn relative-velocity magnitude |
| `CONJUNCTION_RADIUS` | 5 m | Distance defining a "docking" conjunction |
| `approach_corridor_angle_deg` | 30° | Half-angle of the acceptable docking corridor |
| `STANDOFF_DISTANCE` | 30 m | Body-fixed waypoint distance from the docking port |
| `WAYPOINT_CAPTURE_RADIUS` | 10 m | Sphere radius that triggers the Phase 0→1 transition |

**[resources/weights.py](resources/weights.py)** — reward-term weights (see [Reward Structure](#reward-structure) for what each term does).

**[resources/hyperparameters.py](resources/hyperparameters.py)** — PPO hyperparameters: `learning_rate` (3e-4), `entropy_coeff` (0.001), `max_grad_norm` (0.5), `clip_range` (0.1). Note `docking_sim_multi_process.py` also sets `n_steps`, `batch_size`, and `gamma` directly at construction time rather than from this file.

**[resources/constants.py](resources/constants.py)** — physical/astrodynamic constants (gravitational parameters, body radii, J2, solar radiation pressure) per Vallado's *Fundamentals of Astrodynamics and Applications*.

## Output Artifacts

| Directory | Contents |
|---|---|
| `logs/training_run_<timestamp>/` | TensorBoard event files + `progress.csv` (SB3 training metrics) + `evaluations.npz` (EvalCallback history) |
| `models/training_run_<timestamp>/` | `best_model.zip`, periodic `..._checkpoint_<steps>_steps.zip`, and a final saved model |
| `models/archive/`, `models/gold_copy_high_dv/`, etc. | Named/curated checkpoints from past training runs kept for comparison or as warm-start sources |
| `results/` | Evaluation plots, Monte Carlo CSVs, and Vizard playback binaries produced by the `src/test/` scripts |

These directories are populated by running the scripts above — they are not build artifacts checked in for any other purpose, and old runs can be safely deleted once superseded.

## License

MIT — see [LICENSE](LICENSE).
