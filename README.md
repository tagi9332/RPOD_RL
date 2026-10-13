# RPOD_RL
**Reinforcement Learning for Rendezvous, Proximity Operations, and Docking**

Trains an **Inspector** satellite to rendezvous with and dock to a **Resident Space Object (RSO)** using PPO ([Stable-Baselines3](https://stable-baselines3.readthedocs.io/)) on top of the [Basilisk](http://hanspeterschaub.info/basilisk/) astrodynamics simulator via [bsk-rl](https://avslab.github.io/bsk_rl/).

The Inspector starts ~1 km from a randomly oriented RSO near GEO with a limited Δv budget. It must capture a 30 m standoff waypoint in front of the RSO's docking port, then dock inside a 30° approach corridor. The agent only decides **how to thrust** (an impulsive Δv in the RSO's Hill frame) and **how long to drift** before its next decision. Everything else is simulated by Basilisk.

## How It Works

**Action**: [`ImpulsiveThrustHill`](src/actions/impulsive_thrust_hill.py), a continuous action in `[-1, 1]^4`. Indices 0–2 are the Hill-frame Δv, scaled by `MAX_DV`. Index 3 is the drift duration, mapped to `[2·SIM_DT, MAX_DRIFT_DURATION]`.

**Observation** (defined in [docking_sim_training.py](src/train/docking_sim_training.py)):
- remaining Δv and the active Δv penalty weight
- relative position/velocity in both the Hill and RSO body frames
- offset to the standoff waypoint
- Inspector boresight and sun direction
- elapsed time

**Reward** (built by [`get_rewarders()`](src/rewarders/rewarders.py), weighted in [weights.py](resources/weights.py)):

| Term | Purpose |
|---|---|
| `WaypointPhaseReward` | Log-MSE shaping toward the waypoint (Phase 0), then toward the docking port (Phase 1), plus a velocity-braking term. Gives a one-time bonus on waypoint capture. |
| `DockingCorridorReward` | Rewards boresight-corridor alignment at close range |
| `IlluminationReward` | Rewards approaching from the RSO's sunlit side |
| `DeltaVReward` | Quadratic Δv penalty per burn |
| `QuadraticTimePenalty` | Time penalty that ramps up near the episode limit |
| `SparseEventReward` | Terminal events. A conjunction gives a docking reward scaled by alignment, plus a fuel bonus if it's inside the corridor. Exceeding the max range gives a penalty. |

**Randomization**: [`SatArgRandomizer`](src/randomizers/sat_arg_randomizer_rso_random_inertial.py) randomizes the RSO orbit and attitude (set the mode with `rso_att_type`), the Inspector's initial relative state, and optionally the Δv penalty weight each episode. In `mode="test"` the RSO stays fixed across resets.

## Layout

```
resources/     tunable constants: sim_parameters, weights, hyperparameters, constants
src/train/     environment definition + training scripts
src/test/      evaluation / Monte Carlo scripts
src/rewarders/ reward terms
src/randomizers/, src/actions/, src/fsw_modules/, src/curriculum/
utils/         observations, plotting, frame conversions, misc callbacks
logs/ models/ results/   generated outputs
```

## Setup

Requires Python 3.12.

```bash
git clone https://github.com/tagi9332/RPOD_RL.git && cd RPOD_RL
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run all scripts from the repo root. There are no CLI flags. Settings live in [resources/](resources/) and in each script's `__main__` block (model paths, timesteps, schedules).

| Task | Command |
|---|---|
| Train (multi-core PPO) | `python -m src.train.docking_sim_multi_process` |
| Reverse-curriculum training (3 stages) | `python -m src.train.docking_sim_curriculum_multi_process` |
| Smoke test (single-core) | `python -m src.train.docking_sim_training` |
| Monte Carlo evaluation + plots | `python -m src.test.docking_sim_ref_traj` |
| Navigation-error sensitivity | `python -m src.test.docking_sim_eval_fixed_state` |
| Monitor training | `tensorboard --logdir ./logs/` |

Training notes:
- To warm-start from a checkpoint, set `LOAD_MODEL` / `LOAD_PATH`.
- To randomize the Δv weight, set `DV_WEIGHT_MEAN`.
- To enable a parameter curriculum, set any `*_SCHEDULE = (initial, final)`.
- Ctrl+C saves the model before exiting.

Outputs go to `logs/training_run_<timestamp>/` and `models/training_run_<timestamp>/`. Evaluation scripts write plots, CSVs, and Vizard playback files to `results/`.

## License

MIT, see [LICENSE](LICENSE).
