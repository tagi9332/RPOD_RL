# Standard Imports
import os
import multiprocessing
import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

# BSK-RL and Basilisk imports
from bsk_rl import ConstellationTasking, scene
from Basilisk.architecture import bskLogging
from Basilisk.utilities import RigidBodyKinematics as rbk

# Base training script imports
from src.train import (
    rso_sat_args,
    inspector_sat_args,
    RSOSat,
    InspectorSat,
    Sb3BksEnv,
)

# Import custom satellite argument randomizer
from src.randomizers.sat_arg_randomizer_rso_random_inertial import make_sat_arg_randomizer as sat_arg_randomizer

# Import custom rewarders
from src.rewarders import get_rewarders

# Import custom inspector state randomizer
from src.randomizers.inspector_state_randomizer import generate_random_inspector_state

# Local plotting scripts
from utils.plotting import (
    animate_results,
    plot_control_analysis,
    plot_trajectory_analysis,
    process_sim_data,
    interpolate_to_uniform_time,
    plot_interactive_trajectories,
    plot_mc_distributions,
    plot_all_trajectories,
    plot_summary_table,
    plot_pareto_front,
    plot_single_run_rewards,
    plot_last_30m_views,
    plot_waypoint_analysis,
    plot_angular_velocity_analysis,
    plot_vr_diagram,
    plot_approach_angle_history,
    plot_initial_condition_scatter,
    plot_failure_mode_breakdown,
    plot_distance_history,
    plot_dv_vs_distance,
    plot_reward_heatmap,
    plot_sun_angle_vs_distance,
    vizard_output,
    plot_impulse_histogram,
    plot_impulse_timeseries,
    plot_mean_impulse_history,
)
# Import weights
from resources import (
    approach_corridor_angle_deg,
    inspector_boresight,
    docking_port_boresight,
    STANDOFF_DISTANCE,
    WAYPOINT_CAPTURE_RADIUS,
    MAX_REL_POS,
)

# Import sim parameters
from resources import (
    SIM_TIME,
    SIM_DT,
)

# Set BSK logging level
bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

# ============================================================================
# NAVIGATION-ERROR SENSITIVITY STUDY
# ----------------------------------------------------------------------------
# This is NOT a scenario Monte Carlo (that lives in the RSO-attitude / deputy-
# position sweep elsewhere). Every scenario variable is pinned:
#   * epoch / Sun geometry / eclipse  -> world_args["utc_init"] = FIXED_EPOCH
#   * RSO orbit + RSO attitude         -> drawn once from a fixed reset seed
#   * deputy initial relative state    -> single fixed 6-vector
#   * policy                           -> deterministic (distribution mode)
#   * torch / python RNG               -> set_random_seed(0)
# The ONLY thing that varies between runs is a per-control-step Gaussian error
# added to the relative-state estimate the policy observes, i.e. navigation
# error.  Run i draws that error from an independent stream seeded
# NAV_NOISE_SEED_BASE + i, so every run is a distinct, reproducible noise
# realization on an otherwise identical trajectory problem.
# ============================================================================
FIXED_EPOCH = "2018 SEP 29 21:00:00.000 (UTC)"


# Inference environment wrapper
class InferenceEnv(Sb3BksEnv):
    """
    Inherits the exact step/reset logic from the training environment,
    but tacks on extra telemetry for the plotting scripts and injects a
    Gaussian navigation error into the observed relative state.
    """
    def __init__(self, env, agent_name="Inspector", nav_pos_std=0.0, nav_vel_std=0.0):
        super().__init__(env, agent_name)
        self.sim_rate = getattr(env, 'sim_rate', 1.0)
        self.current_sim_time = 0.0
        self._waypoint_ever_captured = False
        self._waypoint_capture_dist = float('inf')

        # --- Navigation-error injection --------------------------------------
        # One physical error vector [dr_N, dv_N] is drawn each control step and
        # added, consistently, to every observation channel that is derived from
        # the relative translational state.  White noise (independent per step),
        # zero-mean, per-inertial-axis 1-sigma of nav_pos_std / nav_vel_std.
        self.nav_pos_std = float(nav_pos_std)   # [m]
        self.nav_vel_std = float(nav_vel_std)   # [m/s]
        self._noise_rng = np.random.default_rng(0)
        self._obs_idx = None                    # {element_name: [flat indices]}
        self._obs_norm = {}                     # {element_name: normalization const}
        bh = np.asarray(docking_port_boresight, dtype=float)
        self._boresight_hat = bh / np.linalg.norm(bh)
        self._last_nav_err = (0.0, 0.0)

    def set_noise_seed(self, seed):
        """Reseed the navigation-error stream. Call once per Monte Carlo run."""
        self._noise_rng = np.random.default_rng(int(seed))

    # ------------------------------------------------------------------------
    def _resolve_obs_layout(self):
        """Map relative-state observation elements to flat-vector indices + norms.

        Read back from bsk_rl at runtime so it tracks the InspectorSat
        observation_spec (element order and normalization constants) without
        hard-coded indices.
        """
        try:
            builder = self.env.satellites[1].observation_builder
            keys = builder.obs_array_keys()
        except Exception:
            return

        groups: dict[str, list[int]] = {}
        for i, key in enumerate(keys):
            elem = key.split(".")[-1].split("[")[0]
            if elem.endswith("_normd"):
                elem = elem[:-6]
            groups.setdefault(elem, []).append(i)

        wanted = ("r_DC_Hc", "v_DC_Hc", "r_DC_C", "v_DC_C", "dist_to_waypoint")
        self._obs_idx = {e: groups[e] for e in wanted if e in groups}

        norms: dict[str, float] = {}
        for getter in getattr(builder, "observation_spec", []):
            specs = list(getattr(getter, "rel_properties", []) or [])
            specs += list(getattr(getter, "obs_properties", []) or [])
            for p in specs:
                nm = p.get("name", p.get("prop", ""))
                if nm.endswith("_normd"):
                    nm = nm[:-6]
                norms[nm] = float(p.get("norm", 1.0))
        self._obs_norm = norms

    def _apply_nav_noise(self, obs, dcm_CN, dcm_HN):
        """Add one consistent Gaussian navigation error to the observed rel. state.

        dcm_CN : DCM from inertial N to RSO body C  (MRP2C(rso.sigma_BN))
        dcm_HN : DCM from inertial N to RSO Hill H  (rso.dynamics.HN)
        """
        if self.nav_pos_std <= 0.0 and self.nav_vel_std <= 0.0:
            return obs
        if self._obs_idx is None:
            self._resolve_obs_layout()
        if not self._obs_idx:
            return obs

        obs = np.array(obs, dtype=np.float64, copy=True)

        dr_N = (self._noise_rng.normal(0.0, self.nav_pos_std, size=3)
                if self.nav_pos_std > 0.0 else np.zeros(3))
        dv_N = (self._noise_rng.normal(0.0, self.nav_vel_std, size=3)
                if self.nav_vel_std > 0.0 else np.zeros(3))
        self._last_nav_err = (float(np.linalg.norm(dr_N)), float(np.linalg.norm(dv_N)))

        # Same physical error, expressed in each frame the observation uses.
        dr_H, dv_H = dcm_HN @ dr_N, dcm_HN @ dv_N
        dr_C, dv_C = dcm_CN @ dr_N, dcm_CN @ dv_N
        deltas = {
            "r_DC_Hc": dr_H,
            "v_DC_Hc": dv_H,
            "r_DC_C": dr_C,
            "v_DC_C": dv_C,
            # dist_to_waypoint = r_DC_C . boresight_hat - standoff
            "dist_to_waypoint": np.array([float(np.dot(dr_C, self._boresight_hat))]),
        }

        for elem, idx in self._obs_idx.items():
            default_norm = MAX_REL_POS if elem == "dist_to_waypoint" else 1.0
            norm = self._obs_norm.get(elem, default_norm)
            comps = np.atleast_1d(deltas[elem])
            for flat_i, comp in zip(idx, comps):
                obs[flat_i] += comp / norm

        return obs  # float64, matches the bsk_rl observation space dtype

    def reset(self, **kwargs):
        self.current_sim_time = 0.0
        self._waypoint_ever_captured = False
        self._waypoint_capture_dist = float('inf')
        self._obs_idx = None
        obs, info = super().reset(**kwargs)

        # Corrupt the initial observation too, so the first action is drawn from
        # the same noisy-estimate assumption as the rest of the episode.
        try:
            rso = self.env.satellites[0]
            dcm_CN = rbk.MRP2C(np.array(rso.dynamics.sigma_BN))
            dcm_HN = np.array(rso.dynamics.HN)
            obs = self._apply_nav_noise(obs, dcm_CN, dcm_HN)
        except Exception:
            pass
        return obs, info

    def step(self, action):
        # 1. Run normal training step logic
        obs, reward, terminated, truncated, info = super().step(action)
        self.current_sim_time = self.env.simulator.sim_time

        # 2. Extract extra telemetry just for inference plots
        rso = self.env.satellites[0]
        inspector = self.env.satellites[1]

        rso_r_N = np.array(rso.dynamics.r_BN_N)
        rso_v_N = np.array(rso.dynamics.v_BN_N)
        insp_r_N = np.array(inspector.dynamics.r_BN_N)
        insp_v_N = np.array(inspector.dynamics.v_BN_N)
        dist = np.linalg.norm(insp_r_N - rso_r_N)

        rso_sigma_BN = np.array(rso.dynamics.sigma_BN)
        dcm_BN_rso = rbk.MRP2C(rso_sigma_BN)

        # Body-frame relative position and velocity
        r_DC_C = dcm_BN_rso @ (insp_r_N - rso_r_N)
        v_DC_C = dcm_BN_rso @ (insp_v_N - rso_v_N)

        # ----------------------------------------------------------------------------
        # METRIC 1: Waypoint capture — direct computation, same logic as DataStore
        # Bypasses the patch mechanism so it works regardless of DataStore lifecycle.
        # ----------------------------------------------------------------------------
        waypoint_B = docking_port_boresight * STANDOFF_DISTANCE
        dist_to_waypoint = float(np.linalg.norm(r_DC_C - waypoint_B))
        if dist_to_waypoint < WAYPOINT_CAPTURE_RADIUS and not self._waypoint_ever_captured:
            self._waypoint_ever_captured = True
            self._waypoint_capture_dist = dist_to_waypoint

        # ----------------------------------------------------------------------------
        # METRIC 2: Sun Angle (angle between inspector→RSO and inspector→Sun vectors)
        # ----------------------------------------------------------------------------
        try:
            world = self.env.simulator.world
            sun_msg = world.gravFactory.spiceObject.planetStateOutMsgs[world.sun_index].read()
            r_sun_N = np.array(sun_msg.PositionVector)
            r_insp_to_rso = rso_r_N - insp_r_N
            r_insp_to_sun = r_sun_N - insp_r_N
            dist_rso = np.linalg.norm(r_insp_to_rso)
            dist_sun = np.linalg.norm(r_insp_to_sun)
            if dist_rso > 1e-6 and dist_sun > 1e-6:
                rho_hat = r_insp_to_rso / dist_rso
                sun_hat = r_insp_to_sun / dist_sun
                cos_sun = np.clip(np.dot(rho_hat, sun_hat), -1.0, 1.0)
                sun_angle_deg = float(np.degrees(np.arccos(cos_sun)))
            else:
                sun_angle_deg = 0.0
        except Exception:
            sun_angle_deg = 0.0

        # ----------------------------------------------------------------------------
        # METRIC 3: Approach Angle (Inspector's position relative to RSO's docking port)
        # ----------------------------------------------------------------------------
        # Standoff waypoint in Hill frame (body-z boresight * 30m, rotated to Hill)
        dcm_HN = np.array(rso.dynamics.HN)
        r_waypoint_H = dcm_HN @ (dcm_BN_rso.T @ (docking_port_boresight * STANDOFF_DISTANCE))

        if dist > 1e-6:
            r_rel_B_hat = dcm_BN_rso @ ((insp_r_N - rso_r_N) / dist)
            approach_angle_rad = np.arccos(np.clip(np.dot(r_rel_B_hat, docking_port_boresight), -1.0, 1.0))
            approach_angle_deg = np.degrees(approach_angle_rad)
        else:
            approach_angle_deg = 0.0

        # ----------------------------------------------------------------------------
        # METRIC 3: Pointing Error (Inspector's camera aiming at the RSO)
        # ----------------------------------------------------------------------------
        sigma_BN_insp = np.array(inspector.dynamics.sigma_BN)
        dcm_BN_insp = rbk.MRP2C(sigma_BN_insp)

        if dist > 1e-6:
            u_Target_B = dcm_BN_insp @ ((rso_r_N - insp_r_N) / dist)
            pointing_error_rad = np.arccos(np.clip(np.dot(u_Target_B, inspector_boresight), -1.0, 1.0))
        else:
            pointing_error_rad = 0.0

        # ----------------------------------------------------------------------------
        # 3. Inject navigation error into the observation the policy will act on
        # ----------------------------------------------------------------------------
        obs = self._apply_nav_noise(obs, dcm_BN_rso, dcm_HN)

        # ----------------------------------------------------------------------------
        # Save to Info Dictionary
        # ----------------------------------------------------------------------------
        if "metrics" in info:
            info["metrics"]["sim_time"] = self.current_sim_time

            # Attitude Metrics
            info["metrics"]["approach_angle_deg"] = approach_angle_deg
            info["metrics"]["sun_angle_deg"] = sun_angle_deg
            info["metrics"]["pointing_error"] = pointing_error_rad
            info["metrics"]["rso_sigma_BN"] = rso_sigma_BN
            info["metrics"]["r_waypoint_H"] = r_waypoint_H

            # Body-frame velocity (for axial/lateral docking velocity analysis)
            info["metrics"]["v_DC_C"] = v_DC_C

            # Waypoint capture (sticky flag, reset each episode in reset())
            info["metrics"]["waypoint_captured"] = self._waypoint_ever_captured
            info["metrics"]["waypoint_capture_dist"] = self._waypoint_capture_dist

            # Navigation error actually applied this step (magnitudes)
            info["metrics"]["nav_err_pos"] = self._last_nav_err[0]
            info["metrics"]["nav_err_vel"] = self._last_nav_err[1]

            # Hardware Metrics
            insp_torque_cmd = inspector.dynamics.satellite.data_store.satellite.dynamics.satellite.fsw.rwMotorTorque.rwMotorTorqueOutMsg.payloadPointer.motorTorque[0:3]
            wheel_speeds = inspector.data_store.satellite.fsw.satellite.dynamics.wheel_speeds
            info["metrics"]["torque_cmd"] = insp_torque_cmd
            info["metrics"]["wheel_speeds"] = wheel_speeds

            # Consumables
            info["metrics"]["dV_remaining"] = inspector.fsw.dv_available if hasattr(inspector.fsw, 'dv_available') else 0.0

            # Reward component telemetry
            if hasattr(self.env, 'rewarder'):
                rewarder_iterable = self.env.rewarder if isinstance(self.env.rewarder, (list, tuple)) else [self.env.rewarder]
                for rew_obj in rewarder_iterable:
                    name = rew_obj.__class__.__name__
                    if name == "ResourceReward":
                        val = rew_obj.reward.get("Inspector", 0.0) if hasattr(rew_obj, 'reward') else 0.0
                    else:
                        val = getattr(rew_obj, 'last_reward', 0.0)
                    info["metrics"][f"rew_{name}"] = val

        return obs, reward, terminated, truncated, info

# --- Inference Loop ---
def enable_eval_reward_telemetry(env):
    """
    Patches each rewarder's calculate_reward to store the last per-step component
    value in .last_reward so InferenceEnv.step() can log reward breakdowns.
    Handles rewarders nested inside a ComposedReward one level deep.
    """
    raw_rewarders = env.rewarder if isinstance(env.rewarder, (list, tuple)) else [env.rewarder]
    flat_rewarders = []
    for r in raw_rewarders:
        if r.__class__.__name__ == "ComposedReward":
            flat_rewarders.extend(r.rewarders)
        else:
            flat_rewarders.append(r)

    for rew_obj in flat_rewarders:
        if hasattr(rew_obj, 'calculate_reward'):
            def create_patched_method(obj, orig_fn):
                def patched_calculate_reward(new_data_dict):
                    reward_dict = orig_fn(new_data_dict)
                    obj.last_reward = reward_dict.get("Inspector", 0.0)
                    return reward_dict
                return patched_calculate_reward

            rew_obj.calculate_reward = create_patched_method(rew_obj, rew_obj.calculate_reward)


def _summarize_run(run_df, run_idx):
    """Build the per-run summary row (shared by every worker)."""
    total_sim_time = run_df["sim_time"].max() if "sim_time" in run_df.columns else 0.0
    final_dist = np.linalg.norm([run_df.iloc[-1]["hill_x"], run_df.iloc[-1]["hill_y"], run_df.iloc[-1]["hill_z"]])

    # --- SUCCESS LOGIC ---
    conjunction = bool(run_df.iloc[-1].get("docked_state", False))
    final_angle = run_df.iloc[-1].get("approach_angle_deg", 180.0)

    # Success is strictly a conjunction WITHIN the cone limit
    success = conjunction and (final_angle <= approach_corridor_angle_deg)

    if success:
        end_status = f"Docked ({final_angle:.1f}°)"
    elif conjunction:
        end_status = f"Collision ({final_angle:.1f}°)"
    elif total_sim_time >= SIM_TIME * 0.99:
        end_status = "Timeout"
    else:
        end_status = "Fuel Exhausted / Bounds Viol."

    # --- WAYPOINT CAPTURE STATS ---
    if "waypoint_captured" in run_df.columns:
        wp_captured = bool(run_df["waypoint_captured"].max())
        if wp_captured and "waypoint_capture_dist" in run_df.columns:
            finite_dists = run_df["waypoint_capture_dist"].replace([np.inf, -np.inf], np.nan).dropna()
            wp_capture_dist = float(finite_dists.min()) if len(finite_dists) > 0 else np.nan
        else:
            wp_capture_dist = np.nan
    else:
        wp_captured = False
        wp_capture_dist = np.nan

    return {
        "run_id": run_idx + 1, "episode_length": len(run_df),
        "total_sim_time": total_sim_time, "end_status": end_status, "success": success,
        "final_distance": final_dist, "waypoint_captured": wp_captured,
        "waypoint_capture_dist": wp_capture_dist,
    }


def _run_chunk(args):
    """Worker: builds its own env + model and runs a chunk of navigation-error runs.

    Every run in every worker uses the *identical* pinned scenario (fixed epoch,
    fixed RSO orbit/attitude from ``scenario_seed``, fixed deputy state). The only
    thing that changes run-to-run is the navigation-noise stream seed.
    """
    (run_indices, model_path, output_folder, fixed_inspector_state,
     scenario_seed, nav_noise_seed_base, nav_pos_std, nav_vel_std, fixed_epoch) = args

    bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)
    set_random_seed(0)  # torch / python / numpy hygiene (policy is deterministic anyway)

    scenario = scene.SphericalRSO(n_points=100, radius=1.0, theta_max=np.radians(30), range_max=250, theta_solar_max=np.radians(60))
    rewarders = get_rewarders()

    env_bsk = ConstellationTasking(
        satellites=[RSOSat("RSO", sat_args=rso_sat_args), InspectorSat("Inspector", sat_args=inspector_sat_args)],
        sat_arg_randomizer=sat_arg_randomizer(mode="test", rso_att_type="velocity", fixed_inspector_state=fixed_inspector_state),
        scenario=scenario,
        rewarder=rewarders,
        world_args={"utc_init": fixed_epoch},   # pin Sun geometry / eclipse across all runs
        time_limit=SIM_TIME,
        sim_rate=SIM_DT,
        log_level="WARNING"
    )
    enable_eval_reward_telemetry(env_bsk)

    inf_env = InferenceEnv(env_bsk, nav_pos_std=nav_pos_std, nav_vel_std=nav_vel_std)
    env = FlattenObservation(inf_env)

    try:
        model = PPO.load(model_path, device="cpu")
    except FileNotFoundError:
        print(f"[PID {os.getpid()}] ERROR: Model not found at {model_path}")
        return []

    chunk_results = []

    for run_idx in run_indices:
        noise_seed = nav_noise_seed_base + run_idx
        print(f"[PID {os.getpid()}] --- Executing Run {run_idx + 1} (nav-noise seed {noise_seed}) ---")

        inf_env.set_noise_seed(noise_seed)
        obs, _ = env.reset(seed=scenario_seed)   # identical pinned scenario every run

        vizard_output(env_bsk, output_folder, run_idx)

        done = False
        run_data_log = []
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)

            if "metrics" in info:
                metrics = info["metrics"]
                flat_metrics = {"run_id": run_idx + 1}
                flat_metrics["reward"] = float(reward)  # Ensure step reward is logged

                for k, v in metrics.items():
                    key_name = "hill" if k == "r_DC_Hc" else k
                    if isinstance(v, (np.ndarray, list)) and len(v) == 3:
                        flat_metrics[f"{key_name}_x"] = v[0]
                        flat_metrics[f"{key_name}_y"] = v[1]
                        flat_metrics[f"{key_name}_z"] = v[2]
                    else:
                        flat_metrics[key_name] = v

                run_data_log.append(flat_metrics)

        run_df = pd.DataFrame(run_data_log)

        summary = _summarize_run(run_df, run_idx)
        summary["total_reward"] = total_reward
        summary["nav_noise_seed"] = noise_seed
        summary["nav_pos_std"] = nav_pos_std
        summary["nav_vel_std"] = nav_vel_std

        # restore column order roughly matching the original schema
        summary = {
            "run_id": summary["run_id"], "total_reward": summary["total_reward"],
            "episode_length": summary["episode_length"], "total_sim_time": summary["total_sim_time"],
            "end_status": summary["end_status"], "success": summary["success"],
            "final_distance": summary["final_distance"], "waypoint_captured": summary["waypoint_captured"],
            "waypoint_capture_dist": summary["waypoint_capture_dist"],
            "nav_noise_seed": summary["nav_noise_seed"],
            "nav_pos_std": summary["nav_pos_std"], "nav_vel_std": summary["nav_vel_std"],
        }

        chunk_results.append((run_df, summary))

    return chunk_results


def run_monte_carlo_inference(model_path: str, output_folder: str, num_runs: int = 30, num_workers: int = 14,
                              nav_pos_std: float = 1.0, nav_vel_std: float = 0.01,
                              scenario_seed: int = 12345, nav_noise_seed_base: int = 1000,
                              fixed_epoch: str = FIXED_EPOCH) -> tuple[list, pd.DataFrame]:
    # Deterministically generate the single fixed deputy initial relative state.
    # (The RSO orbit + attitude are pinned separately, via the fixed reset seed
    # passed to every env.reset() inside the workers.)
    np.random.seed(scenario_seed)
    fixed_inspector_state = generate_random_inspector_state()

    print("--- Fixed scenario (identical for every run) ---")
    print(f"  epoch            : {fixed_epoch}")
    print(f"  scenario seed    : {scenario_seed}")
    print(f"  deputy r0 (Hill) : {np.array(fixed_inspector_state[:3])}")
    print(f"  deputy v0 (Hill) : {np.array(fixed_inspector_state[3:])}")
    print("--- Navigation-error dispersion (the ONLY thing varied) ---")
    print(f"  position 1-sigma : {nav_pos_std} m per inertial axis")
    print(f"  velocity 1-sigma : {nav_vel_std} m/s per inertial axis")
    print(f"  noise seeds      : {nav_noise_seed_base} .. {nav_noise_seed_base + num_runs - 1}")

    # Distribute run indices round-robin across workers for even load balancing
    all_indices = list(range(num_runs))
    actual_workers = min(num_workers, num_runs)
    chunks = [all_indices[i::actual_workers] for i in range(actual_workers)]
    args = [(chunk, model_path, output_folder, fixed_inspector_state,
             scenario_seed, nav_noise_seed_base, nav_pos_std, nav_vel_std, fixed_epoch)
            for chunk in chunks]

    print(f"Running {num_runs} navigation-error realizations across {actual_workers} workers...")

    with multiprocessing.Pool(processes=actual_workers) as pool:
        chunk_results_list = pool.map(_run_chunk, args)

    # Flatten and sort by run_id to restore original ordering
    all_results = [item for chunk in chunk_results_list for item in chunk]
    all_results.sort(key=lambda x: x[1]["run_id"])

    all_runs_data = [r[0] for r in all_results]
    summary_stats = [r[1] for r in all_results]

    print("\n=== Navigation-Error Sensitivity Summary ===")
    summary_df = pd.DataFrame(summary_stats)
    print(f"Nav error 1-sigma: pos {nav_pos_std} m/axis, vel {nav_vel_std} m/s/axis")
    print(f"Mean Reward: {summary_df['total_reward'].mean():.2f} +/- {summary_df['total_reward'].std():.2f}")
    print(f"Success Rate: {(summary_df['success'].sum() / num_runs) * 100:.1f}%")
    print(f"Average Final Distance: {summary_df['final_distance'].mean():.2f}m "
          f"(std {summary_df['final_distance'].std():.2f}m)")
    n_wp = int(summary_df['waypoint_captured'].sum())
    print(f"Waypoint Capture Rate: {(n_wp / num_runs) * 100:.1f}%  ({n_wp}/{num_runs} runs)")
    mean_wp_dist = summary_df['waypoint_capture_dist'].dropna().mean()
    if not np.isnan(mean_wp_dist):
        print(f"Mean Waypoint Capture Distance from Center: {mean_wp_dist:.2f}m")

    summary_df.to_csv(os.path.join(output_folder, "mc_summary_stats.csv"), index=False)
    pd.concat(all_runs_data, ignore_index=True).to_csv(os.path.join(output_folder, "mc_all_runs_data.csv"), index=False)

    return all_runs_data, summary_df

if __name__ == "__main__":
    output_folder = "results"
    os.makedirs(output_folder, exist_ok=True)

    # --------------------------- Configuration ---------------------------
    model_path = r"models\MC_model.zip"
    num_runs = 50
    num_workers = 10

    # Navigation-error sensitivity study: pin every scenario variable, disperse
    # only the per-step Gaussian error added to the observed relative state.
    nav_pos_std = 3.0          # [m]   per-axis 1-sigma position-estimate error
    nav_vel_std = 0.03         # [m/s] per-axis 1-sigma velocity-estimate error
    scenario_seed = 12345      # pins RSO orbit, RSO attitude, deputy initial state
    nav_noise_seed_base = 1000 # run i uses navigation-noise seed nav_noise_seed_base + i
    fixed_epoch = FIXED_EPOCH  # pins Sun geometry / eclipse
    # -------------------------------------------------------------------

    all_runs_data, summary_df = run_monte_carlo_inference(
        model_path, output_folder, num_runs=num_runs, num_workers=num_workers,
        nav_pos_std=nav_pos_std, nav_vel_std=nav_vel_std,
        scenario_seed=scenario_seed, nav_noise_seed_base=nav_noise_seed_base,
        fixed_epoch=fixed_epoch,
    )

    raw_runs_data = all_runs_data  # keep raw per-step data for impulse analysis
    if all_runs_data:
        all_runs_data = [interpolate_to_uniform_time(df) for df in all_runs_data]  # type: ignore[union-attr]

    if all_runs_data:
        plot_all_trajectories(all_runs_data, summary_df, output_folder)
        plot_interactive_trajectories(all_runs_data, summary_df, output_folder)
        plot_summary_table(summary_df, output_folder)
        plot_mc_distributions(all_runs_data, summary_df, output_folder)
        plot_pareto_front(all_runs_data, summary_df, output_folder)
        plot_last_30m_views(all_runs_data, summary_df, output_folder)
        plot_waypoint_analysis(all_runs_data, summary_df, output_folder)
        plot_angular_velocity_analysis(all_runs_data, summary_df, output_folder)
        plot_vr_diagram(all_runs_data, summary_df, output_folder)
        plot_approach_angle_history(all_runs_data, summary_df, output_folder)
        plot_initial_condition_scatter(all_runs_data, summary_df, output_folder)
        plot_failure_mode_breakdown(summary_df, output_folder)
        plot_distance_history(all_runs_data, summary_df, output_folder)
        plot_dv_vs_distance(all_runs_data, summary_df, output_folder)
        plot_reward_heatmap(all_runs_data, summary_df, output_folder)
        plot_sun_angle_vs_distance(all_runs_data, summary_df, output_folder)
        plot_impulse_histogram(raw_runs_data, summary_df, output_folder)
        plot_mean_impulse_history(raw_runs_data, summary_df, output_folder)

    # Trim down to worst and best runs
    worst_run_id = summary_df.sort_values(by="total_reward", ascending=True).iloc[0]["run_id"] #type: ignore
    worst_run_df = all_runs_data[worst_run_id - 1]  # Adjust for 0-indexing #type: ignore
    best_run_id = summary_df.sort_values(by="total_reward", ascending=False).iloc[0]["run_id"] #type: ignore
    best_run_df = all_runs_data[best_run_id - 1]  # Adjust for 0-indexing #type: ignore


    # Process data and save
    processed_data_best = process_sim_data(best_run_df)
    processed_data_worst = process_sim_data(worst_run_df)

    # Create best/worst run folders
    if not os.path.exists(os.path.join(output_folder, "best_run")):
        os.makedirs(os.path.join(output_folder, "best_run"))
    if not os.path.exists(os.path.join(output_folder, "worst_run")):
        os.makedirs(os.path.join(output_folder, "worst_run"))

    # Save processed data for best/worst runs

    plot_control_analysis(processed_data_best, os.path.join(output_folder, "best_run"))
    plot_trajectory_analysis(processed_data_best, os.path.join(output_folder, "best_run"))
    plot_single_run_rewards(best_run_df, os.path.join(output_folder, "best_run"), prefix="best_")

    plot_control_analysis(processed_data_worst, os.path.join(output_folder, "worst_run"))
    plot_trajectory_analysis(processed_data_worst, os.path.join(output_folder, "worst_run"))
    plot_single_run_rewards(worst_run_df, os.path.join(output_folder, "worst_run"), prefix="worst_")

    plot_impulse_timeseries(best_run_df, worst_run_df, output_folder)

    print("\nAll inferences and plots complete!")
