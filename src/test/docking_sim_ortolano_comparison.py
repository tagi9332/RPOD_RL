"""
Ortolano (2018) Sec. 15.3.3, Monte Carlo dispersion analysis of Scenario 1,
Case 3 (Final Approach), Example 3: RL policy vs. Tables 15.8 / 15.9.

Ortolano's setup
    * Chief on a circular Landsat-like LEO (Table 13.2, ~98.7 min period):
          a = 7074.99742 km, e = 0, i = 0.488692190, RAAN = 0.349065850,
          argp = 1.46607657, nu = 0.78539816 (rad)
      giving n = 1.06091165e-3 rad/s.
    * Example 3 (Table 13.10): deputy at rest 100 m out on the v-bar, to rest
      20 m out on the v-bar, t_f = 0.6 chief revolutions, 20 deg approach cone.
          r_ref(t0) = [0, 100, 0] m,  v_ref(t0) = 0  (Hill [x radial, y along-track, z cross-track])
    * Dispersions (Eqs. 15.1-15.2), added to the initial state:
          e_r ~ N(0, sigma_r^2 I),  e_v ~ N(0, n^2 sigma_r^2 I),
          sigma_r = p_r / 100 * ||r_ref(t0)||
      with p_r = 10 % (sigma_r = 10 m, sigma_v ~ 0.01 m/s) and
           p_r =  5 % (sigma_r =  5 m, sigma_v ~ 0.005 m/s), 10 MC runs each.

What this script does
---------------------
Deploys the trained policy in the Basilisk env (degree-10 gravity + Sun, i.e.
"with perturbations") from a nominal (undispersed) start and N_MC dispersed
starts per p_r level, and records the delta-v and time it takes to first come
within TARGET_RANGE (20 m) of the chief.  The episode is stopped there by
default (STOP_AT_TARGET).  The largest approach-cone angle reached on the way
in is checked against Ortolano's 20 deg.

Ortolano's trajectory ends at rest at 20 m, while the policy (trained to dock)
arrives with some closing speed.  dv_rl_target_plus_stop adds the arrival speed
as the impulsive delta-v that would null it, for a like-for-like "arrive and
hold" number.

"""

# Standard Imports
import os
import time
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

# BSK-RL and Basilisk imports
from bsk_rl import ConstellationTasking, scene
from bsk_rl.utils.orbital import cd2hill
from Basilisk.architecture import bskLogging
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import RigidBodyKinematics as rbk

# Base training script imports
from src.train import (
    rso_sat_args,
    inspector_sat_args,
    RSOSat,
    InspectorSat,
    Sb3BksEnv,
)
from src.randomizers.sat_arg_randomizer_rso_random_inertial import make_sat_arg_randomizer as sat_arg_randomizer
from src.rewarders import get_rewarders

from resources import (
    SIM_TIME,
    SIM_DT,
    approach_corridor_angle_deg,
    docking_port_boresight,
)

bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

MU_EARTH = 3.986004418e14  # m^3/s^2 (matches the randomizer)
FIXED_EPOCH = "2018 SEP 29 21:00:00.000 (UTC)"

# ============================================================================
# SCENARIO DEFINITION - Ortolano 2018, Scenario 1 Case 3 Example 3, with dispersions
# ============================================================================
# Chief orbital elements, Table 13.2 (the thesis prints a as "7.07499742 km";
# n = 1.06091165e-3 rad/s only works out for 7074.99742 km)
CHIEF_OE = {
    "a": 7074.99742e3,     # [m]
    "e": 0.0,
    "i": 0.488692190,      # [rad]
    "Omega": 0.349065850,  # [rad]
    "omega": 1.46607657,   # [rad]
    "f": 0.78539816,       # [rad]
}

R0_REF_H = np.array([0.0, 100.0, 0.0])       # [m]   nominal initial Hill position (v-bar)
V0_REF_H = np.zeros(3)                       # [m/s] nominal initial relative velocity
TARGET_RANGE = 20.0                          # [m]   final desired position on the v-bar
TF_REV = 0.6                                 # [rev] Ortolano's transfer time
APPROACH_CONE_DEG = 20.0                     # [deg] Ortolano's approach cone half-angle
STOP_AT_TARGET = True                        # end the episode at TARGET_RANGE

# Hold the RSO docking port on the +v-bar (LVLH-synchronous spin about orbit normal)
RSO_LVLH_HOLD = True

# Ortolano Example 3 nominal (Table 13.10 results): (dv_opt, dv_sim) [m/s]
ORTOLANO_NOMINAL = (0.061, 0.062)

# Ortolano Monte Carlo dv, p_r [%] -> [(dv_opt, dv_sim) per MC run] [m/s]
ORTOLANO_MC = {
    10.0: [  # Table 15.8
        (0.073, 0.074), (0.069, 0.069), (0.068, 0.068), (0.046, 0.047), (0.075, 0.076),
        (0.082, 0.083), (0.057, 0.058), (0.062, 0.063), (0.066, 0.066), (0.059, 0.060),
    ],
    5.0: [   # Table 15.9
        (0.077, 0.077), (0.068, 0.068), (0.070, 0.070), (0.066, 0.067), (0.058, 0.059),
        (0.065, 0.066), (0.084, 0.084), (0.058, 0.059), (0.078, 0.079), (0.065, 0.066),
    ],
}
MC_LEVELS = tuple(ORTOLANO_MC)

# Colors (MATLAB default 'lines' color order)
CASE_COLORS = {1: "#0072BD", 2: "#D95319", 3: "#EDB120", 4: "#7E2F8E", 5: "#77AC30"}
RUN_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30",
              "#4DBEEE", "#A2142F", "#000000", "#808080", "#FF00FF"]
C_ORT_SIM = "#0072BD"   # MATLAB blue   - Ortolano simulation dv
C_RL = "#D95319"        # MATLAB orange - RL policy (-> target)
C_RL_STOP = "#EDB120"   # MATLAB yellow - RL policy (-> target + stop)
C_ORT_OPT = "#7E2F8E"   # MATLAB purple - Ortolano RPO planner dv
C_INK = "#3a3a38"


# ============================================================================
# CHIEF ORBIT / RSO ATTITUDE
# ============================================================================
def make_chief_oe(a, e=0.0, i=0.0, Omega=0.0, omega=0.0, f=0.0):
    """Basilisk ClassicElements from SI / radian values."""
    oe = orbitalMotion.ClassicElements()
    for k, v in dict(a=a, e=e, i=i, Omega=Omega, omega=omega, f=f).items():
        setattr(oe, k, v)
    return oe


def mean_motion(oe):
    return np.sqrt(MU_EARTH / oe.a**3)


def build_chief_state(oe, port="vbar", lvlh_hold=RSO_LVLH_HOLD):
    """Chief orbit + RSO attitude with body +z (docking port) on +v-bar or +r-bar."""
    r_N, v_N = orbitalMotion.elem2rv(MU_EARTH, oe)
    i_n = np.cross(r_N, v_N)
    i_n /= np.linalg.norm(i_n)

    # Same construction as the randomizer's "velocity" / "radial" modes
    body_z = (v_N if port == "vbar" else r_N) / np.linalg.norm(v_N if port == "vbar" else r_N)
    body_y = i_n
    body_x = np.cross(body_y, body_z)
    body_x /= np.linalg.norm(body_x)
    sigma_init = rbk.C2MRP(np.array([body_x, body_y, body_z]))

    # Hill frame rotates at n about +h; body_y == h_hat, so omega_BN_B = [0, n, 0]
    omega_init = np.array([0.0, mean_motion(oe), 0.0]) if lvlh_hold else np.zeros(3)

    return {"chief_orbit": oe, "sigma_init": sigma_init, "omega_init": omega_init}


# ============================================================================
# RL ROLLOUTS IN BASILISK
# ============================================================================
class ComparisonEnv(Sb3BksEnv):
    """Training env + sim time, approach angle and docking-port/v-bar alignment."""

    def _extras(self):
        rso = self.env.satellites[0]
        insp = self.env.satellites[1]
        rc, vc = np.array(rso.dynamics.r_BN_N), np.array(rso.dynamics.v_BN_N)
        rd, vd = np.array(insp.dynamics.r_BN_N), np.array(insp.dynamics.v_BN_N)
        r_H, v_H = cd2hill(rc, vc, rd, vd)

        dcm_CN = rbk.MRP2C(np.array(rso.dynamics.sigma_BN))
        bore_N = dcm_CN.T @ docking_port_boresight
        dist = np.linalg.norm(rd - rc)
        approach = np.degrees(np.arccos(np.clip(np.dot(bore_N, (rd - rc) / max(dist, 1e-9)), -1, 1)))
        vbar_N = vc / np.linalg.norm(vc)
        port_vbar = np.degrees(np.arccos(np.clip(np.dot(bore_N, vbar_N), -1, 1)))

        return {
            "sim_time": float(self.env.simulator.sim_time),
            "x": r_H[0], "y": r_H[1], "z": r_H[2],
            "vx": v_H[0], "vy": v_H[1], "vz": v_H[2],
            "range": float(dist),
            "dv_used": float(inspector_sat_args["dv_available_init"] - insp.fsw.dv_available),
            "approach_angle_deg": float(approach),
            "port_vbar_angle_deg": float(port_vbar),
        }

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        info["row"] = {**self._extras(), "docked": False}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["row"] = {**self._extras(), "docked": bool(info["metrics"]["docked_state"])}
        return obs, reward, terminated, truncated, info


def _run_chunk(args):
    """Worker: build one env + model and roll out a list of (key, ic) tasks."""
    tasks, model_path, chief_state, scenario_seed, time_limit = args

    bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)
    set_random_seed(0)

    randomizer = sat_arg_randomizer(mode="test", rso_att_type="velocity",
                                    fixed_inspector_state=tasks[0][1])
    randomizer._persistent_rso_state = dict(chief_state)  # pin chief orbit + attitude

    # bsk_rl flags a conjunction (= docked, episode ends) at range < r_RSO + r_Inspector;
    # shrink the inspector radius so that sum equals TARGET_RANGE
    insp_radius = TARGET_RANGE - rso_sat_args["conjunction_radius"]  # type: ignore[operator]
    insp_args = {**inspector_sat_args, "conjunction_radius": insp_radius}

    env_bsk = ConstellationTasking(
        satellites=[RSOSat("RSO", sat_args=rso_sat_args), InspectorSat("Inspector", sat_args=insp_args)],
        sat_arg_randomizer=randomizer,
        scenario=scene.SphericalRSO(n_points=100, radius=1.0, theta_max=np.radians(30),
                                    range_max=250, theta_solar_max=np.radians(60)),
        rewarder=get_rewarders(),
        world_args={"utc_init": FIXED_EPOCH},
        time_limit=time_limit,
        sim_rate=SIM_DT,
        log_level="ERROR",
    )
    core = ComparisonEnv(env_bsk)
    core.scheduled_conjunction_radius = insp_radius  # reset() re-applies this to sat_args
    env = FlattenObservation(core)
    model = PPO.load(model_path, device="cpu")

    out = []
    for key, ic in tasks:
        randomizer.fixed_inspector_state = np.asarray(ic)
        print(f"[PID {os.getpid()}] p_r = {key[0]:g} %, run {key[1]}")
        obs, info = env.reset(seed=scenario_seed)
        rows = [info["row"]]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            rows.append(info["row"])
            done = bool(terminated or truncated)
            if STOP_AT_TARGET and info["row"]["range"] <= TARGET_RANGE:
                done = True
        out.append((key, pd.DataFrame(rows)))
    return out


# ============================================================================
# ANALYSIS
# ============================================================================
def analyze_run(df):
    """Metrics for one RL run, evaluated at the first sample inside TARGET_RANGE."""
    hit = np.flatnonzero(df["range"].to_numpy() <= TARGET_RANGE)
    final = df.iloc[-1]
    upto = df.iloc[: hit[0] + 1] if hit.size else df
    max_cone = float(upto["approach_angle_deg"].max())
    row = {
        "rl_docked": bool(final["docked"]),
        "rl_success": bool(final["docked"]) and final["approach_angle_deg"] <= approach_corridor_angle_deg,
        "rl_final_range": float(final["range"]),
        "rl_t_final_s": float(final["sim_time"]),
        "dv_rl_total": float(final["dv_used"]),
        "max_approach_angle_deg": max_cone,
        "within_cone": max_cone <= APPROACH_CONE_DEG,
        "max_port_vbar_deg": float(df["port_vbar_angle_deg"].max()),
    }
    if hit.size:
        s = df.iloc[hit[0]]
        speed = float(np.linalg.norm(s[["vx", "vy", "vz"]].to_numpy(dtype=float)))
        row.update(
            rl_reached_target=True,
            rl_t_target_s=float(s["sim_time"]),
            rl_range_at_target=float(s["range"]),
            rl_speed_at_target=speed,
            rl_approach_angle_at_target=float(s["approach_angle_deg"]),
            dv_rl_target=float(s["dv_used"]),
            dv_rl_target_plus_stop=float(s["dv_used"]) + speed,
        )
    else:
        row.update(rl_reached_target=False, rl_t_target_s=np.nan, rl_range_at_target=np.nan,
                   rl_speed_at_target=np.nan, rl_approach_angle_at_target=np.nan,
                   dv_rl_target=np.nan, dv_rl_target_plus_stop=np.nan)
    return row


def _ortolano_dv(pr, run):
    """Ortolano (dv_opt, dv_sim) for a run; run 0 = nominal, runs past his 10 = NaN."""
    if run == 0:
        return ORTOLANO_NOMINAL
    mc = ORTOLANO_MC.get(pr, [])
    return mc[run - 1] if run <= len(mc) else (np.nan, np.nan)


def print_comparison(table):
    for pr in MC_LEVELS:
        sub = table[table["pr"] == pr]
        s_r, s_v = sub["sigma_r"].iloc[0], sub["sigma_v"].iloc[0]
        print(f"\n=== MC p_r = {pr:g} %  (sigma_r = {s_r:.2f} m, sigma_v = {s_v:.4f} m/s), "
              f"v-bar 100 m -> {TARGET_RANGE:g} m; dv in m/s ===")
        print(f"  {'run':>3} {'r0 [x, y, z] m':>22} {'Ort opt':>8} {'Ort sim':>8} {'dv->tgt':>8} "
              f"{'dv+stop':>8} {'v@tgt':>7} {'t@tgt s':>8} {'max cone':>9} {'in cone':>8}")
        for _, r in sub.iterrows():
            r0 = f"[{r['ic_x']:.1f}, {r['ic_y']:.1f}, {r['ic_z']:.1f}]"
            print(f"  {int(r['run']):>3} {r0:>22} {r['dv_ort_opt']:>8.3f} {r['dv_ort_sim']:>8.3f} "
                  f"{r['dv_rl_target']:>8.4f} {r['dv_rl_target_plus_stop']:>8.4f} {r['rl_speed_at_target']:>7.4f} "
                  f"{r['rl_t_target_s']:>8.0f} {r['max_approach_angle_deg']:>9.2f} {str(bool(r['within_cone'])):>8}")

    nom = table[table["run"] == 0].iloc[0]
    print(f"\n=== Summary: mean [min, max] dv in m/s  (nominal: Ortolano opt/sim "
          f"{ORTOLANO_NOMINAL[0]:.3f}/{ORTOLANO_NOMINAL[1]:.3f}, RL -> tgt {nom['dv_rl_target']:.4f}, "
          f"+ stop {nom['dv_rl_target_plus_stop']:.4f}) ===")
    for pr in MC_LEVELS:
        sub = table[(table["pr"] == pr) & (table["run"] > 0)]
        print(f"  p_r = {pr:>4g} %  ({len(sub)} RL runs, reached {TARGET_RANGE:g} m: "
              f"{int(sub['rl_reached_target'].sum())}, in cone: {int(sub['within_cone'].sum())})")
        for name, col in (("Ortolano planner", "dv_ort_opt"), ("Ortolano sim", "dv_ort_sim"),
                          (f"RL -> {TARGET_RANGE:g} m", "dv_rl_target"),
                          (f"RL -> {TARGET_RANGE:g} m + stop", "dv_rl_target_plus_stop")):
            v = sub[col].dropna()
            print(f"    {name:<20} {v.mean():.4f} [{v.min():.4f}, {v.max():.4f}]")
    print(f"  Max docking-port drift off v-bar : {table['max_port_vbar_deg'].max():.2f} deg")


# ============================================================================
# PLOTTING
# ============================================================================
def _style(ax):
    ax.grid(True, color="#e4e3dc", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=C_INK)


def _cone(ax, length=110.0):
    """Approach cone about +v-bar, drawn in a (y, other-axis) plane."""
    off = length * np.tan(np.radians(APPROACH_CONE_DEG))
    ax.plot([0, length], [0, off], color="#9b9a92", lw=0.8, ls="--")
    ax.plot([0, length], [0, -off], color="#9b9a92", lw=0.8, ls="--",
            label=f"{APPROACH_CONE_DEG:g} deg approach cone")


def _run_style(run, n_runs):
    """Color + legend label for an MC run (labels only when the legend stays readable)."""
    label = f"MC run {run}" if n_runs <= len(RUN_COLORS) else None
    return RUN_COLORS[(run - 1) % len(RUN_COLORS)], label


def plot_trajectories(runs, out_dir):
    """In-plane (y-x) and out-of-plane (y-z) views per p_r level, like Ortolano Figs. 15.15/15.17."""
    nominal = runs[(0.0, 0)]
    fig, axes = plt.subplots(len(MC_LEVELS), 2, figsize=(14, 5.5 * len(MC_LEVELS)), squeeze=False)
    for (ax_ip, ax_op), pr in zip(axes, MC_LEVELS):
        mc = {k[1]: df for k, df in runs.items() if k[0] == pr}
        for run, df in mc.items():
            c, lab = _run_style(run, len(mc))
            for ax, k in ((ax_ip, "x"), (ax_op, "z")):
                ax.plot(df["y"], df[k], color=c, lw=1.0, label=lab)
                ax.plot(df["y"].iloc[0], df[k].iloc[0], "o", ms=4, mfc="white", mec=c, mew=1)
        for ax, k, ylab, title in ((ax_ip, "x", "Radial x [m]", "in-plane"),
                                   (ax_op, "z", "Cross-track z [m]", "out-of-plane")):
            ax.plot(nominal["y"], nominal[k], color=C_INK, lw=1.4, ls="--", label="Nominal")
            _cone(ax)
            ax.plot(0, 0, "*", color=C_INK, ms=12, label="Chief")
            ax.plot(TARGET_RANGE, 0, "s", ms=6, mfc="none", mec=C_INK, label=f"{TARGET_RANGE:g} m v-bar target")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("Along-track y [m]")
            ax.set_ylabel(ylab)
            ax.set_title(f"p_r = {pr:g} %, {title}", color=C_INK)
            _style(ax)
        ax_op.legend(fontsize=7, frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Final approach MC (Ortolano 2018 Sc. 1, Case 3, Ex. 3) - RL policy", color=C_INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectories_mc.png"), dpi=200)
    plt.close(fig)


def _set_axes_equal_3d(ax, pts):
    """Equal scaling on a 3D axis: a cube around all plotted points."""
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))


def plot_trajectories_3d(runs, out_dir, length=110.0):
    """Isometric 3D view (orthographic, equal axes) per p_r level; plotted as (y, z, x) so radial is up."""
    nominal = runs[(0.0, 0)]
    fig = plt.figure(figsize=(7.5 * len(MC_LEVELS), 7))
    # cone about +v-bar: a few generators plus the rim at the far end
    r_cone = length * np.tan(np.radians(APPROACH_CONE_DEG))
    th = np.linspace(0, 2 * np.pi, 73)
    for i, pr in enumerate(MC_LEVELS):
        ax = fig.add_subplot(1, len(MC_LEVELS), i + 1, projection="3d", proj_type="ortho")
        mc = {k[1]: df for k, df in runs.items() if k[0] == pr}
        pts = [np.zeros((1, 3))]
        for run, df in mc.items():
            c, lab = _run_style(run, len(mc))
            ax.plot(df["y"], df["z"], df["x"], color=c, lw=1.0, label=lab)
            ax.scatter(df["y"].iloc[0], df["z"].iloc[0], zs=df["x"].iloc[0], s=12,
                       facecolors="white", edgecolors=c, linewidths=1)
            pts.append(df[["y", "z", "x"]].to_numpy())
        ax.plot(nominal["y"], nominal["z"], nominal["x"], color=C_INK, lw=1.4, ls="--", label="Nominal")
        pts.append(nominal[["y", "z", "x"]].to_numpy())
        ax.plot(np.full_like(th, length), r_cone * np.cos(th), r_cone * np.sin(th),
                color="#9b9a92", lw=0.8, ls="--", label=f"{APPROACH_CONE_DEG:g} deg approach cone")
        for a in np.radians([0, 90, 180, 270]):
            ax.plot([0, length], [0, r_cone * np.cos(a)], [0, r_cone * np.sin(a)],
                    color="#9b9a92", lw=0.8, ls="--")
        ax.scatter(0, 0, zs=0, marker="*", color=C_INK, s=120, label="Chief")
        ax.scatter(TARGET_RANGE, 0, zs=0, marker="s", s=30, facecolors="none", edgecolors=C_INK,
                   label=f"{TARGET_RANGE:g} m v-bar target")
        _set_axes_equal_3d(ax, np.vstack(pts))
        ax.view_init(elev=np.degrees(np.arctan(1 / np.sqrt(2))), azim=-135)  # isometric
        ax.set_xlabel("Along-track y [m]")
        ax.set_ylabel("Cross-track z [m]")
        ax.set_zlabel("Radial x [m]")
        ax.set_title(f"p_r = {pr:g} %, isometric", color=C_INK)
        ax.tick_params(colors=C_INK, labelsize=7)
    fig.suptitle("Final approach MC (Ortolano 2018 Sc. 1, Case 3, Ex. 3) - RL policy, 3D", color=C_INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    # one shared legend below the panels; inside a 3D axis it covers the box
    handles, labels = fig.axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=(len(labels) + 2) // 3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(out_dir, "trajectories_mc_3d.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_control(runs, out_dir):
    """Impulse history and cumulative dv vs. sim time [s] per p_r level, like Ortolano Figs. 15.16/15.18."""
    fig, axes = plt.subplots(len(MC_LEVELS), 2, figsize=(14, 4.5 * len(MC_LEVELS)), squeeze=False)
    for (ax_imp, ax_cum), pr in zip(axes, MC_LEVELS):
        mc = {k[1]: df for k, df in runs.items() if k[0] == pr}
        for run, df in mc.items():
            c, lab = _run_style(run, len(mc))
            dv = np.diff(df["dv_used"].to_numpy(), prepend=0.0)
            burn = dv > 1e-9
            # a step's burn is applied at the start of that step (= previous row's time)
            t_burn = df["sim_time"].shift(1, fill_value=0.0).to_numpy()[burn]
            ax_imp.vlines(t_burn, 0, dv[burn], color=c, lw=1, alpha=0.7)
            ax_cum.plot(df["sim_time"], df["dv_used"], color=c, lw=1.0, label=lab)
        for ax in (ax_imp, ax_cum):
            ax.set_xlim(left=0.0)
            ax.set_xlabel("Time [s]")
            _style(ax)
        ax_imp.set_ylabel("Impulse |$\\Delta v$| [m/s]")
        ax_cum.set_ylabel("Cumulative $\\Delta v$ [m/s]")
        ax_imp.set_title(f"p_r = {pr:g} %, impulses", color=C_INK)
        ax_cum.set_title(f"p_r = {pr:g} %, cumulative", color=C_INK)
        ax_cum.legend(fontsize=7, frameon=False, ncol=2, loc="center right")
    fig.suptitle(f"RL control history, v-bar 100 m -> {TARGET_RANGE:g} m with dispersions", color=C_INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "control_mc.png"), dpi=200)
    plt.close(fig)


def plot_dv_summary(table, out_dir):
    """dv distribution per p_r level: Ortolano planner / simulation vs. RL -> target (+ residual velocity)."""
    groups = [
        ("Ortolano\nplanner", C_ORT_OPT, "dv_ort_opt"),
        ("Ortolano\nsimulation", C_ORT_SIM, "dv_ort_sim"),
        (f"RL simulation\nto {TARGET_RANGE:g} m", C_RL, "dv_rl_target"),
        (f"RL simulation\nto {TARGET_RANGE:g} m\n+ residual velocity", C_RL_STOP, "dv_rl_target_plus_stop"),
    ]
    rng = np.random.default_rng(0)  # jitter only
    fig, axes = plt.subplots(1, len(MC_LEVELS), figsize=(6 * len(MC_LEVELS), 4.5), sharey=True, squeeze=False)
    for ax, pr in zip(axes[0], MC_LEVELS):
        sub = table[(table["pr"] == pr) & (table["run"] > 0)]
        for j, (_, color, col) in enumerate(groups):
            v = sub[col].dropna().to_numpy(dtype=float)
            ax.scatter(j + rng.uniform(-0.12, 0.12, v.size), v, s=18, color=color, alpha=0.8, zorder=3)
            if v.size:
                mean, std = v.mean(), (v.std(ddof=1) if v.size > 1 else 0.0)
                ax.hlines(mean, j - 0.25, j + 0.25, color=C_INK, lw=1.2, zorder=4)
                # +-1 std whisker, offset right of the points; lower arm clipped to stay positive on log axis
                ax.errorbar(j + 0.2, mean, yerr=[[min(std, 0.99 * mean)], [std]], fmt="none",
                            ecolor=C_INK, elinewidth=1.0, capsize=3, zorder=4)
                ax.annotate(f"{mean:.3f}\n± {std:.3f}", (j + 0.27, mean), va="center", fontsize=7, color=C_INK)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_yscale("log")
        ax.set_title(f"p_r = {pr:g} %", color=C_INK)
        _style(ax)
    axes[0][0].set_ylabel("$\\Delta v$ [m/s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "dv_comparison_mc.png"), dpi=200)
    plt.close(fig)


# ============================================================================
# MAIN DRIVER
# ============================================================================
def generate_ics(n, n_mc, ic_seed):
    """Nominal start + n_mc dispersed starts per p_r level (Ortolano Eqs. 15.1-15.2)."""
    rng = np.random.default_rng(ic_seed)
    tasks = [((0.0, 0), np.concatenate([R0_REF_H, V0_REF_H]))]
    for pr in MC_LEVELS:
        sigma_r = pr / 100.0 * np.linalg.norm(R0_REF_H)
        sigma_v = n * sigma_r
        for run in range(1, n_mc + 1):
            e_r = sigma_r * rng.standard_normal(3)
            e_v = sigma_v * rng.standard_normal(3)
            tasks.append(((pr, run), np.concatenate([R0_REF_H + e_r, V0_REF_H + e_v])))
    return tasks


def run_comparison(model_path, output_folder, n_mc=10, num_workers=4, ic_seed=2018,
                   scenario_seed=12345, time_limit=SIM_TIME):
    oe = make_chief_oe(**CHIEF_OE)
    n = mean_motion(oe)
    period = 2 * np.pi / n
    chief_state = build_chief_state(oe, port="vbar")

    c = CHIEF_OE
    print(f"--- Ortolano 2018 Sc. 1 Case 3 Ex. 3 MC: RL policy, v-bar 100 m -> {TARGET_RANGE:g} m ---")
    print(f"  chief: a = {c['a'] / 1e3:.5f} km, e = {c['e']}, i = {np.degrees(c['i']):.2f} deg, "
          f"RAAN = {np.degrees(c['Omega']):.2f} deg, argp = {np.degrees(c['omega']):.2f} deg, "
          f"nu = {np.degrees(c['f']):.2f} deg")
    print(f"  n = {n:.8e} rad/s (Ortolano 1.06091165e-3), period = {period / 60:.2f} min, "
          f"Ortolano t_f = {TF_REV:g} rev ({TF_REV * period:.0f} s)")
    print(f"  p_r levels = {MC_LEVELS} %, {n_mc} MC runs each (+ nominal), ic_seed = {ic_seed}")
    print(f"  RL time limit = {time_limit:.0f} s, stop at {TARGET_RANGE:g} m: {STOP_AT_TARGET}")

    tasks = generate_ics(n, n_mc, ic_seed)
    workers = max(1, min(num_workers, len(tasks)))
    chunks = [tasks[i::workers] for i in range(workers)]
    args = [(ch, model_path, chief_state, scenario_seed, time_limit) for ch in chunks]

    t0 = time.time()
    if workers == 1:
        results = [_run_chunk(args[0])]
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(_run_chunk, args)
    print(f"RL rollouts done in {time.time() - t0:.0f} s")

    runs = {key: df for chunk in results for key, df in chunk}
    runs = {key: runs[key] for key, _ in tasks}

    rows = []
    for (pr, run), ic in tasks:
        df = runs[(pr, run)]
        sigma_r = pr / 100.0 * np.linalg.norm(R0_REF_H)
        dv_opt, dv_sim = _ortolano_dv(pr, run)
        rows.append({"pr": pr, "run": run, "sigma_r": sigma_r, "sigma_v": n * sigma_r,
                     **{f"ic_{k}": v for k, v in zip(["x", "y", "z", "vx", "vy", "vz"], ic)},
                     "dv_ort_opt": dv_opt, "dv_ort_sim": dv_sim,
                     **analyze_run(df)})
        df.assign(pr=pr, run=run).to_csv(
            os.path.join(output_folder, f"rl_run_pr{pr:g}_run{run:02d}.csv"), index=False)

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(output_folder, "comparison_mc.csv"), index=False)
    print_comparison(table)

    plot_trajectories(runs, output_folder)
    plot_trajectories_3d(runs, output_folder)
    plot_control(runs, output_folder)
    plot_dv_summary(table, output_folder)
    print(f"\nResults written to {output_folder}")
    return table


if __name__ == "__main__":
    # --------------------------- Configuration ---------------------------
    model_path = r"models\training_run_2026-10-07_17-57-01\rpo_min_dv_spec.zip"
    n_mc = 10              # MC runs per p_r level (Ortolano: 10)
    num_workers = 4
    ic_seed = 2018         # dispersion draws
    scenario_seed = 12345  # env reset seed
    # -------------------------------------------------------------------

    output_folder = os.path.join("results", "ortolano_comparison", time.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(output_folder, exist_ok=True)

    run_comparison(model_path, output_folder, n_mc=n_mc, num_workers=num_workers, ic_seed=ic_seed,
                   scenario_seed=scenario_seed)
