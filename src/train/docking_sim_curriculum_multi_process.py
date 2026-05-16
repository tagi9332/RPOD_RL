"""
Multi-process PPO training with 3-stage reverse curriculum.

Curriculum stages
-----------------
Stage 0 — Terminal approach (20-24 m from docking port, inside capture sphere).
           Inspector starts in Phase 1 by step 2.  Trains the final ingress
           maneuver with dense reward and short episodes.

Stage 1 — Correct-side capture (65-615 m from docking port, +boresight side).
           Inspector starts in Phase 0.  Trains braking and waypoint capture,
           then chains into the Stage 0 behaviour.

Stage 2 — Full mission (1800-2000 m, any direction).  Trains the complete
           navigate-capture-dock chain and generalises to wrong-side recovery.

Advance thresholds
------------------
Stage 0 → 1:  70 % conjunction rate over 50 episodes.
Stage 1 → 2:  50 % conjunction rate over 100 episodes.

TensorBoard:  tensorboard --logdir="./logs/"
"""

from datetime import datetime
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback,
)

from Basilisk.architecture import bskLogging
from bsk_rl import scene, ConstellationTasking

from utils.misc.time_est import TimeRemainingCallback
from src.curriculum import ConjunctionRadiusScheduler, CorridorAngleScheduler, AttitudeErrorScheduler
from src.curriculum.curriculum_stage_callback import CurriculumStageCallback

from src.train import (
    rso_sat_args,
    inspector_sat_args,
    RSOSat,
    InspectorSat,
    Sb3BksEnv,
)
from src.randomizers.curriculum_sat_arg_randomizer import CurriculumSatArgRandomizer
from src.randomizers.sat_arg_randomizer_rso_random_inertial import make_sat_arg_randomizer
from src.rewarders import get_rewarders

from resources import (
    learning_rate,
    entropy_coeff,
    max_grad_norm,
    SIM_TIME,
    SIM_DT,
)


class CurriculumSb3BksEnv(Sb3BksEnv):
    """
    Extends Sb3BksEnv with a ``curriculum_stage`` attribute so that
    CurriculumStageCallback can advance the stage via set_attr.
    On each reset the stage is forwarded to the CurriculumSatArgRandomizer
    before ConstellationTasking.reset() calls the randomizer.
    """

    def __init__(self, env, randomizer=None):
        super().__init__(env, randomizer=randomizer)
        self.curriculum_stage: int = 0

    def reset(self, **kwargs):
        if self.randomizer is not None:
            self.randomizer.stage = self.curriculum_stage
        return super().reset(**kwargs)


def make_env(rank: int, seed: int = 0, initial_stage: int = 0):
    def _init():
        rso = RSOSat("RSO", sat_args=rso_sat_args)
        inspector = InspectorSat("Inspector", sat_args=inspector_sat_args)
        bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

        scenario = scene.SphericalRSO(
            n_points=100, radius=1.0, theta_max=np.radians(30),
            range_max=250, theta_solar_max=np.radians(60),
        )
        rewarders = get_rewarders()
        randomizer = CurriculumSatArgRandomizer(
            stage=initial_stage,
            mode="train",
            rso_att_type="near_velocity",
        )
        env = ConstellationTasking(
            satellites=[rso, inspector],
            sat_arg_randomizer=randomizer,
            scenario=scenario,
            rewarder=rewarders,
            time_limit=SIM_TIME,
            sim_rate=SIM_DT,
            log_level="ERROR",
        )
        env_sb3 = CurriculumSb3BksEnv(env, randomizer=randomizer)
        env_sb3 = Monitor(env_sb3)
        env_sb3 = FlattenObservation(env_sb3)
        env_sb3.reset(seed=seed + rank)
        return env_sb3

    set_random_seed(seed)
    return _init


def make_eval_env(seed: int = 42):
    """Full-range eval env (Stage 2 distribution) for stable checkpoint comparison."""
    def _init():
        rso = RSOSat("RSO", sat_args=rso_sat_args)
        inspector = InspectorSat("Inspector", sat_args=inspector_sat_args)
        bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

        scenario = scene.SphericalRSO(
            n_points=100, radius=1.0, theta_max=np.radians(30),
            range_max=250, theta_solar_max=np.radians(60),
        )
        rewarders = get_rewarders()
        randomizer = make_sat_arg_randomizer(mode="test", rso_att_type="near_velocity")
        env = ConstellationTasking(
            satellites=[rso, inspector],
            sat_arg_randomizer=randomizer,
            scenario=scenario,
            rewarder=rewarders,
            time_limit=SIM_TIME,
            sim_rate=SIM_DT,
            log_level="ERROR",
        )
        env_sb3 = Sb3BksEnv(env, randomizer=randomizer)
        env_sb3 = Monitor(env_sb3)
        env_sb3 = FlattenObservation(env_sb3)
        env_sb3.reset(seed=seed)
        return env_sb3

    set_random_seed(seed)
    return _init


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    run_name  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir   = f"./logs/curriculum_run_{run_name}/"
    model_dir = f"./models/curriculum_run_{run_name}/"
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Config
    # -------------------------------------------------------------------------
    NUM_CPU         = 14
    N_STEPS_PER_ENV = 512
    TOTAL_TIMESTEPS = 10_000_000

    # Set > 0 to skip earlier stages (e.g. resume from a Stage-1 checkpoint).
    INITIAL_STAGE = 0   # 0 = terminal, 1 = correct-side, 2 = full mission

    # Optional parameter schedulers (set to None to disable).
    CONJ_RADIUS_SCHEDULE:    tuple | None = None   # e.g. (200.0, 10.0) → m
    CORRIDOR_ANGLE_SCHEDULE: tuple | None = None   # e.g. (360.0, 30.0) → °
    ATTITUDE_ERROR_SCHEDULE: tuple | None = None   # e.g. (45.0,   5.0) → °
    # -------------------------------------------------------------------------

    train_env = SubprocVecEnv(
        [make_env(i, seed=0, initial_stage=INITIAL_STAGE) for i in range(NUM_CPU)]
    )
    eval_env = DummyVecEnv([make_eval_env(seed=42)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device="cpu",
        n_steps=N_STEPS_PER_ENV,
        batch_size=1024,
        learning_rate=learning_rate,
        ent_coef=entropy_coeff,
        max_grad_norm=max_grad_norm,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    custom_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    model.set_logger(custom_logger)

    eval_freq = max(50_000 // NUM_CPU, 1)

    active_callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=model_dir,
            log_path=log_dir,
            eval_freq=eval_freq,
            deterministic=True,
            n_eval_episodes=10,
            render=False,
        ),
        TimeRemainingCallback(total_steps=int(TOTAL_TIMESTEPS)),
        CheckpointCallback(
            save_freq=eval_freq,
            save_path=model_dir,
            name_prefix="ppo_curriculum_checkpoint",
        ),
        CurriculumStageCallback(
            window=100,
            initial_stage=INITIAL_STAGE,
            verbose=1,
        ),
    ]

    if CONJ_RADIUS_SCHEDULE is not None:
        i, f = CONJ_RADIUS_SCHEDULE
        active_callbacks.append(ConjunctionRadiusScheduler(i, f))
        print(f"ConjunctionRadiusScheduler: {i} → {f} m")

    if CORRIDOR_ANGLE_SCHEDULE is not None:
        i, f = CORRIDOR_ANGLE_SCHEDULE
        active_callbacks.append(CorridorAngleScheduler(i, f))
        print(f"CorridorAngleScheduler: {i} → {f} °")

    if ATTITUDE_ERROR_SCHEDULE is not None:
        i, f = ATTITUDE_ERROR_SCHEDULE
        active_callbacks.append(AttitudeErrorScheduler(i, f))
        print(f"AttitudeErrorScheduler: {i} → {f} °")

    print(
        f"Curriculum training | {NUM_CPU} cores | "
        f"Stage {INITIAL_STAGE} | {TOTAL_TIMESTEPS:,} steps"
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList(active_callbacks),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
    finally:
        save_path = os.path.join(model_dir, "rpo_curriculum.zip")
        model.save(save_path)
        print(f"Model saved to {save_path}")
