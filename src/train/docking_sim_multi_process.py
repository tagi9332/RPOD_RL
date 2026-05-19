"""
Multi-process training script for the RPO docking simulation using Stable Baselines3's PPO algorithm.

For monitoring via TensorBoard, run the following command in the terminal:
tensorboard --logdir="./logs/"

"""

# Standard libraries
from datetime import datetime
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

# RL libraries
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# Import callbacks
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from utils.misc.time_est import TimeRemainingCallback


# Basilisk & BSK-RL imports
from Basilisk.architecture import bskLogging
from bsk_rl import scene, data, ConstellationTasking
from bsk_rl.sim import fsw

# Import curriculum schedulers
from src.curriculum import ConjunctionRadiusScheduler, CorridorAngleScheduler, AttitudeErrorScheduler, DeltaVPenaltyScheduler

# Base training script imports
from src.train import (
    rso_sat_args,
    inspector_sat_args,
    RSOSat,
    InspectorSat,
    Sb3BksEnv,
) 

# Import randomizer
from src.randomizers.sat_arg_randomizer_rso_random_inertial import make_sat_arg_randomizer

# Import rewarders
from src.rewarders import get_rewarders

# Import weight scheduler
from src.rewarders.weight_scheduler import CurriculumPenalty

# Import weights
from resources import (
    learning_rate,
    entropy_coeff,
    max_grad_norm,
)

# Import sim parameters
from resources import (
    SIM_TIME,
    SIM_DT,
)

def make_env(rank: int, seed: int = 0):
    """
    Utility function for multiprocessed training env.
    Instantiates fresh Basilisk objects for each CPU core.
    """
    def _init():
        rso = RSOSat("RSO", sat_args=rso_sat_args)
        inspector = InspectorSat("Inspector", sat_args=inspector_sat_args)

        bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

        scenario = scene.SphericalRSO(
            n_points=100, radius=1.0, theta_max=np.radians(30),
            range_max=250, theta_solar_max=np.radians(60)
        )
        rewarders = get_rewarders()
        randomizer = make_sat_arg_randomizer(mode="train", rso_att_type="random")

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

        env_sb3.reset(seed=seed + rank)
        return env_sb3

    set_random_seed(seed)
    return _init


def make_eval_env(seed: int = 42):
    """
    Eval env factory. Uses mode='test' so the RSO orbit persists across eval
    episodes, giving a stable scenario for comparing checkpoints. The inspector
    starting position is still re-randomized each episode so the policy is
    evaluated across varied approach geometries.
    """
    def _init():
        rso = RSOSat("RSO", sat_args=rso_sat_args)
        inspector = InspectorSat("Inspector", sat_args=inspector_sat_args)

        bskLogging.setDefaultLogLevel(bskLogging.BSK_ERROR)

        scenario = scene.SphericalRSO(
            n_points=100, radius=1.0, theta_max=np.radians(30),
            range_max=250, theta_solar_max=np.radians(60)
        )
        rewarders = get_rewarders()
        randomizer = make_sat_arg_randomizer(mode="test", rso_att_type="random")

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
    # Setup directories
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"./logs/training_run_{run_name}/"
    model_path = f"./models/training_run_{run_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Detect number of CPU cores for parallel environments (leave 2 cores free to avoid system overload)
    cpu = os.cpu_count()
    num_cpu = max(cpu - 4, 1) if cpu is not None else 1
    print(f"Detected {cpu} CPU cores. Setting up {num_cpu} parallel environments for training.")

    # Config
    n_steps_per_env = 512
    total_timesteps = 1_000_000 
    
    # Create multi-core training env
    env = SubprocVecEnv([make_env(i, seed=0) for i in range(num_cpu)])

    # Create evaluation env (fixed RSO orbit via mode='test', diverse inspector starts)
    eval_env = DummyVecEnv([make_eval_env(seed=42)])    

    # ------------------------- Model Initialization -------------------------
    # Initialize model
    LOAD_MODEL = True  # Set to False to train from scratch, True to load existing model
    LOAD_PATH = r"models\training_run_2026-05-18_07-07-29\rpo_min_dv_spec.zip"
    # -------------------------------------------------------------------------

    # Optional hyperparameter overrides when loading a model (set to None to keep saved values)
    OVERRIDE_LEARNING_RATE: float | None = 5e-5  # e.g. 5e-5
    OVERRIDE_ENT_COEF:      float | None = 5e-4  # e.g. 1e-4

    if LOAD_MODEL and os.path.exists(LOAD_PATH):
        print(f"Loading existing model from {LOAD_PATH}...")
        custom_objects = {}
        if OVERRIDE_LEARNING_RATE is not None:
            custom_objects["learning_rate"] = OVERRIDE_LEARNING_RATE
            print(f"  Overriding learning_rate → {OVERRIDE_LEARNING_RATE}")
        if OVERRIDE_ENT_COEF is not None:
            custom_objects["ent_coef"] = OVERRIDE_ENT_COEF
            print(f"  Overriding ent_coef → {OVERRIDE_ENT_COEF}")
        model = PPO.load(
            LOAD_PATH,
            env=env,
            device="cpu",
            custom_objects=custom_objects if custom_objects else None,
        )
    else:
        print("Creating a fresh model from scratch...")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            device="cpu",
            n_steps=n_steps_per_env,
            batch_size=1024,  
            learning_rate=learning_rate,
            ent_coef=entropy_coeff,
            max_grad_norm=max_grad_norm,
        )


    # --- CALLBACKS CONFIGURATION --
    # 1. Standard callbacks
    custom_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    time_callback = TimeRemainingCallback(total_steps=int(total_timesteps))
    
    eval_freq = max(50000 // num_cpu, 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        n_eval_episodes=10,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq, 
        save_path=model_path, 
        name_prefix="ppo_inspector_multicore_checkpoint"
    )

    # 2. Curriculum scheduler config — set to None to disable a scheduler
    # -------------------------------------------------------------------------
    CONJ_RADIUS_SCHEDULE:    tuple[float, float] | None = None   # e.g. (200, 10)   → m
    CORRIDOR_ANGLE_SCHEDULE: tuple[float, float] | None = None   # e.g. (360, 30)   → °
    ATTITUDE_ERROR_SCHEDULE: tuple[float, float] | None = None # e.g. (90, 5)     → °
    DV_PENALTY_SCHEDULE:     tuple[float, float] | None = (0.1, 1.0)   # e.g. (0.0, 0.5)  → weight
    # -------------------------------------------------------------------------

    active_callbacks = [eval_callback, time_callback, checkpoint_callback]

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

    if DV_PENALTY_SCHEDULE is not None:
        i, f = DV_PENALTY_SCHEDULE
        active_callbacks.append(DeltaVPenaltyScheduler(i, f))
        print(f"DeltaVPenaltyScheduler: {i} → {f}")

    callbacks = CallbackList(active_callbacks)

    model.set_logger(custom_logger)
    
    print(f"Starting training on {num_cpu} cores...")
  
    try:
        # Run Training
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callbacks, 
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model and logs...")
    
    finally:
        # Final Save
        model_save_path = os.path.join(model_path, "rpo_min_dv_spec.zip")
        model.save(model_save_path)
        print(f"Training Complete. Model saved as {model_save_path}")