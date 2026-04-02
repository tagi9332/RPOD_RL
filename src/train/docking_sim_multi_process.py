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

# Import custom callbacks
from src.conjunction_radius_scheduler import ConjunctionRadiusScheduler

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
from src.weight_scheduler import CurriculumPenalty

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

def make_env(rank: int, seed: int = 0, num_cpu: int = 1, n_steps_per_env: int = 512, total_timesteps: int = 1_000_000):
    """
    Utility function for multiprocessed env.
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
        randomizer = make_sat_arg_randomizer(mode="train", rso_att_type="velocity")

        env = ConstellationTasking(
            satellites=[rso, inspector],
            sat_arg_randomizer=randomizer,
            scenario=scenario,
            rewarder=rewarders,
            time_limit=SIM_TIME,
            sim_rate=SIM_DT,
            log_level="ERROR", 
        )

        env_sb3 = Sb3BksEnv(env)
        env_sb3 = Monitor(env_sb3)
        env_sb3 = FlattenObservation(env_sb3)
        
        env_sb3.reset(seed=seed + rank)
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
    

    # Config
    num_cpu = 14
    n_steps_per_env = 512
    total_timesteps = 2_000_000 
    
    # Create multi-core training env
    env = SubprocVecEnv([make_env(i, seed=0, total_timesteps=total_timesteps, num_cpu=num_cpu, n_steps_per_env=n_steps_per_env) for i in range(num_cpu)])    

    # Creat evaluation env
    eval_env = DummyVecEnv([make_env(rank=99, seed=42, total_timesteps=total_timesteps, num_cpu=1, n_steps_per_env=n_steps_per_env)])    

    # ------------------------- Model Initialization -------------------------
    # Initialize model
    LOAD_MODEL = True  # Set to False to train from scratch, True to load existing model
    LOAD_PATH = r"models\training_run_2026-03-31_20-08-21\rpo_min_dv_spec.zip"
    # -------------------------------------------------------------------------

    if LOAD_MODEL and os.path.exists(LOAD_PATH):
        print(f"Loading existing model from {LOAD_PATH}...")
        # Load the model and bind it to your new multi-core environment
        model = PPO.load(
            LOAD_PATH, 
            env=env, 
            device="cpu",
            # Note: hyperparameters like learning_rate are loaded from the zip.
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
        n_eval_episodes=5,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq, 
        save_path=model_path, 
        name_prefix="ppo_inspector_multicore_checkpoint"
    )

    # 2. Build the callback list dynamically
    active_callbacks = [eval_callback, time_callback, checkpoint_callback]

    # 3. Handle the toggleable Conjunction Radius Scheduler
    USE_CONJ_RADIUS_SCHEDULER = False
    
    if USE_CONJ_RADIUS_SCHEDULER:
        initial_radius = 30.0
        final_radius = 10.0
        conj_radius_scheduler = ConjunctionRadiusScheduler(
            initial_radius=initial_radius, 
            final_radius=final_radius
        )
        active_callbacks.append(conj_radius_scheduler) # Only add if True
        print(f"Using Conjunction Radius Scheduler: {initial_radius} -> {final_radius}")
    else:
        print("Using constant Conjunction Radius (Scheduler disabled).")

    # 4. Finalize the list for the model
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