# Standard libraries
from datetime import datetime
import os
import numpy as np

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

# Base training script imports
from src.train import (
    rso_sat_args,
    inspector_sat_args,
    RSOSat,
    InspectorSat,
    Sb3BksEnv,
    sat_arg_randomizer
) 

# Import custom rewarders
from src.rewarders import (
    RelativeRangeLogReward
)

# Import weight scheduler
from src.weight_scheduler import CurriculumPenalty

# Import weights
from resources import (
    dv_reward_weight,
    rel_range_log_weight,
    learning_rate,
    entropy_coeff,
    max_grad_norm,
)

def make_env(rank: int, seed: int = 0):
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
        max_episodes=total_timesteps // num_cpu // n_steps_per_env
        rewarders = (
            data.ResourceReward(
                resource_fn=lambda sat: sat.fsw.dv_available if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel) else 0.0,
                
                # Curriculum Penalty: starts small to allow early exploration, then ramps up to the full penalty
                reward_weight=CurriculumPenalty(start_weight=dv_reward_weight, end_weight=1.0, max_episodes=max_episodes)
            ),
            RelativeRangeLogReward(alpha=rel_range_log_weight, delta_x_max=np.array([1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0])),

        )

        env = ConstellationTasking(
            satellites=[rso, inspector],
            sat_arg_randomizer=sat_arg_randomizer,
            scenario=scenario,
            rewarder=rewarders,
            time_limit=5000,
            sim_rate=1.0,
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
    
    new_logger = configure(log_dir, ["stdout", "csv"])
    
    # Config
    num_cpu = 14
    n_steps_per_env = 512
    total_timesteps = 500_000 
    
    # Create multi-core training env
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # Creat evaluation env
    eval_env = DummyVecEnv([make_env(rank=99, seed=42)])
    
    # ------------------------- Model Initialization -------------------------
    # Initialize model
    LOAD_MODEL = False  # Set to False to train from scratch, True to load existing model
    LOAD_PATH = "ppo_inspector_crawl" # Exclude the .zip extension
    # -------------------------------------------------------------------------

    if LOAD_MODEL and os.path.exists(LOAD_PATH + ".zip"):
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
        
    model.set_logger(new_logger)
    
    # Callbacks
    time_callback = TimeRemainingCallback(total_steps=int(total_timesteps))

    eval_freq = max(50000 // num_cpu, 1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,      # Tests the actual policy, no random thruster noise
        n_eval_episodes=5,       # Tests 5 different random starting positions
        render=False
    )

    checkpointcallback = CheckpointCallback(save_freq=eval_freq, save_path=model_path, name_prefix="ppo_inspector_multicore_checkpoint")

    # Combine them into a list
    callbacks = CallbackList([eval_callback, time_callback, checkpointcallback])

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
        model.save("ppo_inspector_final")
        print("Training Complete. Model and Logs saved.")