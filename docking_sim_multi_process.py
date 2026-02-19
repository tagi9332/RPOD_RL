# Standard Libraries
import os
import numpy as np

# RL Libraries
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

# Import Callbacks
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback
from utils.misc.time_est import TimeRemainingCallback


# Basilisk & BSK-RL components needed for the factory function
from Basilisk.architecture import bskLogging
from bsk_rl import scene, data, ConstellationTasking
from bsk_rl.sim import fsw

# --- IMPORT EVERYTHING FROM YOUR SINGLE-CORE SCRIPT ---
from docking_sim_training import (
    rso_sat_args,
    inspector_sat_args,
    RSOSat,
    InspectorSat,
    SB3CompatibleEnv,
    sat_arg_randomizer,
    SimulationLoggerCallback
)

# Import weights
from weights import reward_weight,alpha,weight


class CurriculumPenalty:
    """Slowly increases the penalty weight over a set number of episodes."""
    
    def __init__(self, start_weight: float = 0.01, end_weight: float = 1.0, max_episodes: int = 1000):
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.max_episodes = max_episodes
        self.current_episode = 0

    def __call__(self) -> float:
        # 1. Calculate how far along we are (from 0.0 to 1.0)
        progress = min(1.0, self.current_episode / self.max_episodes)
        
        # 2. Linearly interpolate between the start and end weights
        current_weight = self.start_weight + progress * (self.end_weight - self.start_weight)
        
        # 3. Increment the episode counter for the next time the env resets
        self.current_episode += 1
        
        return current_weight


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

        rewarders = (
            data.ResourceReward(
                resource_fn=lambda sat: sat.fsw.dv_available if isinstance(sat.fsw, fsw.MagicOrbitalManeuverFSWModel) else 0.0,
                
                # Curriculum Penalty: starts small to allow early exploration, then ramps up to the full penalty
                reward_weight=CurriculumPenalty(start_weight=0.01, end_weight=1.0, max_episodes=total_timesteps // num_cpu // n_steps_per_env)
            ),
            data.RelativeRangeLogReward(alpha=alpha, delta_x_max=np.array([1000.0, 1000.0, 1000.0, 20.0, 20.0, 20.0])),

            data.RelativeCosineReward(weight=weight)
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

        env_sb3 = SB3CompatibleEnv(env)
        env_sb3 = Monitor(env_sb3)
        env_sb3 = FlattenObservation(env_sb3)
        
        env_sb3.reset(seed=seed + rank)
        return env_sb3
        
    set_random_seed(seed)
    return _init


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Setup Directories
    log_path = "./ppo_logs/"
    model_path = "./models/best_model/"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    new_logger = configure(log_path, ["stdout", "csv"])

    # 2. Config
    num_cpu = 14
    n_steps_per_env = 512
    total_timesteps = 300_000 
    
    # 3. Create Training Env (Multi-core)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    
    # NEW: Create Evaluation Env (Single-core is best for testing)
    eval_env = DummyVecEnv([make_env(rank=99, seed=42)])
    
    # 4. Initialize Model
    LOAD_MODEL = True  # Set to False if you want to train from scratch again
    LOAD_PATH = "ppo_inspector_final" # Exclude the .zip extension
    
    if LOAD_MODEL and os.path.exists(LOAD_PATH + ".zip"):
        print(f"Loading existing model from {LOAD_PATH}...")
        # Load the model and bind it to your new multi-core environment
        model = PPO.load(
            LOAD_PATH, 
            env=env, 
            device="cpu",
            # Note: hyperparameters like learning_rate are loaded from the zip.
            # If you want to overwrite them, you can pass custom_objects={"learning_rate": 1e-4}
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
            learning_rate=3e-4,
            ent_coef=0.02,
            max_grad_norm=0.5,
        )
        
    model.set_logger(new_logger)
    
    # --- SETUP CALLBACKS ---
    # sim_logger = SimulationLoggerCallback(save_freq=1000, save_path="training_log_multicore.csv")
    time_callback = TimeRemainingCallback(total_steps=int(total_timesteps))

    # Important Math for eval_freq:
    # SB3 counts eval_freq based on the number of steps PER environment. 
    # To evaluate every ~10,000 global timesteps, we divide 10,000 by 14 cores.
    eval_freq = max(10000 // num_cpu, 1) 

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=eval_freq,
        deterministic=True,      # Tests the actual policy, no random thruster noise
        n_eval_episodes=5,       # Tests 5 different random starting positions
        render=False
    )


    # Save frequency
    checkpoint_freq = max(50000 // num_cpu, 1)  # Save every 50,000 global steps, divided by number of cores

    checkpointcallback = CheckpointCallback(save_freq=checkpoint_freq, save_path=model_path, name_prefix="ppo_inspector_multicore_checkpoint")

    # Combine them into a list
    callbacks = CallbackList([eval_callback, time_callback, checkpointcallback])

    print(f"Starting training on {num_cpu} cores...")
    
    try:
        # 5. Run Training
        model.learn(
            total_timesteps=total_timesteps, 
            callback=callbacks, 
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current model and logs...")
    
    finally:
        # 6. Final Save
        model.save("ppo_inspector_final")
        print("Training Complete. Model and Logs saved.")