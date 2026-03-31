import time
from stable_baselines3.common.callbacks import BaseCallback

class TimeRemainingCallback(BaseCallback):
    """
    Calculates and prints the estimated time remaining based on TOTAL timesteps.
    """
    def __init__(self, total_steps, verbose=0):
        super(TimeRemainingCallback, self).__init__(verbose)
        self.total_steps = total_steps
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Check progress every 100000 AGGREGATE steps
        # We use num_timesteps because it accounts for all parallel environments
        if self.num_timesteps % 100000 < self.training_env.num_envs:
            if self.num_timesteps > 0:
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                # THE FIX: Use num_timesteps for real FPS
                fps = self.num_timesteps / elapsed_time
                
                remaining_steps = self.total_steps - self.num_timesteps
                remaining_time_sec = remaining_steps / fps if fps > 0 else 0
                
                # Convert to h:m:s
                mins, secs = divmod(int(remaining_time_sec), 60)
                hours, mins = divmod(mins, 60)
                print(f"\n----------------------------------------------------------")
                print(f"\nProgress: {self.num_timesteps}/{self.total_steps} steps")
                print(f"Total Speed: {fps:.2f} FPS (Aggregate across {self.training_env.num_envs} cores)")
                print(f"Estimated Time Remaining: {hours:02d}h:{mins:02d}m:{secs:02d}s")
                print(f"Time of Completion (ETA): {time.ctime(current_time + remaining_time_sec)}\n")
                print(f"----------------------------------------------------------\n")
                
        return True