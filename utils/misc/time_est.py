import time
from stable_baselines3.common.callbacks import BaseCallback

class TimeRemainingCallback(BaseCallback):
    """
    Calculates and prints the estimated time remaining every N steps.
    """
    def __init__(self, total_steps, verbose=0):
        super(TimeRemainingCallback, self).__init__(verbose)
        self.total_steps = total_steps
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Update every 10,000 steps
        if self.n_calls % 50000 == 0 and self.n_calls > 0:
            current_time = time.time()
            elapsed_time = current_time - self.start_time # type: ignore
            fps = self.n_calls / elapsed_time
            
            remaining_steps = self.total_steps - self.n_calls
            remaining_time_sec = remaining_steps / fps
            
            # Convert to minutes and seconds
            mins, secs = divmod(int(remaining_time_sec), 60)
            hours, mins = divmod(mins, 60)
            
            print(f"\n>>> Progress: {self.n_calls}/{self.total_steps} steps")
            print(f">>> Current Speed: {fps:.2f} FPS")
            print(f">>> Estimated Time Remaining: {hours:02d}h:{mins:02d}m:{secs:02d}s")
            
        return True