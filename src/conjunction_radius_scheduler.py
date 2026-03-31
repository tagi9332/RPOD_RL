from stable_baselines3.common.callbacks import BaseCallback

class ConjunctionRadiusScheduler(BaseCallback):
    """
    Linearly shrinks the conjunction radius as training progresses.
    """
    def __init__(self, initial_radius=490.0, final_radius=30.0, verbose=0):
        super().__init__(verbose)
        self.initial_radius = initial_radius
        self.final_radius = final_radius

    def _on_step(self) -> bool:
        # Calculate training progress (0.0 to 1.0)
        total_steps = self.locals.get("total_timesteps")
        progress = self.num_timesteps / total_steps
        progress = min(1.0, max(0.0, progress)) # Clamp between 0 and 1
        
        # Linearly decay the radius
        current_radius = self.initial_radius - progress * (self.initial_radius - self.final_radius)
        
        # FIX: Use set_attr instead of env_method. 
        # This bypasses the wrapper method block and sets the variable directly!
        self.training_env.set_attr("scheduled_conjunction_radius", current_radius)
        
        # Log to TensorBoard so you can watch it shrink
        self.logger.record("curriculum/conjunction_radius", current_radius)
        
        return True