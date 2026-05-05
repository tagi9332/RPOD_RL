import numpy as np
from bsk_rl.act.continuous_actions import ImpulsiveThrustHill

class ImpulsiveThrustHillScaled(ImpulsiveThrustHill):
    """
    Tricks the trained RL model by maintaining the original action space bounds, 
    but scales the physical delta-V output to a new maximum limit.
    """
    
    def __init__(self, chief_name, trained_max_dv, new_max_dv, *args, **kwargs):
        """
        Args:
            chief_name: Chief to use for Hill frame.
            trained_max_dv: The max_dv the model was originally trained on (keeps the model happy).
            new_max_dv: The new physical thrust limit you actually want to use.
        """
        self.trained_max_dv = trained_max_dv
        self.new_max_dv = new_max_dv
        
        # Initialize the parent class using the TRAINED limit. 
        # This defines the spaces.Box so it perfectly matches your saved model.
        super().__init__(chief_name=chief_name, max_dv=trained_max_dv, *args, **kwargs)

    def set_action(self, action: np.ndarray) -> None:
        """Scale the agent's output to the new physical limits."""
        
        # 1. Calculate the scaling ratio
        scale_factor = self.new_max_dv / self.trained_max_dv
        
        # 2. Scale the dV components of the action (indices 0, 1, 2)
        # If the network asks for 100% of the trained thrust, this scales it to 100% of the new thrust.
        action[0:3] = action[0:3] * scale_factor
        
        # 3. Temporarily update the max_dv attribute so the parent class's 
        # internal clamping logic doesn't undo our scaling!
        self.max_dv = self.new_max_dv
        
        # 4. Pass the scaled action up the chain to be executed
        super().set_action(action)
        
        # 5. Restore the original max_dv just in case the RL library checks the space again
        self.max_dv = self.trained_max_dv