class CurriculumPenalty:
    """Slowly increases the penalty weight over a set number of episodes."""
    
    def __init__(self, start_weight: float, end_weight: float, max_episodes: int):
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.max_episodes = max_episodes
        self.current_episode = 0

    def __call__(self) -> float:
        # Compute current progress
        progress = min(1.0, self.current_episode / self.max_episodes)
        
        # Linear interpolation between start and end weight
        current_weight = self.start_weight + progress * (self.end_weight - self.start_weight)
        
        # Increment the episode counter for the next time the env resets
        self.current_episode += 1
        
        return current_weight
