# Source - https://stackoverflow.com/a
# Posted by ttronas
# Retrieved 2025-11-17, License - CC BY-SA 4.0

import os, sys

from random import randint
from numpy import inf, float32, array, int32, int64
import gym
from gym.wrappers import FlattenObservation
from stable_baselines3 import A2C, DQN, PPO
import numpy as np

"""Roulette environment class"""
class Roulette_Environment(gym.Env):

    metadata = {'render.modes': ['human', 'text']}

    """Initialize the environment"""
    def __init__(self):
        super(Roulette_Environment, self).__init__()

        # Some global variables
        self.max_table_limit = 1000
        self.initial_bankroll = 2000

        # Spaces
        # Each number on roulette board can have 0-1000 units placed on it
        self.action_space = gym.spaces.Box(low=0, high=1000, shape=(37,))

        # We're going to keep track of how many times each number shows up
        # while we're playing, plus our current bankroll and the max
        # table betting limit so the agent knows how much $ in total is allowed
        # to be placed on the table. Going to use a Dict space for this.
        self.observation_space = gym.spaces.Dict(
            {
                "0": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "1": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "2": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "3": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "4": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "5": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "6": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "7": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "8": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "9": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "10": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "11": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "12": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "13": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "14": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "15": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "16": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "17": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "18": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "19": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "20": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "21": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "22": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "23": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "24": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "25": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "26": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "27": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "28": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "29": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "30": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "31": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "32": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "33": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "34": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "35": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                "36": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
                
                "current_bankroll": gym.spaces.Box(low=-inf, high=inf, shape=(1,), dtype=int),
                
                "max_table_limit": gym.spaces.Box(low=0, high=inf, shape=(1,), dtype=int),
            }
        )

    """Reset the Environment"""
    def reset(self):
        self.current_bankroll = self.initial_bankroll
        self.done = False

        # Take a sample from the observation_space to modify the values of
        self.current_state = self.observation_space.sample()
        
        # Reset each number being tracked throughout gameplay to 0
        for i in range(0, 37):
            self.current_state[str(i)] = np.array([0], dtype=int)

        # Reset our globals
        self.current_state['current_bankroll'] = np.array([self.current_bankroll], dtype=int)
        self.current_state['max_table_limit'] = np.array([self.max_table_limit], dtype=int)
        
        return self.current_state


    """Step Through the Environment"""
    def step(self, action):
        
        # Convert actions to ints cuz they show up as floats,
        # even when defined as ints in the environment.
        # https://github.com/openai/gym/issues/3107
        for i in range(len(action)):
            action[i] = int(action[i])
        self.current_action = action
        
        # Subtract your bets from bankroll
        sum_of_bets = sum([bet for bet in self.current_action])

        # Spin the wheel
        self.current_number = randint(a=0, b=36)

        # Calculate payout/reward
        self.reward = 36 * self.current_action[self.current_number] - sum_of_bets

        self.current_bankroll += self.reward

        # Update the current state
        self.current_state['current_bankroll'] = np.array([self.current_bankroll], dtype=int)
        self.current_state[str(self.current_number)] += np.array([1], dtype=int)

        # If we've doubled our money, or lost our money
        if self.current_bankroll >= self.initial_bankroll * 2 or self.current_bankroll <= 0:
            self.done = True

        return self.current_state, self.reward, self.done, {}


    """Render the Environment"""
    def render(self, mode='text'):
        # Text rendering
        if mode == "text":
            print(f'Bets Placed: {self.current_action}')
            print(f'Number rolled: {self.current_number}')
            print(f'Reward: {self.reward}')
            print(f'New Bankroll: {self.current_bankroll}')

env = Roulette_Environment()

model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=10)

obs = env.reset()
# obs = FlattenObservation(obs)

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    # action, _state = model.predict(FlattenObservation(obs), deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
