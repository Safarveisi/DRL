from typing import Optional, Tuple

import gymnasium as gym
from gymnasium import spaces

import numpy as np

class GridWorldEnv(gym.Env):

    def __init__(self, size: int) -> None:
        self.size = size  # The size of the square grid
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to 'right', 'up', 'left', 'down'
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _get_obs(self) -> np.ndarray:
        # Use the difference between the two arrays to represent the observation
        return self._target_location - self._agent_location
    
    def _get_info(self) -> dict:
        return { 
            'distance': np.linalg.norm(
                self._agent_location - self._target_location, ord=2
            )
        }

    def reset(self, seed: Optional[int] = None) -> Tuple[dict, dict]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[dict, int, bool, bool, dict]:
        # Map the action (element of {0, 1, 2, 3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        # Only if we reach the target
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info