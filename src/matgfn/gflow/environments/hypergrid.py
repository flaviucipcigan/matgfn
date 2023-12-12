# Farama Gymnasium
import gymnasium as gym

# Data science
import numpy as np
import matplotlib.pyplot as plt

class HypergridEnvironment(gym.Env):
    def __init__(self, dimensions, grid_size, R0=0.1, R1=0.5, R2=2):
        super().__init__()
        self.dimensions = dimensions
        self.grid_size = grid_size
        # Each dimension in [0, grid_size-1] (both ends included)
        self.observation_space = gym.spaces.Box(low = np.array([0] * dimensions), high = np.array([grid_size - 1] * dimensions), dtype=int)
        # Increment each dimension plus termination action
        self.action_space = gym.spaces.Discrete(dimensions + 1)

        # Rewards
        self.R0 = R0
        self.R1 = R1
        self.R2 = R2

    def _get_obs(self):
        return np.copy(self._x)
    
    def _get_info(self):
        # Compute the allowed forward and backward actions
        
        # If we are at the starting position, all foward actions are legal
        # No backward actions are legal
        allowed_forward_actions = []
        allowed_backward_actions = []

        if self._terminated:
            # When terminated, only backward action is un-terminated
            allowed_backward_actions.append(self.action_space.n - 1)
        else: 
            # Actions to increment dimension i
            for i in range(0, len(self._x)):
                # We can go backwards in dimension i if we are not at the lowest edge
                if self._x[i] > self.observation_space.low[i]:
                    allowed_backward_actions.append(i)

                # We can go forwards in dimension i if we are not at the highest edge
                if self._x[i] < self.observation_space.high[i]:
                    allowed_forward_actions.append(i)
            # Termination action
            allowed_forward_actions.append(self.action_space.n - 1)
            
        return {"allowed_forward_actions": allowed_forward_actions,
                "allowed_backward_actions": allowed_backward_actions}

    def _get_reward(self):
        if not self._terminated:
            return None

        else: 
            assert self._x in self.observation_space
            add_R1 = True
            add_R2 = True

            for dim_index in range(0, len(self._x)):
                shifted_coord = abs(float(self._x[dim_index]) / float(self.observation_space.high[dim_index]) - 0.5)
                if not(0.25 <= shifted_coord):
                    add_R1 = False
                if not((0.3 <= shifted_coord) and (shifted_coord <= 0.4)):
                    add_R2 = False

            reward = self.R0

            if add_R1:
                reward = reward + self.R1

            if add_R2:
                reward = reward + self.R2

            return reward

    def reset(self, seed=None):
        self._x = np.copy(self.observation_space.low)
        self._terminated = False

        return self._get_obs(), self._get_info()

    def step(self, action):
        assert action in self.action_space

        if action == self.action_space.n - 1:
            assert not self._terminated # We only terminate once
            self._terminated = True
        else:
            self._x[action] = self._x[action] + 1
            assert self._x in self.observation_space # Actions should not take agent off grid

        return self._get_obs(), self._get_reward(), self._terminated, False, self._get_info()

    def _compute_all_rewards(self):
        pass

    def render(self):
        rewards = np.zeros(self.observation_space.high + 1)

        # In 2D, visualise the environment
        if self.action_space.n == 3:
            x_pos = int(self._x[0])
            y_pos = int(self._x[1])
            rewards[x_pos, y_pos] = 1

            plt.matshow(rewards)
        else:
            return np.copy(self._x)
