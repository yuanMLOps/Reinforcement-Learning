import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyGridWorld(gym.Env):
    
    metadata = {"render_modes": ["rgb_array", "human"],
                "render_fps": 4,
                }

    def __init__(self, render_mode:str="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = 3
        self.n_states = self.grid_size ** 2
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)  # 0=up, 1=right, 2=down, 3=left

        self.state = 0  # Start at grid 0
        self.target = 8
        self.mountains = {4, 7}
        self.P = {}

        for s in range(self.n_states):
            self.P[s] = {}
            for a in range(4):
                next_s, reward, terminated, _, _ = self.traverse(s, a)
                self.P[s][a] = [(1.0, next_s, reward, terminated)] 

    def reset(self, seed=None, options=None):
        self.state = 0
        return self.state, {}

    def traverse(self, state, action):
        row, col = divmod(state, self.grid_size)
        next_row, next_col = row, col

        if action == 3 and row > 0: next_row -= 1  # up
        elif action == 2 and col < self.grid_size - 1: next_col += 1  # right
        elif action == 1 and row < self.grid_size - 1: next_row += 1  # down
        elif action == 0 and col > 0: next_col -= 1  # left

        next_state = next_row * self.grid_size + next_col
        reward = self._get_reward(state, next_state)
        terminated = next_state == self.target

        return next_state, reward, terminated, False, {}

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.traverse(self.state, action)
        self.state = next_state

        return next_state, reward, terminated, truncated, info
    

    
    def render(self):
        if self.render_mode != "rgb_array":
            return  # Only support rgb_array for now
    
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
    
        # Color definitions
        agent_color = [0, 0, 255]       # Blue
        target_color = [0, 255, 0]      # Green
        # mountain_color = [139, 69, 19]  # Brown
        mountain_color = [255, 0, 0]
        empty_color = [255, 255, 255]   # White
    
        # Fill grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if idx == self.state:
                    grid[i, j] = agent_color
                elif idx == self.target:
                    grid[i, j] = target_color
                elif idx in self.mountains:
                    grid[i, j] = mountain_color
                else:
                    grid[i, j] = empty_color
    
        return grid

    def _get_reward(self, current, next_):
        if next_ == self.target and current in {5, 7}:
            return 10
        elif next_ in self.mountains:
            return -2
        elif current in self.mountains:
            return -1
        else:
            return -1


class MyGridWorldMC(gym.Env):
    
    metadata = {"render_modes": ["rgb_array", "human"],
                "render_fps": 4,
                }

    def __init__(self, render_mode:str="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.grid_size = 3
        self.n_states = self.grid_size * 2
        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(4)  # 0=up, 1=right, 2=down, 3=left

        self.state = 0  # Start at grid 0
        self.target = 5
        self.mountains = {4}
        
       
    def reset(self, seed=None, options=None):
        self.state = 0
        return self.state, {}

    def traverse(self, state, action):
        row, col = divmod(state, self.grid_size)
        next_row, next_col = row, col

        if action == 3 and row > 0: next_row -= 1  # up
        elif action == 2 and col < self.grid_size - 1: next_col += 1  # right
        elif action == 1 and row < self.grid_size - 2: next_row += 1  # down
        elif action == 0 and col > 0: next_col -= 1  # left

        next_state = next_row * self.grid_size + next_col
        reward = int(self._get_reward(state, next_state))
        terminated = next_state == self.target

        return next_state, reward, terminated, False, {}

    def step(self, action):
        next_state, reward, terminated, truncated, info = self.traverse(self.state, action)
        self.state = next_state

        return next_state, reward, terminated, truncated, info   

    
    def render(self):
        if self.render_mode != "rgb_array":
            return  # Only support rgb_array for now
    
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
    
        # Color definitions
        agent_color = [0, 0, 255]       # Blue
        target_color = [0, 255, 0]      # Green
        # mountain_color = [139, 69, 19]  # Brown
        mountain_color = [255, 0, 0]
        empty_color = [255, 255, 255]   # White
    
        # Fill grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                idx = i * self.grid_size + j
                if idx == self.state:
                    grid[i, j] = agent_color
                elif idx == self.target:
                    grid[i, j] = target_color
                elif idx in self.mountains:
                    grid[i, j] = mountain_color
                else:
                    grid[i, j] = empty_color
    
        return grid

    def _get_reward(self, current, next_) -> int:
        if next_ == self.target and current in {4, 2}:
            return 10
        elif next_ in self.mountains:
            return -2
        elif current in self.mountains:
            return -1
        else:
            return -1
