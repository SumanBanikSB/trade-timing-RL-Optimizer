import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.max_steps = len(df) - 1
        self.action_space = spaces.Discrete(3)  # 0 = Hold, 1 = Buy, 2 = Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.step_idx = 0
        self.balance = 1000
        self.position = 0
        self.total_profit = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return self.df.iloc[self.step_idx].values.astype(np.float32)

    def step(self, action):
        done = self.step_idx >= self.max_steps
        price = self.df.iloc[self.step_idx]["Close"]
        reward = 0

        if action == 1 and self.position == 0:  # Buy
            self.position = self.balance / price
            self.balance = 0
        elif action == 2 and self.position > 0:  # Sell
            self.balance = self.position * price
            reward = self.balance - 1000
            self.total_profit += reward
            self.position = 0

        self.step_idx += 1
        return self._get_obs(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.step_idx}, Balance: {self.balance:.2f}, Profit: {self.total_profit:.2f}")
