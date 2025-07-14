import pandas as pd
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from utils.indicators import add_indicators

df = pd.read_csv("data/aapl.csv")
df = add_indicators(df)

env = TradingEnv(df)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("models/trade_ppo")
