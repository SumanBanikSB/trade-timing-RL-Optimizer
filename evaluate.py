import pandas as pd
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from utils.indicators import add_indicators

df = pd.read_csv("data/aapl.csv")
df = add_indicators(df)

env = TradingEnv(df)
model = PPO.load("models/trade_ppo")

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
