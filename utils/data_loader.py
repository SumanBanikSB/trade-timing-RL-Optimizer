import yfinance as yf
import pandas as pd

def download_data(ticker="AAPL", start="2015-01-01", end="2024-01-01", interval="1d"):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.to_csv("data/aapl.csv")
    return df
