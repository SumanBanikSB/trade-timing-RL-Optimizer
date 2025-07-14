import pandas as pd
import ta

def add_indicators(df):
    df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume")
    df = df.dropna()
    return df
