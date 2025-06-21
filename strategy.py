import pandas as pd
import numpy as np
import joblib
import ta

def preprocess(df):
    df = df.copy()
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd_diff()
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    df['volatility'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['return'] = df['close'].pct_change()

    for col in ['rsi', 'macd', 'ema_12', 'ema_26', 'volatility']:
        df[f'{col}_lag1'] = df[col].shift(1)

    df.dropna(inplace=True)
    return df

def make_prediction(df, model_path="xgb_model.pkl"):
    model = joblib.load(model_path)
    X = df[['rsi_lag1', 'macd_lag1', 'ema_12_lag1', 'ema_26_lag1',
            'volatility_lag1', 'volume', 'close', 'return']].iloc[-1:]
    proba = model.predict_proba(X)[0, 1]
    return proba
