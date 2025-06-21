import pandas as pd
import numpy as np
import joblib
import ta
import streamlit as st

def preprocess(df):
    df = df.copy()
    close = df['close'].squeeze()
    df['rsi'] = ta.momentum.RSIIndicator(close).rsi()
    df['macd'] = ta.trend.MACD(close).macd_diff()
    df['ema_12'] = ta.trend.EMAIndicator(close, window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close, window=26).ema_indicator()
    df['volatility'] = df["close"]#ta.volatility.AverageTrueRange(
        #high=df.iloc[:,1], low=df.iloc[:,2], close=close).average_true_range()
    df['return'] = close.pct_change()

    for col in ['rsi', 'macd', 'ema_12', 'ema_26', 'volatility']:
        df[f'{col}_lag1'] = df[col].shift(1)

    df.dropna(inplace=True)
    return df


def make_prediction(df, model_path="xgb_model.pkl"):
    required_cols = ['rsi_lag1', 'macd_lag1', 'ema_12_lag1', 'ema_26_lag1',
                     'volatility_lag1', 'volume', 'close', 'return']

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"⚠️ Colonnes manquantes pour la prédiction : {missing}")
        return None

    model = joblib.load(model_path)
    X = df[required_cols].iloc[-1:]
    proba = model.predict_proba(X)[0, 1]
    return proba
