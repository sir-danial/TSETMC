import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

from core.data_providers.finpy_provider import fetch_daily_history


def analyze_symbol(symbol: str) -> dict:
    """
    Main analyzer for a single symbol
    Returns JSON-serializable dict
    """

    df = fetch_daily_history(symbol)

    if df is None or df.empty:
        return {
            "error": "no data"
        }

    df = df.tail(100).copy()

    close = df["Close"]

    # =====================
    # RSI
    # =====================
    rsi_indicator = RSIIndicator(close=close, window=14)
    df["rsi"] = rsi_indicator.rsi()
    last_rsi = float(df["rsi"].iloc[-1])

    # =====================
    # Trend (EMA)
    # =====================
    ema_fast = EMAIndicator(close=close, window=20).ema_indicator()
    ema_slow = EMAIndicator(close=close, window=50).ema_indicator()

    trend = "up" if ema_fast.iloc[-1] > ema_slow.iloc[-1] else "down"

    # =====================
    # Prices
    # =====================
    last_price = float(close.iloc[-1])

    recent_low = float(df["Low"].tail(20).min())
    recent_high = float(df["High"].tail(20).max())

    stop_loss = recent_low
    take_profit = recent_high

    risk_percent = abs((last_price - stop_loss) / last_price) * 100
    return_percent = abs((take_profit - last_price) / last_price) * 100

    # =====================
    # Recommendation
    # =====================
    if trend == "up" and last_rsi < 70:
        recommendation = "buy"
    elif last_rsi > 70:
        recommendation = "sell"
    else:
        recommendation = "neutral"

    return {
        "symbol": symbol,
        "last_price": round(last_price, 2),
        "trend": trend,
        "rsi": round(last_rsi, 2),
        "risk_percent": round(risk_percent, 2),
        "return_percent": round(return_percent, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "recommendation": recommendation
    }
