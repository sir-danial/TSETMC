# core/service/analyzer.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator

from core.data_providers.finpy_provider import fetch_daily_history


def _f(x) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def analyze_symbol(symbol: str) -> Dict[str, Any]:
    symbol = (symbol or "").strip()
    if not symbol:
        return {"error": "symbol is required"}

    df = fetch_daily_history(symbol)
    if df is None or df.empty:
        return {"error": "no data"}

    # مشابه نسخه پایدار: محدودسازی
    df = df.tail(300).reset_index(drop=True)

    # RAW columns for display/levels (correct prices)
    required_raw = ["Close", "High", "Low"]
    for c in required_raw:
        if c not in df.columns:
            return {"error": f"missing column: {c}"}

    close_raw = df["Close"].astype(float)
    high_raw = df["High"].astype(float)
    low_raw = df["Low"].astype(float)

    last_price = float(close_raw.iloc[-1])

    # Adjusted columns ONLY for indicators (if present)
    close_adj = df["Adj Close"].astype(float) if "Adj Close" in df.columns else close_raw
    high_adj = df["Adj High"].astype(float) if "Adj High" in df.columns else high_raw
    low_adj = df["Adj Low"].astype(float) if "Adj Low" in df.columns else low_raw

    # Indicators
    rsi = RSIIndicator(close=close_adj, window=14).rsi()
    ema_fast = EMAIndicator(close=close_adj, window=20).ema_indicator()
    ema_slow = EMAIndicator(close=close_adj, window=50).ema_indicator()

    # ADX
    adx_ind = ADXIndicator(high=high_adj, low=low_adj, close=close_adj, window=14)
    adx = adx_ind.adx()

    last_rsi = _f(rsi.iloc[-1]) or 0.0
    adx_v = _f(adx.iloc[-1]) or 0.0

    trend_direction = "up" if float(ema_fast.iloc[-1]) > float(ema_slow.iloc[-1]) else "down"
    trend_text = "صعودی" if trend_direction == "up" else "نزولی"

    # Stop/TP based on RAW (important)
    lookback = 20
    recent_low = float(low_raw.tail(lookback).min())
    stop_loss = recent_low
    risk_percent = abs((last_price - stop_loss) / last_price) * 100 if last_price else 0.0

    # ساده و قابل‌فهم: تارگت = سقف 20 کندل اخیر (RAW)
    recent_high = float(high_raw.tail(lookback).max())
    take_profit = recent_high

    # امتیاز + دلیل‌ها (برای شفافیت بدون بهم‌ریختن UI)
    score = 0
    reasons: List[str] = []

    # EMA direction
    if trend_direction == "up":
        score += 2
        reasons.append("EMA سریع بالاتر از EMA کند است (روند صعودی)")
    else:
        score -= 2
        reasons.append("EMA سریع پایین‌تر از EMA کند است (روند نزولی)")

    # ADX strength
    if adx_v >= 25:
        score += 1
        reasons.append("ADX بالاست (قدرت روند خوب)")
    elif adx_v < 20:
        score -= 1
        reasons.append("ADX پایین است (بازار رنج/بدون روند)")

    # RSI sanity
    if last_rsi > 70:
        score -= 1
        reasons.append("RSI بالای ۷۰ (اشباع خرید → امتیاز کم شد)")
    elif last_rsi < 30:
        score += 1
        reasons.append("RSI زیر ۳۰ (اشباع فروش → امتیاز اضافه شد)")

    # Risk constraint
    if risk_percent > 7:
        score -= 1
        reasons.append("ریسک بیشتر از ۷٪ (امتیاز کم شد)")

    # Final recommendation
    if score >= 3:
        recommendation = "buy"
    elif score <= -3:
        recommendation = "sell"
    else:
        recommendation = "hold"

    return {
        "symbol": symbol,
        "last_price": round(last_price, 2),          # ✅ RAW درست
        "trend": trend_text,                         # برای UI فارسی
        "rsi": round(last_rsi, 2),
        "adx": round(adx_v, 2),
        "risk_percent": round(risk_percent, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "recommendation": recommendation,

        # اضافات شفاف (UI اگر نشون نده هم مشکلی نیست)
        "score": score,
        "reasons": reasons,
        "data_rows_used": int(len(df)),
    }
