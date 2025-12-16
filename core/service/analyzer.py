from __future__ import annotations

from typing import Any, Dict, List, Tuple
import math

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

# اگر نام تابع دیتاپرووایدر شما فرق دارد، فقط این import را مطابق پروژه خودت درست کن
from core.data_providers.finpy_provider import fetch_daily_history


# ---------------------------
# Helpers: JSON-safe casting
# ---------------------------
def _f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        if isinstance(x, (np.floating, np.integer)):
            return float(x)
        if isinstance(x, (pd.Timestamp,)):
            return float(default)
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _i(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, (np.integer,)):
            return int(x)
        return int(float(x))
    except Exception:
        return int(default)


def _s(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ---------------------------
# SuperTrend (simple)
# ---------------------------
def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      st: SuperTrend line
      direction: 1 (bullish) / -1 (bearish)
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    atr = AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
    hl2 = (high + low) / 2.0
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    # init
    st.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]

        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]
        prev_upper = upperband.iloc[i - 1]
        prev_lower = lowerband.iloc[i - 1]

        # bands "final"
        if curr_upper > prev_upper and close.iloc[i - 1] <= prev_upper:
            curr_upper = prev_upper
        if curr_lower < prev_lower and close.iloc[i - 1] >= prev_lower:
            curr_lower = prev_lower

        # direction
        if prev_st == prev_upper:
            curr_dir = 1 if close.iloc[i] > curr_upper else -1
        else:
            curr_dir = -1 if close.iloc[i] < curr_lower else 1

        direction.iloc[i] = curr_dir
        st.iloc[i] = curr_lower if curr_dir == 1 else curr_upper

    return st, direction


# ---------------------------
# Structure (very light)
# ---------------------------
def _market_structure(df: pd.DataFrame, lookback: int = 40) -> Dict[str, Any]:
    """
    Very lightweight HH/HL/LH/LL detection using recent swing approximation.
    Output is JSON-safe.
    """
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    if len(df) < lookback + 5:
        return {"state": "unknown", "hh": False, "hl": False, "lh": False, "ll": False}

    w = min(lookback, len(df) - 1)
    h = high.tail(w).reset_index(drop=True)
    l = low.tail(w).reset_index(drop=True)

    # pick "recent extremes" in two halves to estimate swing
    mid = w // 2
    h1, h2 = float(h.iloc[:mid].max()), float(h.iloc[mid:].max())
    l1, l2 = float(l.iloc[:mid].min()), float(l.iloc[mid:].min())

    hh = h2 > h1
    ll = l2 < l1
    hl = l2 > l1
    lh = h2 < h1

    # infer state
    if hh and hl:
        state = "uptrend"
    elif ll and lh:
        state = "downtrend"
    else:
        state = "range"

    return {"state": state, "hh": bool(hh), "hl": bool(hl), "lh": bool(lh), "ll": bool(ll)}


def _slope_percent_per_bar(series: pd.Series, window: int = 30) -> float:
    s = series.astype(float).dropna()
    if len(s) < window:
        return 0.0
    y = s.tail(window).values
    x = np.arange(len(y))
    # simple linear regression slope
    denom = (x.var() if x.var() != 0 else 1.0)
    slope = np.cov(x, y, bias=True)[0, 1] / denom
    last = float(y[-1]) if float(y[-1]) != 0 else 1.0
    return (slope / last) * 100.0


# ---------------------------
# Score engine
# ---------------------------
def _score_engine(
    *,
    last_price: float,
    e20: float,
    e50: float,
    e100: float,
    slope_pct: float,
    adx: float,
    pdi: float,
    mdi: float,
    rsi: float,
    struct: Dict[str, Any],
    rr: float,
    atr_pct: float,
    st_dir: int,
) -> Tuple[int, str, List[str], Dict[str, float]]:
    """
    Returns: (score_0_100, recommendation, reasons[], component_scores)
    """
    reasons: List[str] = []
    comp: Dict[str, float] = {}

    # 1) Trend (30)
    trend_score = 0.0
    ema_bull = (e20 > e50 > e100)
    ema_bear = (e20 < e50 < e100)

    if ema_bull:
        trend_score += 16
        reasons.append("EMA20 بالاتر از EMA50 و EMA100 است (روند کلی صعودی).")
    elif ema_bear:
        trend_score += 6
        reasons.append("EMA20 پایین‌تر از EMA50 و EMA100 است (روند کلی نزولی).")
    else:
        trend_score += 10
        reasons.append("EMAها هم‌راستا نیستند (بازار می‌تواند رنج/درحال تغییر فاز باشد).")

    if slope_pct > 0.03:
        trend_score += 7
        reasons.append("شیب قیمت مثبت است (شتاب رشد قابل قبول).")
    elif slope_pct < -0.03:
        trend_score += 2
        reasons.append("شیب قیمت منفی است (ریسک ادامه افت).")
    else:
        trend_score += 4
        reasons.append("شیب قیمت کم است (حرکت کند/رنج).")

    if adx >= 25:
        trend_score += 7
        reasons.append(f"ADX حدود {round(adx, 1)} است (روند معتبر/قوی).")
    elif adx >= 20:
        trend_score += 5
        reasons.append(f"ADX حدود {round(adx, 1)} است (روند متوسط).")
    else:
        trend_score += 2
        reasons.append(f"ADX حدود {round(adx, 1)} است (روند ضعیف/رنج).")

    trend_score = _clip(trend_score, 0, 30)
    comp["trend"] = trend_score

    # 2) Momentum (25)
    mom_score = 0.0
    if 50 <= rsi <= 65:
        mom_score += 18
        reasons.append("RSI در محدوده مناسب (۵۰ تا ۶۵) است.")
    elif 65 < rsi <= 75:
        mom_score += 12
        reasons.append("RSI بالاتر از ۶۵ است (قدرت خرید خوب، اما نزدیک اشباع).")
    elif rsi > 75:
        mom_score += 6
        reasons.append("RSI خیلی بالا است (احتمال اصلاح/فشار فروش کوتاه‌مدت).")
    elif 35 <= rsi < 50:
        mom_score += 10
        reasons.append("RSI زیر ۵۰ است (قدرت خرید متوسط/ضعیف).")
    else:
        mom_score += 7
        reasons.append("RSI پایین است (می‌تواند فرصت برگشت باشد اما پرریسک).")

    # DI direction bonus
    if pdi > mdi:
        mom_score += 5
        reasons.append("+DI بالاتر از -DI است (غلبه قدرت خرید).")
    else:
        mom_score += 2
        reasons.append("-DI بالاتر از +DI است (غلبه قدرت فروش).")

    # Supertrend direction bonus
    if st_dir == 1:
        mom_score += 2
        reasons.append("سوپرترند در فاز صعودی است.")
    else:
        mom_score += 0
        reasons.append("سوپرترند در فاز نزولی است.")

    mom_score = _clip(mom_score, 0, 25)
    comp["momentum"] = mom_score

    # 3) Structure (20)
    struct_score = 0.0
    state = _s(struct.get("state"))
    hh = bool(struct.get("hh"))
    hl = bool(struct.get("hl"))
    lh = bool(struct.get("lh"))
    ll = bool(struct.get("ll"))

    if state == "uptrend" and hh and hl:
        struct_score += 18
        reasons.append("ساختار بازار HH/HL دارد (روند ساختاری صعودی).")
    elif state == "downtrend" and ll and lh:
        struct_score += 6
        reasons.append("ساختار بازار LL/LH دارد (روند ساختاری نزولی).")
    else:
        struct_score += 12
        reasons.append("ساختار بازار رنج/نامشخص است (به شکست سطح‌ها حساس).")

    struct_score = _clip(struct_score, 0, 20)
    comp["structure"] = struct_score

    # 4) Risk/Reward (15)
    rr_score = 0.0
    if rr >= 2.0:
        rr_score += 15
        reasons.append(f"نسبت ریسک به بازده مناسب است (R/R≈{round(rr, 2)}).")
    elif rr >= 1.5:
        rr_score += 11
        reasons.append(f"نسبت ریسک به بازده متوسط است (R/R≈{round(rr, 2)}).")
    elif rr >= 1.2:
        rr_score += 7
        reasons.append(f"نسبت ریسک به بازده ضعیف است (R/R≈{round(rr, 2)}).")
    else:
        rr_score += 4
        reasons.append(f"نسبت ریسک به بازده پایین است (R/R≈{round(rr, 2)}).")

    rr_score = _clip(rr_score, 0, 15)
    comp["rr"] = rr_score

    # 5) Volatility (10) - ATR%
    vol_score = 0.0
    # ATR% = ATR / price * 100
    if atr_pct <= 2.0:
        vol_score += 9
        reasons.append("نوسان (ATR%) پایین/مناسب است (ریسک کنترل‌شده).")
    elif atr_pct <= 4.0:
        vol_score += 7
        reasons.append("نوسان (ATR%) متوسط است.")
    elif atr_pct <= 6.0:
        vol_score += 5
        reasons.append("نوسان (ATR%) بالاست (ریسک افزایش می‌یابد).")
    else:
        vol_score += 3
        reasons.append("نوسان (ATR%) خیلی بالاست (معامله بسیار پرریسک).")

    vol_score = _clip(vol_score, 0, 10)
    comp["volatility"] = vol_score

    # total score
    total = trend_score + mom_score + struct_score + rr_score + vol_score
    score = int(round(_clip(total, 0, 100)))

    # Recommendation
    # Base by score, then adjust by trend direction (ema + supertrend)
    if score >= 75 and (ema_bull or st_dir == 1):
        rec = "buy"
        reasons.append("امتیاز کلی بالا است و شرایط کلی به نفع خرید است.")
    elif score <= 45 and (ema_bear or st_dir == -1):
        rec = "sell"
        reasons.append("امتیاز کلی پایین است و شرایط کلی به نفع فروش/اجتناب است.")
    else:
        rec = "hold"
        reasons.append("امتیاز کلی متوسط است (بهتر است منتظر تایید بیشتر بمانید).")

    return score, rec, reasons, comp


# ---------------------------
# Main public function
# ---------------------------
def analyze_symbol(symbol: str) -> Dict[str, Any]:
    symbol = (symbol or "").strip()
    if not symbol:
        return {"error": "symbol is required"}

    df = fetch_daily_history(symbol)
    if df is None or df.empty:
        return {"error": "no data"}

    # Keep last 300 bars for stability/perf
    df = df.tail(300).copy().reset_index(drop=True)

    # Ensure columns exist
    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(set(df.columns)):
        return {"error": "bad data columns", "columns": list(df.columns)}

    # Use raw close for decision/display
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    last_price = _f(close.iloc[-1])

    # Indicators
    ema20 = EMAIndicator(close=close, window=20).ema_indicator()
    ema50 = EMAIndicator(close=close, window=50).ema_indicator()
    ema100 = EMAIndicator(close=close, window=100).ema_indicator()

    rsi_series = RSIIndicator(close=close, window=14).rsi()

    adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
    adx = adx_ind.adx()
    pdi = adx_ind.adx_pos()
    mdi = adx_ind.adx_neg()

    atr_series = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    st, st_dir = _supertrend(df, period=10, multiplier=3.0)

    # latest values
    e20 = _f(ema20.iloc[-1])
    e50 = _f(ema50.iloc[-1])
    e100 = _f(ema100.iloc[-1])
    rsi = _f(rsi_series.iloc[-1])
    adx_last = _f(adx.iloc[-1])
    pdi_last = _f(pdi.iloc[-1])
    mdi_last = _f(mdi.iloc[-1])
    atr_last = _f(atr_series.iloc[-1])
    st_last = _f(st.iloc[-1])
    st_dir_last = _i(st_dir.iloc[-1], default=1)

    # Slope (last 30 bars)
    slope_pct = _f(_slope_percent_per_bar(close, window=30))

    # Structure
    struct = _market_structure(df, lookback=40)

    # Trend label (simple)
    if e20 > e50 > e100:
        trend = "up"
    elif e20 < e50 < e100:
        trend = "down"
    else:
        trend = "range"

    # Stop/TP (keep simple & consistent)
    atr_pct = (atr_last / last_price * 100.0) if last_price > 0 else 0.0

    # Default: use SuperTrend as protective line when possible
    if trend == "up" or st_dir_last == 1:
        stop_loss = min(st_last, last_price - 2.0 * atr_last)
        take_profit = last_price + 2.0 * atr_last
    elif trend == "down" or st_dir_last == -1:
        # for "sell" idea: stop above
        stop_loss = max(st_last, last_price + 2.0 * atr_last)
        take_profit = max(0.0, last_price - 2.0 * atr_last)
    else:
        stop_loss = last_price - 2.0 * atr_last
        take_profit = last_price + 2.0 * atr_last

    stop_loss = _f(stop_loss)
    take_profit = _f(take_profit)

    # Risk percent (distance to stop)
    risk_percent = 0.0
    if last_price > 0:
        risk_percent = abs((last_price - stop_loss) / last_price) * 100.0

    # RR ratio
    reward = abs(take_profit - last_price)
    risk = abs(last_price - stop_loss)
    rr = (reward / risk) if risk > 0 else 0.0

    # Score engine
    score, recommendation, reasons, comp_scores = _score_engine(
        last_price=last_price,
        e20=e20,
        e50=e50,
        e100=e100,
        slope_pct=slope_pct,
        adx=adx_last,
        pdi=pdi_last,
        mdi=mdi_last,
        rsi=rsi,
        struct=struct,
        rr=rr,
        atr_pct=atr_pct,
        st_dir=st_dir_last,
    )

    # Keep return_percent as earlier placeholder (0) unless you already compute it elsewhere
    return_percent = 0.0

    # IMPORTANT: JSON-safe output
    out: Dict[str, Any] = {
        "symbol": symbol,
        "last_price": round(last_price, 2),
        "trend": trend,
        "rsi": round(rsi, 2),
        "risk_percent": round(_f(risk_percent), 2),
        "return_percent": round(_f(return_percent), 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "recommendation": recommendation,   # buy/sell/hold
        "score": _i(score),
        "reasons": [_s(r) for r in reasons],
        # metrics (for later charts / debugging)
        "metrics": {
            "ema20": round(_f(e20), 2),
            "ema50": round(_f(e50), 2),
            "ema100": round(_f(e100), 2),
            "adx": round(_f(adx_last), 2),
            "plus_di": round(_f(pdi_last), 2),
            "minus_di": round(_f(mdi_last), 2),
            "supertrend_dir": int(st_dir_last),
            "supertrend": round(_f(st_last), 2),
            "atr": round(_f(atr_last), 2),
            "atr_percent": round(_f(atr_pct), 2),
            "slope_percent_per_bar": round(_f(slope_pct), 4),
            "rr": round(_f(rr), 2),
            "score_components": {k: round(_f(v), 2) for k, v in comp_scores.items()},
            "structure": {
                "state": _s(struct.get("state")),
                "hh": bool(struct.get("hh")),
                "hl": bool(struct.get("hl")),
                "lh": bool(struct.get("lh")),
                "ll": bool(struct.get("ll")),
            },
        },
        "lookbacks": {
            "data_tail_bars": 300,
            "rsi": 14,
            "adx": 14,
            "atr": 14,
            "ema20": 20,
            "ema50": 50,
            "ema100": 100,
            "supertrend_period": 10,
            "supertrend_multiplier": 3.0,
            "slope_window": 30,
            "structure_lookback": 40,
        },
    }

    return out
