from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import math

import pandas as pd

from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


# =========================
# Helpers
# =========================

def _f(x: Any, default: float = 0.0) -> float:
    """Safe float for numpy/pandas scalars."""
    try:
        if x is None:
            return float(default)
        if isinstance(x, (pd.Timestamp,)):
            return float(default)
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _s(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _json_safe(obj: Any) -> Any:
    """
    Django JsonResponse sometimes chokes on numpy.bool_ / numpy types.
    This makes everything JSON-serializable.
    """
    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # pandas/numpy scalars
    if hasattr(obj, "item"):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass

    # dict
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]

    # fallback
    return str(obj)


# =========================
# Data Fetch
# =========================

def _fetch_history(symbol: str) -> Optional[pd.DataFrame]:
    """
    Expected output columns (minimum):
      Date (optional), Open, High, Low, Close
    If you already have provider: core.data_providers.finpy_provider.fetch_daily_history
    we use it.
    """
    try:
        from core.data_providers.finpy_provider import fetch_daily_history
        df = fetch_daily_history(symbol)
        return df
    except Exception:
        return None


# =========================
# Indicators
# =========================

def _calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
    """
    Returns:
      supertrend_line (float Series)
      supertrend_dir (int Series: +1 up, -1 down)
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    atr = AverageTrueRange(high, low, close, window=period).average_true_range()

    hl2 = (high + low) / 2.0
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    st = pd.Series(index=df.index, dtype="float64")
    direction = pd.Series(index=df.index, dtype="int64")

    # initialize
    st.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]

        cur_ub = upperband.iloc[i]
        cur_lb = lowerband.iloc[i]
        prev_ub = upperband.iloc[i - 1]
        prev_lb = lowerband.iloc[i - 1]

        # final upper/lower band
        final_ub = cur_ub if (cur_ub < prev_ub or close.iloc[i - 1] > prev_ub) else prev_ub
        final_lb = cur_lb if (cur_lb > prev_lb or close.iloc[i - 1] < prev_lb) else prev_lb

        # direction switch logic
        if prev_dir == 1:
            if close.iloc[i] < final_lb:
                cur_dir = -1
            else:
                cur_dir = 1
        else:
            if close.iloc[i] > final_ub:
                cur_dir = 1
            else:
                cur_dir = -1

        direction.iloc[i] = cur_dir
        st.iloc[i] = final_lb if cur_dir == 1 else final_ub

    return st, direction


def _structure_state(close: pd.Series, pivot: int = 3) -> Dict[str, Any]:
    """
    Very light "market structure" heuristic:
    - detect recent swing highs/lows with small pivot window
    - compare last two highs and lows => HH/HL or LH/LL
    """
    c = close.astype(float).reset_index(drop=True)
    n = len(c)
    if n < pivot * 2 + 5:
        return {"state": "unknown", "hh": False, "hl": False, "lh": False, "ll": False}

    highs = []
    lows = []

    for i in range(pivot, n - pivot):
        window = c.iloc[i - pivot:i + pivot + 1]
        if c.iloc[i] == window.max():
            highs.append((i, float(c.iloc[i])))
        if c.iloc[i] == window.min():
            lows.append((i, float(c.iloc[i])))

    if len(highs) < 2 or len(lows) < 2:
        return {"state": "unknown", "hh": False, "hl": False, "lh": False, "ll": False}

    h1, h2 = highs[-2], highs[-1]
    l1, l2 = lows[-2], lows[-1]

    hh = h2[1] > h1[1]
    lh = h2[1] < h1[1]
    hl = l2[1] > l1[1]
    ll = l2[1] < l1[1]

    if hh and hl:
        state = "bullish"
    elif lh and ll:
        state = "bearish"
    else:
        state = "mixed"

    return {"state": state, "hh": hh, "hl": hl, "lh": lh, "ll": ll}


def _ema_slope_percent_per_bar(ema: pd.Series, bars: int = 20) -> float:
    """
    slope of EMA in percent per bar, using simple linear approx:
      (ema_last - ema_prev) / ema_prev * 100 / bars
    """
    if len(ema) < bars + 1:
        return 0.0
    last = float(ema.iloc[-1])
    prev = float(ema.iloc[-(bars + 1)])
    if prev == 0:
        return 0.0
    return ((last - prev) / prev) * 100.0 / float(bars)


# =========================
# Main Analyzer
# =========================

def analyze_symbol(symbol: str) -> Dict[str, Any]:
    symbol = (symbol or "").strip()
    if not symbol:
        return {"error": "symbol is required"}

    df = _fetch_history(symbol)
    if df is None or df.empty:
        return {"error": "no data"}

    # Normalize expected columns
    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            return {"error": f"missing column: {col}"}

    # limit
    df = df.tail(300).reset_index(drop=True)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    last_price = float(close.iloc[-1])

    # ============ Lookbacks (keep for transparency) ============
    lookbacks = {
        "history_bars_used": int(len(df)),
        "rsi": 14,
        "adx": 14,
        "atr": 14,
        "ema20": 20,
        "ema50": 50,
        "ema100": 100,
        "supertrend_period": 10,
        "supertrend_multiplier": 3.0,
        "slope_bars": 20,
        "return_window_bars": 60,   # برای بازده تقریبی (مثلاً ~3 ماه کاری)
    }

    # ============ Indicators ============
    e20_series = EMAIndicator(close, window=20).ema_indicator()
    e50_series = EMAIndicator(close, window=50).ema_indicator()
    e100_series = EMAIndicator(close, window=100).ema_indicator()

    e20 = _f(e20_series.iloc[-1])
    e50 = _f(e50_series.iloc[-1])
    e100 = _f(e100_series.iloc[-1])

    rsi_series = RSIIndicator(close, window=lookbacks["rsi"]).rsi()
    rsi = _f(rsi_series.iloc[-1])

    adx_ind = ADXIndicator(high, low, close, window=lookbacks["adx"])
    adx_series = adx_ind.adx()
    pdi_series = adx_ind.adx_pos()
    mdi_series = adx_ind.adx_neg()

    adx_last = _f(adx_series.iloc[-1])
    pdi_last = _f(pdi_series.iloc[-1])
    mdi_last = _f(mdi_series.iloc[-1])

    atr_series = AverageTrueRange(high, low, close, window=lookbacks["atr"]).average_true_range()
    atr_last = _f(atr_series.iloc[-1])

    st_line, st_dir = _calc_supertrend(df, period=lookbacks["supertrend_period"], multiplier=lookbacks["supertrend_multiplier"])
    st_last = _f(st_line.iloc[-1])
    st_dir_last = int(_f(st_dir.iloc[-1], 1))  # +1 or -1

    slope_pct = _f(_ema_slope_percent_per_bar(e20_series, bars=lookbacks["slope_bars"]))

    struct = _structure_state(close)

    # ============ Trend ============
    ema_trend_up = (e20 > e50 > e100)
    ema_trend_down = (e20 < e50 < e100)

    if ema_trend_up:
        trend = "up"
    elif ema_trend_down:
        trend = "down"
    else:
        trend = "range"

    # ============ Return (keep your previous style) ============
    # return_percent based on return_window_bars
    ret_window = int(lookbacks["return_window_bars"])
    if len(close) > ret_window:
        base = float(close.iloc[-(ret_window + 1)])
        return_percent = ((last_price - base) / base * 100.0) if base else 0.0
    else:
        return_percent = 0.0

    # ============ Risk / Targets ============
    # ساده و قابل‌فهم: SL=1.5ATR / TP=2ATR
    stop_loss = last_price - (1.5 * atr_last)
    take_profit = last_price + (2.0 * atr_last)

    risk_percent = ((last_price - stop_loss) / last_price * 100.0) if last_price else 0.0

    # ============ Recommendation ============
    # منطق پایه (همون سبک قبلی) + آماده برای شفاف‌سازی با reasons
    # buy: trend up + adx strong + supertrend up + rsi not overbought
    # sell: trend down + adx strong + supertrend down
    if trend == "up" and adx_last >= 25 and st_dir_last == 1 and rsi < 70:
        recommendation = "buy"
    elif trend == "down" and adx_last >= 25 and st_dir_last == -1:
        recommendation = "sell"
    else:
        recommendation = "neutral"

    # =========================
    # ✅ Reasons (NEW)
    # =========================
    reasons = []

    # EMA alignment
    if ema_trend_up:
        reasons.append("EMA20 بالای EMA50 و EMA50 بالای EMA100 است (چیدمان صعودی).")
    elif ema_trend_down:
        reasons.append("EMA20 پایین EMA50 و EMA50 پایین EMA100 است (چیدمان نزولی).")
    else:
        reasons.append("چیدمان EMAها یک‌دست نیست (احتمالاً بازار رنج/نامطمئن).")

    # ADX strength
    if adx_last >= 25:
        reasons.append("قدرت روند مناسب است (ADX بالای ۲۵).")
    else:
        reasons.append("قدرت روند ضعیف است (ADX زیر ۲۵).")

    # DI dominance
    if pdi_last > mdi_last:
        reasons.append("قدرت خریداران بیشتر است (+DI > -DI).")
    elif mdi_last > pdi_last:
        reasons.append("قدرت فروشندگان بیشتر است (-DI > +DI).")
    else:
        reasons.append("قدرت خریدار و فروشنده نزدیک به هم است (+DI ≈ -DI).")

    # SuperTrend
    if st_dir_last == 1:
        reasons.append("SuperTrend صعودی است.")
    else:
        reasons.append("SuperTrend نزولی است.")

    # RSI warnings
    if rsi >= 70:
        reasons.append("هشدار: RSI در محدوده اشباع خرید است (بالای ۷۰).")
    elif rsi <= 30:
        reasons.append("هشدار: RSI در محدوده اشباع فروش است (زیر ۳۰).")
    else:
        reasons.append("RSI در محدوده نرمال است.")

    # Structure
    st_state = _s(struct.get("state"))
    if st_state == "bullish":
        reasons.append("ساختار بازار صعودی است (HH و HL).")
    elif st_state == "bearish":
        reasons.append("ساختار بازار نزولی است (LH و LL).")
    elif st_state == "mixed":
        reasons.append("ساختار بازار ترکیبی/نامشخص است.")
    else:
        reasons.append("ساختار بازار قابل تشخیص نبود (داده/پیوت کافی نیست).")

    # Slope
    if slope_pct > 0:
        reasons.append(f"شیب EMA20 مثبت است (~{slope_pct:.4f}% به‌ازای هر کندل).")
    elif slope_pct < 0:
        reasons.append(f"شیب EMA20 منفی است (~{slope_pct:.4f}% به‌ازای هر کندل).")
    else:
        reasons.append("شیب EMA20 نزدیک به صفر است.")

    # Final reason
    if recommendation == "buy":
        reasons.append("جمع‌بندی: ترکیب روند/قدرت/سوپرترند به نفع خرید است.")
    elif recommendation == "sell":
        reasons.append("جمع‌بندی: ترکیب روند/قدرت/سوپرترند به نفع فروش است.")
    else:
        reasons.append("جمع‌بندی: سیگنال قطعی نیست (بهتر است منتظر تأیید بمانیم).")

    # =========================
    # Output (KEEP EVERYTHING + add reasons)
    # =========================
    out = {
        "symbol": symbol,
        "last_price": round(_f(last_price), 2),
        "trend": trend,
        "rsi": round(_f(rsi), 2),
        "return_percent": round(_f(return_percent), 2),
        "risk_percent": round(_f(risk_percent), 2),
        "stop_loss": round(_f(stop_loss), 2),
        "take_profit": round(_f(take_profit), 2),
        "recommendation": recommendation,

        # ✅ NEW
        "reasons": reasons,

        # metrics (for later charts) - KEEP
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
            "slope_percent_per_bar": round(_f(slope_pct), 4),
            "structure": {
                "state": _s(struct.get("state")),
                "hh": bool(struct.get("hh")),
                "hl": bool(struct.get("hl")),
                "lh": bool(struct.get("lh")),
                "ll": bool(struct.get("ll")),
            },
        },

        # keep lookbacks
        "lookbacks": lookbacks,
    }

    return _json_safe(out)
