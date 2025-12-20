from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import math

import numpy as np
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import AverageTrueRange

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
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)

    atr = AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
    hl2 = (high + low) / 2.0
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    st.iloc[0] = upperband.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        prev_st = st.iloc[i - 1]
        prev_dir = direction.iloc[i - 1]

        curr_upper = upperband.iloc[i]
        curr_lower = lowerband.iloc[i]
        prev_upper = upperband.iloc[i - 1]
        prev_lower = lowerband.iloc[i - 1]

        if curr_upper > prev_upper and close.iloc[i - 1] <= prev_upper:
            curr_upper = prev_upper
        if curr_lower < prev_lower and close.iloc[i - 1] >= prev_lower:
            curr_lower = prev_lower

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
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    if len(df) < lookback + 5:
        return {"state": "unknown", "hh": False, "hl": False, "lh": False, "ll": False}

    w = min(lookback, len(df) - 1)
    h = high.tail(w).reset_index(drop=True)
    l = low.tail(w).reset_index(drop=True)

    mid = w // 2
    h1, h2 = float(h.iloc[:mid].max()), float(h.iloc[mid:].max())
    l1, l2 = float(l.iloc[:mid].min()), float(l.iloc[mid:].min())

    hh = h2 > h1
    ll = l2 < l1
    hl = l2 > l1
    lh = h2 < h1

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
    denom = (x.var() if x.var() != 0 else 1.0)
    slope = np.cov(x, y, bias=True)[0, 1] / denom
    last = float(y[-1]) if float(y[-1]) != 0 else 1.0
    return (slope / last) * 100.0


# ---------------------------
# Live helpers (AllSymbols)
# ---------------------------
def _live_power_ratio(live: Dict[str, Any]) -> Optional[float]:
    """
    نسبت قدرت خریدار حقیقی به فروش حقیقی (بر اساس حجم)
    > 1.2 مثبت، < 0.8 منفی
    """
    if not live:
        return None
    buy_i = _f(live.get("Buy_I_Volume"), 0.0)
    sell_i = _f(live.get("Sell_I_Volume"), 0.0)
    if buy_i <= 0 or sell_i <= 0:
        return None
    return buy_i / sell_i


def _live_orderbook_pressure(live: Dict[str, Any]) -> Dict[str, Any]:
    """
    تفسیر ساده صف خرید/فروش از لایه 1
    qd1/pd1/zd1 (buy)
    qo1/po1/zo1 (sell)
    """
    if not live:
        return {"buy_queue": 0.0, "sell_queue": 0.0, "signal": "unknown"}

    qd1 = _f(live.get("qd1"), 0.0)
    qo1 = _f(live.get("qo1"), 0.0)
    pd1 = _f(live.get("pd1"), 0.0)
    po1 = _f(live.get("po1"), 0.0)

    buy_q = qd1
    sell_q = qo1

    signal = "neutral"
    if buy_q > sell_q * 1.5 and pd1 > 0:
        signal = "buy_pressure"
    elif sell_q > buy_q * 1.5 and po1 > 0:
        signal = "sell_pressure"

    return {"buy_queue": buy_q, "sell_queue": sell_q, "signal": signal}


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
    live: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str, List[str], Dict[str, float]]:
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

    # Live confirmation / contradiction
    if live:
        pl = _f(live.get("pl"), 0.0)
        pc = _f(live.get("pc"), 0.0)
        py = _f(live.get("py"), 0.0)
        plc = _f(live.get("plc"), 0.0)

        if ema_bull and pc > 0 and pl > 0 and (pl < pc) and (plc < 0):
            trend_score -= 3
            reasons.append("⚠️ با وجود روند تکنیکال صعودی، قیمت آخر زیر پایانی و تغییر منفی است (تایید لایو ضعیف).")

        if ema_bull and py > 0 and pl > 0 and (pl > py) and (plc > 0):
            trend_score += 2
            reasons.append("✅ لایو: قیمت آخر بالاتر از دیروز و تغییر مثبت است (تایید روند).")

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
        reasons.append("RSI خیلی بالا است (احتمال اصلاح کوتاه‌مدت).")
    elif 35 <= rsi < 50:
        mom_score += 10
        reasons.append("RSI زیر ۵۰ است (قدرت خرید متوسط/ضعیف).")
    else:
        mom_score += 7
        reasons.append("RSI پایین است (می‌تواند فرصت برگشت باشد اما پرریسک).")

    if pdi > mdi:
        mom_score += 5
        reasons.append("+DI بالاتر از -DI است (غلبه قدرت خرید).")
    else:
        mom_score += 2
        reasons.append("-DI بالاتر از +DI است (غلبه قدرت فروش).")

    if st_dir == 1:
        mom_score += 2
        reasons.append("سوپرترند در فاز صعودی است.")
    else:
        reasons.append("سوپرترند در فاز نزولی است.")

    if live:
        ratio = _live_power_ratio(live)
        if ratio is not None:
            if ratio >= 1.2:
                mom_score += 3
                reasons.append(f"✅ قدرت خریدار حقیقی بالاست (Buy_I/Sell_I≈{round(ratio, 2)}).")
            elif ratio <= 0.8:
                mom_score -= 2
                reasons.append(f"⚠️ قدرت فروش حقیقی بالاتر است (Buy_I/Sell_I≈{round(ratio, 2)}).")
            else:
                reasons.append(f"قدرت خریدار/فروش حقیقی متعادل است (Buy_I/Sell_I≈{round(ratio, 2)}).")

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

    if live:
        ob = _live_orderbook_pressure(live)
        if ob.get("signal") == "buy_pressure":
            struct_score += 1.5
            reasons.append("✅ فشار تقاضا در سفارشات (سطح ۱) بیشتر از عرضه است.")
        elif ob.get("signal") == "sell_pressure":
            struct_score -= 1.5
            reasons.append("⚠️ فشار عرضه در سفارشات (سطح ۱) بیشتر از تقاضاست.")

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

    # 5) Volatility (10)
    vol_score = 0.0
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

    total = trend_score + mom_score + struct_score + rr_score + vol_score
    score = int(round(_clip(total, 0, 100)))

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
# Safe analyze without history
# ---------------------------
def _analyze_with_live_only(symbol: str, live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    اگر دیتای تاریخی (finpy) در دسترس نبود، خروجی را با داده‌های لایو ادامه می‌دهیم
    تا endpoint /api/analyze نخوابد و 500 ندهد.
    """
    pl = _f((live or {}).get("pl"), 0.0)
    plp = _f((live or {}).get("plp"), 0.0)
    plc = _f((live or {}).get("plc"), 0.0)

    if plc > 0 or plp > 0:
        trend = "up"
    elif plc < 0 or plp < 0:
        trend = "down"
    else:
        trend = "range"

    live_summary = None
    if live:
        live_summary = {
            "l18": _s(live.get("l18")),
            "l30": _s(live.get("l30")),
            "cs": _s(live.get("cs")),
            "time": _s(live.get("time")),
            "pl": _f(live.get("pl"), 0.0),
            "pc": _f(live.get("pc"), 0.0),
            "py": _f(live.get("py"), 0.0),
            "plc": _f(live.get("plc"), 0.0),
            "plp": _f(live.get("plp"), 0.0),
        }

    out: Dict[str, Any] = {
        "symbol": symbol,
        "last_price": round(pl, 2),
        "trend": trend,
        "rsi": None,
        "risk_percent": None,
        "return_percent": 0.0,
        "stop_loss": None,
        "take_profit": None,
        "recommendation": "hold",
        "score": 50,
        "reasons": [
            "تحلیل بر اساس داده‌های لحظه‌ای انجام شد.",
            "دیتای تاریخی (finpy / old.tsetmc) در دسترس نبود یا اتصال قطع شد.",
        ],
        "live": live_summary,
        "metrics": {
            "ema20": None,
            "ema50": None,
            "ema100": None,
            "adx": None,
            "plus_di": None,
            "minus_di": None,
            "supertrend_dir": None,
            "supertrend": None,
            "atr": None,
            "atr_percent": None,
            "slope_percent_per_bar": None,
            "rr": None,
            "score_components": {},
            "structure": {
                "state": "unknown",
                "hh": False,
                "hl": False,
                "lh": False,
                "ll": False,
            },
        },
    }
    return out


# ---------------------------
# Main public function
# ---------------------------
def analyze_symbol(symbol: str, live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    symbol = (symbol or "").strip()
    if not symbol:
        return {"error": "symbol is required"}

    # ✅ مهم: اگر finpy کند/قطع شود، endpoint نخوابد
    try:
        df = fetch_daily_history(symbol)
    except Exception as e:
        # تحلیل را نکُش؛ فقط با لایو ادامه بده
        # (این print را اگر لاگینگ داری تبدیل به logger.warning کن)
        print(f"[WARN] fetch_daily_history failed for {symbol}: {e}")
        return _analyze_with_live_only(symbol, live)

    if df is None or df.empty:
        return _analyze_with_live_only(symbol, live)

    df = df.tail(300).copy().reset_index(drop=True)

    required_cols = {"Open", "High", "Low", "Close"}
    if not required_cols.issubset(set(df.columns)):
        # اگر ستون‌ها مشکل داشتند هم، باز 500 ندهیم
        return _analyze_with_live_only(symbol, live)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    last_price = _f(close.iloc[-1])

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

    slope_pct = _f(_slope_percent_per_bar(close, window=30))
    struct = _market_structure(df, lookback=40)

    if e20 > e50 > e100:
        trend = "up"
    elif e20 < e50 < e100:
        trend = "down"
    else:
        trend = "range"

    atr_pct = (atr_last / last_price * 100.0) if last_price > 0 else 0.0

    if trend == "up" or st_dir_last == 1:
        stop_loss = min(st_last, last_price - 2.0 * atr_last)
        take_profit = last_price + 2.0 * atr_last
    elif trend == "down" or st_dir_last == -1:
        stop_loss = max(st_last, last_price + 2.0 * atr_last)
        take_profit = max(0.0, last_price - 2.0 * atr_last)
    else:
        stop_loss = last_price - 2.0 * atr_last
        take_profit = last_price + 2.0 * atr_last

    stop_loss = _f(stop_loss)
    take_profit = _f(take_profit)

    risk_percent = 0.0
    if last_price > 0:
        risk_percent = abs((last_price - stop_loss) / last_price) * 100.0

    reward = abs(take_profit - last_price)
    risk = abs(last_price - stop_loss)
    rr = (reward / risk) if risk > 0 else 0.0

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
        live=live,
    )

    return_percent = 0.0

    # live summary (for UI)
    live_summary = None
    if live:
        live_summary = {
            "l18": _s(live.get("l18")),
            "l30": _s(live.get("l30")),
            "cs": _s(live.get("cs")),
            "time": _s(live.get("time")),
            "pl": _f(live.get("pl"), 0.0),
            "pc": _f(live.get("pc"), 0.0),
            "py": _f(live.get("py"), 0.0),
            "plc": _f(live.get("plc"), 0.0),
            "plp": _f(live.get("plp"), 0.0),
        }

    out: Dict[str, Any] = {
        "symbol": symbol,
        "last_price": round(last_price, 2),
        "trend": trend,
        "rsi": round(rsi, 2),
        "risk_percent": round(_f(risk_percent), 2),
        "return_percent": round(_f(return_percent), 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "recommendation": recommendation,
        "score": _i(score),
        "reasons": [_s(r) for r in reasons],
        "live": live_summary,
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
    }

    return out
