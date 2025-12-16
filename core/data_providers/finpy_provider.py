# core/data_providers/finpy_provider.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import finpy_tse as tse


# ===== Cache settings =====
CACHE_DIR = Path(os.environ.get("TSETMC_CACHE_DIR", ".cache_tsetmc"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = int(os.environ.get("TSETMC_CACHE_TTL", "600"))  # 10 minutes default
DEFAULT_TAIL_ROWS = int(os.environ.get("TSETMC_TAIL_ROWS", "400"))  # only last 400 daily candles


def _cache_path(symbol: str) -> Path:
    safe = symbol.strip().replace("/", "_").replace("\\", "_").replace(" ", "_")
    return CACHE_DIR / f"{safe}.parquet"


def _is_cache_fresh(p: Path) -> bool:
    if not p.exists():
        return False
    age = time.time() - p.stat().st_mtime
    return age <= CACHE_TTL_SECONDS


def fetch_daily_history(symbol: str, tail_rows: int = DEFAULT_TAIL_ROWS) -> pd.DataFrame:
    """
    Fetch daily price history using finpy_tse (network).
    Optimized:
      - uses disk cache (parquet)
      - limits rows to last tail_rows
      - fallback to cache if network fails
    Returns DataFrame with columns including Close/High/Low/... and maybe Adj Close/Adj High/Adj Low.
    """
    symbol = (symbol or "").strip()
    if not symbol:
        return pd.DataFrame()

    cp = _cache_path(symbol)

    # 1) Fresh cache
    if _is_cache_fresh(cp):
        try:
            return pd.read_parquet(cp)
        except Exception:
            pass  # if cache corrupted, continue to refetch

    # 2) Try network fetch
    try:
        df = tse.Get_Price_History(
            stock=symbol,
            start_date="1390-01-01",
            end_date="1405-01-01",
            adjust_price=True,
        )
        if df is None or df.empty:
            # if cache exists return it; otherwise empty
            if cp.exists():
                try:
                    return pd.read_parquet(cp)
                except Exception:
                    return pd.DataFrame()
            return pd.DataFrame()

        df = df.reset_index(drop=False)

        # Some finpy outputs have "Date" or "J-Date". Sort if possible.
        if "Date" in df.columns:
            try:
                df = df.sort_values("Date")
            except Exception:
                pass

        # limit data (huge performance win)
        if tail_rows and len(df) > tail_rows:
            df = df.tail(tail_rows).reset_index(drop=True)

        # write cache
        try:
            df.to_parquet(cp, index=False)
        except Exception:
            pass

        return df

    except Exception:
        # 3) Fallback to old cache if exists
        if cp.exists():
            try:
                return pd.read_parquet(cp)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()
