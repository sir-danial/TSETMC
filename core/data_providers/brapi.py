import os
import time
import requests
from datetime import datetime


BRSAPI_URL = "https://brsapi.ir/Api/Tsetmc/History.php"


class BRSAPIError(Exception):
    pass


def fetch_daily_history(symbol: str, max_years: int = 10):
    """
    Fetch daily OHLC history for a TSETMC symbol using brsapi.ir

    Returns:
        List[dict]:
        [
            {
                "date": datetime.date,
                "open": int,
                "high": int,
                "low": int,
                "close": int,
                "volume": int
            },
            ...
        ]
    """

    api_key = os.getenv("BRSAPI_KEY")
    if not api_key:
        raise BRSAPIError("BRSAPI_KEY is not set in environment variables")

    params = {
        "key": api_key,
        "type": 0,        # 0 = Bourse (TSETMC)
        "l18": symbol     # Persian symbol name, e.g. فملی
    }

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0"
    })

    try:
        resp = session.get(BRSAPI_URL, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise BRSAPIError(f"brsapi request failed: {e}")

    try:
        data = resp.json()
    except Exception:
        raise BRSAPIError("Invalid JSON response from brsapi")

    if "History" not in data or not data["History"]:
        return []

    rows = []

    current_year = datetime.now().year
    min_year = current_year - max_years

    for item in data["History"]:
        try:
            date = datetime.strptime(item["DEven"], "%Y%m%d").date()
        except Exception:
            continue

        if date.year < min_year:
            continue

        rows.append({
            "date": date,
            "open": int(item.get("PriceFirst", 0)),
            "high": int(item.get("PriceMax", 0)),
            "low": int(item.get("PriceMin", 0)),
            "close": int(item.get("PriceClose", 0)),
            "volume": int(item.get("Volume", 0)),
        })

    rows.sort(key=lambda x: x["date"])
    return rows
