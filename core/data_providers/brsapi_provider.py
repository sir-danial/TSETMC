import requests
from django.conf import settings
from django.core.cache import cache
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo


BASE_URL = "https://BrsApi.ir/Api/Tsetmc/Index.php"
ALL_SYMBOLS_URL = "https://BrsApi.ir/Api/Tsetmc/AllSymbols.php"
GOLD_CURRENCY_URL = "https://BrsApi.ir/Api/Market/Gold_Currency.php"

HEADERS = {
     "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 OPR/106.0.0.0",
        "Accept": "application/json, text/plain, */*"
}


TZ_TEHRAN = ZoneInfo("Asia/Tehran")


def _now_tehran() -> datetime:
    return datetime.now(TZ_TEHRAN)


def _in_window(start_hm: tuple[int, int], end_hm: tuple[int, int]) -> bool:
    """
    True if Tehran time is between [start, end] inclusive.
    """
    now = _now_tehran().time()
    start = dtime(start_hm[0], start_hm[1], 0)
    end = dtime(end_hm[0], end_hm[1], 0)
    return start <= now <= end


def _get_json(url: str, params: dict) -> object:
    res = requests.get(url, params=params, headers=HEADERS, timeout=10)
    res.raise_for_status()
    return res.json()


# =========================
# TSETMC Index (type=1/2/3)
# =========================
def fetch_index(type_id: int):
    """
    کش Redis:
    - فقط بین 09:01 تا 12:40 اجازه رفرش داریم
    - TTL = 5 دقیقه
    - خارج از بازه: آخرین کش را می‌دهیم و رفرش نمی‌کنیم
    """
    type_id = int(type_id)
    cache_key = f"brsapi:tsetmc:index:{type_id}"

    TTL_SECONDS = 10 * 60  # 10 min
    REFRESH_WINDOW = (9, 1), (12, 40)

    cached = cache.get(cache_key)
    if cached is not None:
        # اگر در بازه مجاز هستیم، می‌گذاریم TTL تصمیم بگیرد:
        # (با expire خود Redis)
        return cached

    # کش خالی است. اگر خارج از بازه هستیم، باز هم یک بار می‌گیریم تا صفحه خالی نماند
    # ولی با TTL بلندتر که تا فردا بماند.
    in_win = _in_window(*REFRESH_WINDOW)
    ttl = TTL_SECONDS if in_win else 24 * 60 * 60  # اگر خارج از ساعت بازار: 24h

    params = {"key": settings.BRS_API_KEY, "type": type_id}
    data = _get_json(BASE_URL, params=params)
    cache.set(cache_key, data, timeout=ttl)
    return data


# =========================
# Gold / Currency / Crypto
# =========================

def fetch_gold_currency():
    cache_key = "brsapi:market:gold_currency"
    TTL_SECONDS = 5 * 60

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    url = f"{GOLD_CURRENCY_URL}?key={settings.BRS_API_KEY}"

    res = requests.get(url, headers=HEADERS, timeout=10)
    res.raise_for_status()
    data = res.json()

    cache.set(cache_key, data, timeout=TTL_SECONDS)
    return data


# =========================
# AllSymbols (Live Market)
# =========================
def fetch_all_symbols(type_id: int = 1):
    """
    کش Redis:
    - رفرش فقط بین 09:01 تا 18:10
    - هر 6 دقیقه (TTL=6min)
    - خارج از بازه: آخرین کش را می‌دهیم و رفرش نمی‌کنیم
    """
    type_id = int(type_id)
    cache_key = f"brsapi:tsetmc:all_symbols:{type_id}"

    TTL_SECONDS = 6 * 60
    REFRESH_WINDOW = (9, 1), (18, 10)

    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    in_win = _in_window(*REFRESH_WINDOW)
    ttl = TTL_SECONDS if in_win else 24 * 60 * 60

    params = {"key": settings.BRS_API_KEY, "type": type_id}
    data = _get_json(ALL_SYMBOLS_URL, params=params)
    cache.set(cache_key, data, timeout=ttl)
    return data


def get_symbol_live_by_l18(symbol_l18: str, type_id: int = 1):
    """
    جستجوی نماد بر اساس l18 داخل دیتای AllSymbols (از Redis)
    """
    symbol_l18 = (symbol_l18 or "").strip()
    if not symbol_l18:
        return None

    data = fetch_all_symbols(type_id=type_id)
    if not isinstance(data, list):
        return None

    # match دقیق
    for row in data:
        if str(row.get("l18", "")).strip() == symbol_l18:
            return row

    # match با فاصله/نیم‌فاصله (ملایم)
    normalized = symbol_l18.replace("‌", "").replace(" ", "")
    for row in data:
        l18 = str(row.get("l18", "")).strip().replace("‌", "").replace(" ", "")
        if l18 == normalized:
            return row

    return None
