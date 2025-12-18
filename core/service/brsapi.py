from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import requests

# کش ساده داخل حافظه پروسه (برای MVP)
_CACHE: Dict[Tuple[str, int], Tuple[float, Any]] = {}
CACHE_TTL_SECONDS = 15 * 60  # ۱۵ دقیقه

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 OPR/106.0.0.0",
    "Accept": "application/json, text/plain, */*",
}

def fetch_tsetmc_index(*, api_key: str, type_number: int) -> Any:
    """
    type:
      1 = شاخص بورس
      2 = شاخص فرابورس
      3 = شاخص‌های منتخب
    خروجی: JSON (ممکن است dict یا list باشد)
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return {"error": "BRSAPI_KEY is missing"}

    cache_key = (api_key, int(type_number))
    now = time.time()

    # cache hit
    if cache_key in _CACHE:
        ts, data = _CACHE[cache_key]
        if now - ts <= CACHE_TTL_SECONDS:
            return data

    url = f"https://BrsApi.ir/Api/Tsetmc/Index.php?key={api_key}&type={int(type_number)}"
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=20)
        if r.status_code != 200:
            data = {"error": f"upstream status {r.status_code}"}
            _CACHE[cache_key] = (now, data)
            return data

        data = r.json()
        _CACHE[cache_key] = (now, data)
        return data
    except Exception as e:
        data = {"error": f"request failed: {str(e)}"}
        _CACHE[cache_key] = (now, data)
        return data
