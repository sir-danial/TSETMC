import requests
from django.conf import settings

BASE_URL = "https://BrsApi.ir/Api/Tsetmc/Index.php"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def fetch_index(type_id: int):
    params = {
        "key": settings.BRS_API_KEY,
        "type": type_id
    }

    res = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=10)
    res.raise_for_status()
    return res.json()