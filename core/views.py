from __future__ import annotations
from typing import Any, Dict

from django.http import JsonResponse
from django.shortcuts import render
from django.conf import settings

from core.service.analyzer import analyze_symbol
from core.data_providers.brsapi_provider import (
    fetch_index,
    fetch_gold_currency,
    get_symbol_live_by_l18,
)


# =====================
# Pages
# =====================
def home_page(request):
    return render(request, "core/home.html")


def symbol_page(request):
    symbol = (request.GET.get("symbol") or "").strip()
    return render(request, "core/symbol.html", {"symbol": symbol})


# =====================
# APIs
# =====================
def api_analyze(request):
    symbol = (request.GET.get("symbol") or "").strip()
    if not symbol:
        return JsonResponse(
            {"error": "symbol is required"},
            status=400,
            json_dumps_params={"ensure_ascii": False},
        )

    try:
        live = get_symbol_live_by_l18(symbol, type_id=1)
        data: Dict[str, Any] = analyze_symbol(symbol, live=live)
        return JsonResponse(data, json_dumps_params={"ensure_ascii": False})
    except Exception as e:
        # این خیلی مهمه: جلوی HTTP 500 خاموش رو می‌گیره
        return JsonResponse(
            {
                "error": "analyze_failed",
                "detail": str(e),
            },
            status=502,
            json_dumps_params={"ensure_ascii": False},
        )


def api_tsetmc_index(request):
    try:
        type_id = int(request.GET.get("type", 1))
        data = fetch_index(type_id)

        # شاخص بورس / فرابورس
        if type_id in (1, 2) and isinstance(data, dict):
            idx = data.get("index", 0) or 0
            chg = data.get("index_change", 0) or 0
            data["index_change_percent"] = round((chg / idx) * 100, 2) if idx else 0

        # شاخص‌های منتخب
        elif type_id == 3 and isinstance(data, list):
            for row in data:
                idx = row.get("index", 0) or 0
                chg = row.get("index_change", 0) or 0
                row["index_change_percent"] = round((chg / idx) * 100, 2) if idx else 0

        return JsonResponse(
            data,
            safe=(type_id != 3),
            json_dumps_params={"ensure_ascii": False},
        )

    except Exception as e:
        return JsonResponse(
            {
                "error": "index_fetch_failed",
                "detail": str(e),
            },
            status=502,
            json_dumps_params={"ensure_ascii": False},
        )


def api_market_gold_currency(request):
    """
    داده طلا/ارز/کریپتو از Redis (هر 5 دقیقه رفرش)
    """
    try:
        data = fetch_gold_currency()
        return JsonResponse(data, safe=False, json_dumps_params={"ensure_ascii": False})
    except Exception as e:
        return JsonResponse(
            {
                "error": "gold_currency_failed",
                "detail": str(e),
            },
            status=502,
            json_dumps_params={"ensure_ascii": False},
        )


def api_symbol_live(request):
    """
    اطلاعات لایو یک نماد از Redis (AllSymbols)
    """
    symbol = (request.GET.get("symbol") or "").strip()
    if not symbol:
        return JsonResponse(
            {"error": "symbol is required"},
            status=400,
            json_dumps_params={"ensure_ascii": False},
        )

    try:
        row = get_symbol_live_by_l18(symbol, type_id=1)
        if not row:
            return JsonResponse(
                {"error": "symbol not found in live cache"},
                status=404,
                json_dumps_params={"ensure_ascii": False},
            )
        return JsonResponse(row, json_dumps_params={"ensure_ascii": False})
    except Exception as e:
        return JsonResponse(
            {
                "error": "symbol_live_failed",
                "detail": str(e),
            },
            status=502,
            json_dumps_params={"ensure_ascii": False},
        )


# =====================
# Debug (TEMP)
# =====================
def api_debug_env(request):
    return JsonResponse(
        {
            "DEBUG": bool(settings.DEBUG),
            "BRS_API_KEY_is_set": bool(getattr(settings, "BRS_API_KEY", None)),
            "BRS_API_KEY_preview": (
                settings.BRS_API_KEY[:4] + "****"
                if getattr(settings, "BRS_API_KEY", None)
                else None
            ),
            "REDIS_CONFIGURED": bool(getattr(settings, "CACHES", None)),
            "REDIS_LOCATION": settings.CACHES.get("default", {}).get("LOCATION")
            if hasattr(settings, "CACHES")
            else None,
        },
        json_dumps_params={"ensure_ascii": False},
    )
