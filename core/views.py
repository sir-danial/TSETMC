from __future__ import annotations
from typing import Any, Dict

from django.http import JsonResponse
from django.shortcuts import render

from core.service.analyzer import analyze_symbol
from core.data_providers.brsapi_provider import (
    fetch_index,
    fetch_gold_currency,
    get_symbol_live_by_l18,
)


def home_page(request):
    return render(request, "core/home.html")


def symbol_page(request):
    symbol = (request.GET.get("symbol") or "").strip()
    return render(request, "core/symbol.html", {"symbol": symbol})


def api_analyze(request):
    symbol = (request.GET.get("symbol") or "").strip()

    live = get_symbol_live_by_l18(symbol, type_id=1)  # از Redis (AllSymbols)
    data: Dict[str, Any] = analyze_symbol(symbol, live=live)

    return JsonResponse(data, json_dumps_params={"ensure_ascii": False})


def api_tsetmc_index(request):
    type_id = int(request.GET.get("type", 1))
    data = fetch_index(type_id)  # اکنون کش‌شده با Redis است

    # شاخص بورس / فرابورس
    if type_id in (1, 2):
        idx = (data or {}).get("index", 0) or 0
        chg = (data or {}).get("index_change", 0) or 0
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


def api_market_gold_currency(request):
    """
    داده طلا/ارز/کریپتو از Redis (هر 5 دقیقه رفرش)
    """
    data = fetch_gold_currency()
    return JsonResponse(data, safe=False, json_dumps_params={"ensure_ascii": False})


def api_symbol_live(request):
    """
    اطلاعات لایو یک نماد از Redis (AllSymbols)
    """
    symbol = (request.GET.get("symbol") or "").strip()
    row = get_symbol_live_by_l18(symbol, type_id=1)
    if not row:
        return JsonResponse(
            {"error": "symbol not found in live cache"},
            status=404,
            json_dumps_params={"ensure_ascii": False},
        )

    return JsonResponse(row, json_dumps_params={"ensure_ascii": False})
