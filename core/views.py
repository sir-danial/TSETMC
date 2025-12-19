from __future__ import annotations
from typing import Any, Dict

from django.http import JsonResponse
from django.shortcuts import render

from core.service.analyzer import analyze_symbol
from core.data_providers.brsapi_provider import fetch_index


def home_page(request):
    return render(request, "core/home.html")


def symbol_page(request):
    symbol = (request.GET.get("symbol") or "").strip()
    return render(request, "core/symbol.html", {"symbol": symbol})


def api_analyze(request):
    symbol = (request.GET.get("symbol") or "").strip()
    data: Dict[str, Any] = analyze_symbol(symbol)
    return JsonResponse(data, json_dumps_params={"ensure_ascii": False})


def api_tsetmc_index(request):
    type_id = int(request.GET.get("type", 1))
    data = fetch_index(type_id)

    # شاخص بورس
    if type_id == 1:
        idx = data.get("index", 0)
        chg = data.get("index_change", 0)
        data["index_change_percent"] = round((chg / idx) * 100, 2) if idx else 0

    # شاخص فرابورس
    elif type_id == 2:
        idx = data.get("index", 0)
        chg = data.get("index_change", 0)
        data["index_change_percent"] = round((chg / idx) * 100, 2) if idx else 0

    # شاخص‌های منتخب
    elif type_id == 3:
        for row in data:
            idx = row.get("index", 0)
            chg = row.get("index_change", 0)
            row["index_change_percent"] = round((chg / idx) * 100, 2) if idx else 0

    return JsonResponse(
        data,
        safe=(type_id != 3),
        json_dumps_params={"ensure_ascii": False},
    )


from core.data_providers.brsapi_provider import fetch_gold_currency


def api_gold_currency(request):
    data = fetch_gold_currency()
    return JsonResponse(
        data,
        safe=False,
        json_dumps_params={"ensure_ascii": False},
    )
