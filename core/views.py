from __future__ import annotations

from typing import Any, Dict

from django.http import JsonResponse
from django.shortcuts import render

from core.service.analyzer import analyze_symbol


def symbol_page(request):
    symbol = (request.GET.get("symbol") or "").strip()
    return render(request, "core/symbol.html", {"symbol": symbol})


def api_analyze(request):
    symbol = (request.GET.get("symbol") or "").strip()
    data: Dict[str, Any] = analyze_symbol(symbol)

    # Ensure_ascii False for Persian, JsonResponse handles dict
    # IMPORTANT: do not pass numpy types; analyzer already converts.
    return JsonResponse(data, json_dumps_params={"ensure_ascii": False})
