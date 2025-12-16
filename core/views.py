from django.http import JsonResponse
from django.shortcuts import render
from core.service.analyzer import analyze_symbol


def symbol_page(request):
    """
    صفحه اصلی فرانت (فرم + خروجی)
    """
    return render(request, "core/symbol.html")


def api_analyze(request):
    """
    API تحلیل نماد
    مثال:
    /api/analyze/?symbol=فملی
    """
    symbol = request.GET.get("symbol")

    if not symbol:
        return JsonResponse({"error": "symbol is required"}, status=400)

    result = analyze_symbol(symbol)
    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})
