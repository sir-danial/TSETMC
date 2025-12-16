# core/views.py
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET

from core.service.analyzer import analyze_symbol


@require_GET
def symbol_page(request):
    symbol = request.GET.get("symbol", "").strip()
    return render(request, "core/symbol.html", {"symbol": symbol})


@require_GET
def api_analyze(request):
    symbol = request.GET.get("symbol", "").strip()
    try:
        data = analyze_symbol(symbol)
    except Exception as e:
        data = {"error": f"analyzer failed: {str(e)}"}

    # همیشه 200 بده تا فرانت بتونه پیام خطا رو نشون بده
    return JsonResponse(data, status=200)
