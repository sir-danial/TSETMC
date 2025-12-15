from django.shortcuts import render
from django.http import JsonResponse
from .service.analyzer import analyze_symbol


def symbol_page(request):
    symbol = request.GET.get("symbol")
    context = {}

    if symbol:
        context["symbol"] = symbol

    return render(request, "core/symbol.html", context)


def api_analyze(request):
    symbol = request.GET.get("symbol")
    if not symbol:
        return JsonResponse({"error": "symbol is required"}, status=400)

    result = analyze_symbol(symbol)
    return JsonResponse(result, json_dumps_params={"ensure_ascii": False})
