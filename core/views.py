from django.http import JsonResponse
from django.shortcuts import render

from core.service.analyzer import analyze_symbol


def home(request):
    """
    صفحه اصلی سایت: همون فرانت (فرم وارد کردن نماد + نمایش نتیجه)
    """
    return render(request, "core/symbol.html")


def api_analyze(request):
    symbol = request.GET.get("symbol", "").strip()
    if not symbol:
        return JsonResponse({"error": "symbol is required"}, status=400)

    data = analyze_symbol(symbol)

    # نکته مهم برای جلوگیری از خطای JSON (numpy/bool و ...)
    def _safe(v):
        if isinstance(v, bool):
            return bool(v)
        if isinstance(v, (int, float, str)) or v is None:
            return v
        try:
            # numpy types
            import numpy as np
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                return float(v)
            if isinstance(v, (np.bool_,)):
                return bool(v)
        except Exception:
            pass
        # fallback
        return str(v)

    safe_data = {k: _safe(v) for k, v in (data or {}).items()}

    return JsonResponse(safe_data, json_dumps_params={"ensure_ascii": False})
