from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_page),

    # داشبورد (index types)
    path("api/tsetmc/index/", views.api_tsetmc_index),

    # طلا/ارز/کریپتو (کش‌شده)
    path("api/market/gold-currency/", views.api_market_gold_currency),

    # صفحه تحلیل نماد
    path("symbol/", views.symbol_page),
    path("api/analyze/", views.api_analyze),

    # لایو نماد (از AllSymbols و Redis)
    path("api/tsetmc/symbol-live/", views.api_symbol_live),
]
