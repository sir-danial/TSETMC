from django.contrib import admin
from .models import Symbol, DailyPrice


@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = ("ticker", "name")
    search_fields = ("ticker", "name")


@admin.register(DailyPrice)
class DailyPriceAdmin(admin.ModelAdmin):
    list_display = ("symbol", "date", "close", "volume")
    list_filter = ("symbol",)
    ordering = ("-date",)
