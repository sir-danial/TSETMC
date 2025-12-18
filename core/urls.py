from django.urls import path
from . import views

urlpatterns = [
    path("", views.home_page),
    path("api/tsetmc/index/", views.api_tsetmc_index),
    path("symbol/", views.symbol_page),
    path("api/analyze/", views.api_analyze),
]
