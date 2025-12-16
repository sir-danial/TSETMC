from django.urls import path
from . import views

urlpatterns = [
    path("", views.symbol_page, name="symbol_page"),
    path("api/analyze/", views.api_analyze, name="api_analyze"),
]
