from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.shortcuts import render, redirect

from .models import UserProfile


def login_view(request):
    if request.user.is_authenticated:
        return redirect("/")

    error = None
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            next_url = request.GET.get("next", "/")
            return redirect(next_url)
        else:
            error = "نام کاربری یا رمز عبور اشتباه است."

    return render(request, "accounts/login.html", {"error": error})


def register_view(request):
    if request.user.is_authenticated:
        return redirect("/")

    error = None
    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "")
        password2 = request.POST.get("password2", "")

        if not username or not password:
            error = "نام کاربری و رمز عبور الزامی است."
        elif len(password) < 6:
            error = "رمز عبور باید حداقل ۶ کاراکتر باشد."
        elif password != password2:
            error = "رمز عبور و تکرار آن یکسان نیست."
        elif User.objects.filter(username=username).exists():
            error = "این نام کاربری قبلاً ثبت شده است."
        else:
            user = User.objects.create_user(username=username, password=password)
            UserProfile.objects.create(user=user, days_remaining=0)
            login(request, user)
            return redirect("/")

    return render(request, "accounts/register.html", {"error": error})


def logout_view(request):
    logout(request)
    return redirect("/")


def account_view(request):
    if not request.user.is_authenticated:
        return redirect("/accounts/login/")

    profile, _ = UserProfile.objects.get_or_create(user=request.user)
    return render(request, "accounts/account.html", {"profile": profile})
