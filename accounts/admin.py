from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

from .models import UserProfile


class UserProfileInline(admin.StackedInline):
    model = UserProfile
    can_delete = False
    verbose_name = "پروفایل اشتراک"
    verbose_name_plural = "پروفایل اشتراک"
    fields = ("days_remaining", "last_check_date")
    readonly_fields = ("last_check_date",)


class UserAdmin(BaseUserAdmin):
    inlines = (UserProfileInline,)
    list_display = (
        "username",
        "is_active",
        "get_days_remaining",
        "date_joined",
    )
    list_filter = ("is_active", "is_staff")

    @admin.display(description="روز اعتبار", ordering="profile__days_remaining")
    def get_days_remaining(self, obj):
        try:
            return obj.profile.days_remaining
        except UserProfile.DoesNotExist:
            return "—"

    def save_formset(self, request, form, formset, change):
        super().save_formset(request, form, formset, change)
        if formset.model == UserProfile:
            UserProfile.objects.get_or_create(user=form.instance)


admin.site.unregister(User)
admin.site.register(User, UserAdmin)

admin.site.site_header = "پنل مدیریت داشبورد بازار"
admin.site.site_title = "پنل مدیریت"
admin.site.index_title = "مدیریت کاربران و اشتراک‌ها"
