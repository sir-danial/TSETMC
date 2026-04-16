from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    days_remaining = models.IntegerField("روزهای اعتبار باقی‌مانده", default=0)
    last_check_date = models.DateField("آخرین بررسی اعتبار", null=True, blank=True)

    class Meta:
        verbose_name = "پروفایل کاربر"
        verbose_name_plural = "پروفایل کاربران"

    def __str__(self):
        return f"{self.user.username} — {self.days_remaining} روز"

    @property
    def is_subscription_active(self):
        """Check if user still has days remaining after consuming today."""
        self._consume_today()
        return self.days_remaining > 0

    def _consume_today(self):
        """Decrement one day if we haven't already checked today."""
        today = timezone.localdate()
        if self.last_check_date == today:
            return
        if self.days_remaining > 0:
            self.days_remaining -= 1
        self.last_check_date = today
        self.save(update_fields=["days_remaining", "last_check_date"])

    def add_days(self, count):
        self.days_remaining += count
        self.save(update_fields=["days_remaining"])
