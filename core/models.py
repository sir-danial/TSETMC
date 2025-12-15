from django.db import models


class Symbol(models.Model):
    ticker = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=200, blank=True)

    def __str__(self):
        return self.ticker


class DailyPrice(models.Model):
    symbol = models.ForeignKey(
        Symbol,
        on_delete=models.CASCADE,
        related_name="prices",
    )
    date = models.DateField()

    open = models.FloatField(null=True, blank=True)
    high = models.FloatField(null=True, blank=True)
    low = models.FloatField(null=True, blank=True)
    close = models.FloatField(null=True, blank=True)
    volume = models.BigIntegerField(null=True, blank=True)

    class Meta:
        unique_together = ("symbol", "date")
        ordering = ["date"]

    def __str__(self):
        return f"{self.symbol.ticker} - {self.date}"
