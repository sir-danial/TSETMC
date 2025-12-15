from django.core.management.base import BaseCommand
from core.models import Symbol, DailyPrice
from core.data_providers.brapi import fetch_daily_history


class Command(BaseCommand):
    help = "Import daily historical prices for a symbol"

    def add_arguments(self, parser):
        parser.add_argument(
            "--ticker",
            type=str,
            required=True,
            help="Symbol ticker (e.g. فملی)",
        )
        parser.add_argument(
            "--years",
            type=int,
            default=10,
            help="Number of years to fetch",
        )

    def handle(self, *args, **options):
        ticker = options["ticker"]
        years = options["years"]

        self.stdout.write(
            self.style.WARNING(
                f"Fetching up to {years} years of daily data for '{ticker}' ..."
            )
        )

        try:
            symbol = Symbol.objects.get(ticker=ticker)
        except Symbol.DoesNotExist:
            self.stderr.write(
                self.style.ERROR(
                    f"Symbol '{ticker}' not found. Please create it in admin first."
                )
            )
            return

        try:
            rows = fetch_daily_history(symbol=ticker, max_years=years)
        except Exception as exc:
            self.stderr.write(
                self.style.ERROR(f"Data provider error: {exc}")
            )
            return

        if not rows:
            self.stderr.write(
                self.style.ERROR("No data received from data provider.")
            )
            return

        created = 0
        updated = 0

        for row in rows:
            obj, is_created = DailyPrice.objects.update_or_create(
                symbol=symbol,
                date=row["date"],
                defaults={
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                },
            )

            if is_created:
                created += 1
            else:
                updated += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Import finished for {ticker}: created={created}, updated={updated}"
            )
        )
