import finpy_tse as tse


def fetch_daily_history(symbol: str):
    """
    Fetch daily price history using finpy_tse
    """
    df = tse.Get_Price_History(
        stock=symbol,
        start_date="1390-01-01",
        end_date="1405-01-01",
        adjust_price=True,
    )

    if df is None or df.empty:
        return []

    df = df.reset_index()

    return df
