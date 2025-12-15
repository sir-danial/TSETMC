import finpy_tse as tse

print("Starting TSETMC test...")

try:
    df = tse.Get_Price_History(
        stock='فملی',
        start_date='1401-01-01',
        end_date='1402-01-01',
        adjust_price=True
    )

    print("SUCCESS ✅")
    print("Rows:", len(df))
    print(df.head())

except Exception as e:
    print("FAILED ❌")
    print(e)
