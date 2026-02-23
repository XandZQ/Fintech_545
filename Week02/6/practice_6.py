import numpy as np
import pandas as pd


def return_calculate(prices: pd.DataFrame, method: str = "ARITH", date_column: str = "Date") -> pd.DataFrame:
    data = prices.copy()
    if date_column not in data.columns:
        raise ValueError("Date column not found.")

    dates = data[date_column]
    px = data.drop(columns=[date_column]).astype(float)

    if method.upper() == "LOG":
        rets = np.log(px / px.shift(1))
    else:
        rets = px.pct_change()

    out = pd.concat([dates, rets], axis=1)
    out = out.iloc[1:].reset_index(drop=True)
    return out


def main():
    prices = pd.read_csv("test6.csv")

    # 6.1 Arithmetic returns
    rout = return_calculate(prices, method="ARITH", date_column="Date")
    rout.to_csv("testout_6.1.csv", index=False)

    # 6.2 Log returns
    rout = return_calculate(prices, method="LOG", date_column="Date")
    rout.to_csv("testout_6.2.csv", index=False)


if __name__ == "__main__":
    main()
