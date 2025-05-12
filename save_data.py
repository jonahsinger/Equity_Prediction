import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ────────────────────────────────
TICKER = "GOOG"
START  = "2000-01-01"
END    = "2025-01-01"
# ────────────────────────────────


def week_return_dataset(ticker: str,
                        start: str,
                        end: str,
                        extend: int = 40) -> pd.DataFrame:
    """
    Feature matrix for predicting a stock’s 1-week (5-day) return.

    Returned columns (all features are lagged one trading day):
        ret_1d             – 1-day return
        ret_1w             – trailing 1-week return
        ret_1m             – trailing 1-month return
        volume_rel_1d      – yesterday’s volume ÷ 5-day avg
        volume_rel_1w      – 5-day avg volume ÷ 20-day avg
        volatility_ratio   – realised σ (5 d) ÷ realised σ (20 d)
        ticker             – constant string identifying the symbol
        y                  – **target**: next-week return
    """
    # ── resolve dates & warm-up window ─────────────────────────────
    start_dt = pd.to_datetime(start)
    end_dt   = pd.to_datetime(end)
    fetch_from = start_dt - timedelta(days=extend)

    df = yf.download(
        ticker,
        start=fetch_from.strftime("%Y-%m-%d"),
        end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        progress=False
    )
    if df.empty:
        raise ValueError("No data returned – check ticker or dates.")

    # choose adjusted or raw close
    price = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]

    # ── price returns ──────────────────────────────────────────────
    df["ret_1d"] = price.pct_change()
    df["ret_1w"] = price.pct_change(5)
    df["ret_1m"] = price.pct_change(20)

    # ── volume relatives ──────────────────────────────────────────
    df["volume_rel_1d"] = df["Volume"] / df["Volume"].rolling(5).mean()
    df["volume_rel_1w"] = (
        df["Volume"].rolling(5).mean() / df["Volume"].rolling(20).mean()
    )

    # ── volatility ratio ──────────────────────────────────────────
    rv_1w = df["ret_1d"].rolling(5).std()
    rv_1m = df["ret_1d"].rolling(20).std()
    df["volatility_ratio"] = rv_1w / rv_1m

    # ── prediction target ─────────────────────────────────────────
    df["y"] = price.pct_change(5).shift(-5)

    # ── assemble final DataFrame inside requested window ─────────
    feat_cols = [
        "ret_1d", "ret_1w", "ret_1m",
        "volume_rel_1d", "volume_rel_1w", "volatility_ratio"
    ]
    window = df.loc[start_dt:end_dt]

    data = pd.concat([window[feat_cols].shift(1), window["y"]], axis=1).dropna()

    # flatten any 1-element tuple column names that pandas created
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    # add constant ticker column for easy multi-asset concatenation later
    data.insert(0, "ticker", ticker)

    return data


def main():
    df = week_return_dataset(TICKER, START, END)
    print(f"\nFeature set for {TICKER}")
    print(df.head(), f"\n\nRows: {len(df)}")
    df.to_csv("week_features.csv", index_label="date")


if __name__ == "__main__":
    main()
