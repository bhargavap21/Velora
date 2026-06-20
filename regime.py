"""
Lightweight market regime classifier — trending / mean_reverting / choppy.

Reuses backtester's cached parquet price data so this stays fast (<10ms) and
deterministic. Classification is based on a trailing window of daily closes:

  - trending: a strong, consistent directional move (high R^2 on a linear fit,
    large normalized slope)
  - mean_reverting: price oscillates around its window mean with frequent
    crossings and a weak linear trend
  - choppy: catch-all — meaningful volatility without a clear trend or a
    clean oscillation pattern
"""

from __future__ import annotations

import numpy as np

from backtester import _DATA

_WINDOW = 60
_TREND_SLOPE_THRESHOLD = 0.08   # total normalized move over the window
_TREND_R2_THRESHOLD = 0.4
_REVERT_CROSSING_RATE_THRESHOLD = 0.25
_REVERT_R2_THRESHOLD = 0.3


def classify_regime(ticker: str, window: int = _WINDOW) -> str:
    """Classify the trailing `window` days of `ticker` as trending/mean_reverting/choppy.

    Returns "unknown" if there isn't enough cached history for the ticker.
    """
    df = _DATA.get(ticker)
    if df is None or len(df) < window:
        return "unknown"

    closes = df["Close"].iloc[-window:].to_numpy(dtype=float)
    x = np.arange(len(closes), dtype=float)

    slope, intercept = np.polyfit(x, closes, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((closes - pred) ** 2))
    ss_tot = float(np.sum((closes - closes.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    normalized_slope = slope * len(closes) / closes.mean()

    mean = closes.mean()
    signs = np.sign(closes - mean)
    crossings = int(np.sum(np.diff(signs) != 0))
    crossing_rate = crossings / len(closes)

    if abs(normalized_slope) > _TREND_SLOPE_THRESHOLD and r2 > _TREND_R2_THRESHOLD:
        return "trending"
    if crossing_rate > _REVERT_CROSSING_RATE_THRESHOLD and r2 < _REVERT_R2_THRESHOLD:
        return "mean_reverting"
    return "choppy"


if __name__ == "__main__":
    from backtester import get_available_tickers

    print(f"Regime classification (trailing {_WINDOW} days, as of end of cached data):\n")
    for ticker in get_available_tickers():
        regime = classify_regime(ticker)
        print(f"  {ticker:6s} -> {regime}")

    print("\nRolling check — regime over time for TSLA (90-day windows ending every 30 trading days):")
    df = _DATA["TSLA"]
    for end in range(_WINDOW, len(df), 30):
        window_df = df.iloc[max(0, end - _WINDOW):end]
        closes = window_df["Close"].to_numpy(dtype=float)
        x = np.arange(len(closes), dtype=float)
        slope, intercept = np.polyfit(x, closes, 1)
        pred = slope * x + intercept
        r2 = 1 - np.sum((closes - pred) ** 2) / np.sum((closes - closes.mean()) ** 2)
        date = window_df.index[-1].date()
        # Reuse classify_regime logic by faking a truncated _DATA lookup window
        normalized_slope = slope * len(closes) / closes.mean()
        mean = closes.mean()
        crossings = int(np.sum(np.diff(np.sign(closes - mean)) != 0))
        crossing_rate = crossings / len(closes)
        if abs(normalized_slope) > _TREND_SLOPE_THRESHOLD and r2 > _TREND_R2_THRESHOLD:
            label = "trending"
        elif crossing_rate > _REVERT_CROSSING_RATE_THRESHOLD and r2 < _REVERT_R2_THRESHOLD:
            label = "mean_reverting"
        else:
            label = "choppy"
        print(f"  {date}  slope_norm={normalized_slope:+.3f}  r2={r2:.2f}  crossing_rate={crossing_rate:.2f}  -> {label}")
