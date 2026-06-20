"""
Intraday market simulator for the optimal-execution environment.

Price paths come from real 1-hour OHLCV bars downloaded from Yahoo Finance (up to 2
years of history per ticker). Each episode replays an actual historical trading day:
the real hourly closes are used as anchor points and linearly interpolated to produce
n_slices+1 slice-boundary prices, so path[0] is the true open and path[-1] is the
true close of that day.

Market impact follows the standard Almgren-Chriss split: temporary (decays after the
slice) + permanent (persists for the rest of the episode).

TODO(market_sim): calibrate _TEMP_IMPACT_COEF / _PERM_IMPACT_COEF against realistic
magnitudes (impact should be small relative to the day's natural range for a "reasonable"
order size, and grow superlinearly for outsized orders).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_TICKERS = ["TSLA", "NVDA", "AAPL", "SPY"]
_START = "2022-01-01"
_END = "2024-12-31"
_DATA_DIR = Path(__file__).parent.parent / "data_cache"

# Almgren-Chriss-style impact coefficients -- TODO: calibrate, see module docstring.
_TEMP_IMPACT_COEF = 0.1
_PERM_IMPACT_COEF = 0.05


def load_intraday_data() -> dict[str, pd.DataFrame]:
    """Download and cache real 1-hour intraday bars per ticker (up to 2 years back).

    Each DataFrame is indexed by tz-naive datetime (America/New_York) with columns
    [Open, High, Low, Close, Volume]. Cached to data_cache/{ticker}_1h.parquet.
    """
    _DATA_DIR.mkdir(exist_ok=True)
    result = {}
    missing = []
    for ticker in _TICKERS:
        cache_path = _DATA_DIR / f"{ticker}_1h.parquet"
        if cache_path.exists():
            result[ticker] = pd.read_parquet(cache_path)
        else:
            missing.append(ticker)

    for ticker in missing:
        raw = yf.download(ticker, period="2y", interval="1h", auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw = raw.xs(ticker, axis=1, level=1)
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        if df.index.tzinfo is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
        df.dropna(inplace=True)
        df.to_parquet(_DATA_DIR / f"{ticker}_1h.parquet")
        result[ticker] = df

    return result


def get_intraday_path(day_bars: pd.DataFrame, n_slices: int) -> np.ndarray:
    """Build a real price path for one trading day from 1-hour bars.

    Uses the day's actual open as path[0] and each hourly close as an anchor,
    linearly interpolating to produce n_slices+1 evenly-spaced slice-boundary
    prices. path[0] = real open, path[-1] = real close of the last bar.
    """
    prices = np.concatenate([
        [float(day_bars["Open"].iloc[0])],
        day_bars["Close"].values.astype(float),
    ])
    x_anchors = np.linspace(0, n_slices, len(prices))
    x_target = np.arange(n_slices + 1, dtype=float)
    return np.interp(x_target, x_anchors, prices)


def u_shaped_volume_curve(n_slices: int) -> np.ndarray:
    """Returns normalized per-slice volume weights (sums to 1), U-shaped: heavier at
    open and close, lighter midday -- matches the well-documented real intraday volume
    profile even without real intraday data.
    """
    x = np.linspace(-1, 1, n_slices)
    weights = 0.6 + 0.4 * x**2  # TODO: tune curvature against published volume-profile shapes
    return weights / weights.sum()




@dataclass
class ImpactModel:
    adv: float  # average daily volume for this ticker, used to normalize order size

    def temporary_impact(self, qty: float) -> float:
        """Price impact (fraction of price) that affects only this slice's fill, then
        decays away. TODO: validate against literature magnitudes."""
        participation = qty / max(self.adv, 1.0)
        return _TEMP_IMPACT_COEF * np.sqrt(participation)

    def permanent_impact(self, qty: float) -> float:
        """Price impact (fraction of price) that persists for the rest of the episode."""
        participation = qty / max(self.adv, 1.0)
        return _PERM_IMPACT_COEF * participation


if __name__ == "__main__":
    data = load_intraday_data()
    for ticker, df in data.items():
        n_days = df.groupby(df.index.date).ngroups
        print(f"{ticker}: {len(df)} 1h bars across {n_days} trading days")
    print("U-shaped volume curve (10 slices):", u_shaped_volume_curve(10).round(3))
