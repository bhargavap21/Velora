"""
Synthetic intraday market simulator for the optimal-execution environment.

We don't have real historical order-book or intraday tick data for 2022-2024, so the
price path is synthetic: a Brownian bridge between the day's real (open, close), scaled
to the day's real (high, low) range, calibrated from cached daily OHLCV. Market impact
follows the standard Almgren-Chriss split: a temporary component that only affects the
current slice's execution price, and a permanent component that persists for the rest
of the episode. This is the standard simplification used in execution-RL research when
L2 data isn't available -- not a shortcut unique to this project.

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


def load_daily_data() -> dict[str, pd.DataFrame]:
    """Loads cached daily OHLCV per ticker (downloads + caches on first run).

    Mirrors ../../backtester.py's _load_data, duplicated here so this module has no
    dependency on the legacy StratRL code.
    """
    _DATA_DIR.mkdir(exist_ok=True)
    result = {}
    missing = []
    for ticker in _TICKERS:
        path = _DATA_DIR / f"{ticker}.parquet"
        if path.exists():
            result[ticker] = pd.read_parquet(path)
        else:
            missing.append(ticker)

    if missing:
        raw = yf.download(missing, start=_START, end=_END, auto_adjust=True, progress=False)
        for ticker in missing:
            df = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df.dropna(inplace=True)
            df.to_parquet(_DATA_DIR / f"{ticker}.parquet")
            result[ticker] = df

    return result


def u_shaped_volume_curve(n_slices: int) -> np.ndarray:
    """Returns normalized per-slice volume weights (sums to 1), U-shaped: heavier at
    open and close, lighter midday -- matches the well-documented real intraday volume
    profile even without real intraday data.
    """
    x = np.linspace(-1, 1, n_slices)
    weights = 0.6 + 0.4 * x**2  # TODO: tune curvature against published volume-profile shapes
    return weights / weights.sum()


def generate_intraday_path(day_row: pd.Series, n_slices: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic intraday price path of length n_slices+1 (slice boundaries), calibrated
    to the real day's (Open, High, Low, Close).

    TODO(market_sim): current version is a placeholder linear interpolation + noise.
    Replace with a proper Brownian bridge clipped to [Low, High] so the path's realized
    volatility matches the day's actual range.
    """
    open_, high, low, close = day_row["Open"], day_row["High"], day_row["Low"], day_row["Close"]
    base = np.linspace(open_, close, n_slices + 1)
    noise_scale = (high - low) * 0.1
    noise = rng.normal(0, noise_scale, size=n_slices + 1)
    noise[0] = 0.0
    path = np.clip(base + noise, low, high)
    path[0] = open_
    path[-1] = close
    return path


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
    data = load_daily_data()
    print({k: len(v) for k, v in data.items()})
    print("U-shaped volume curve (10 slices):", u_shaped_volume_curve(10).round(3))
