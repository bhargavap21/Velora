"""
Intraday market simulator for the optimal-execution environment.

Daily and per-slice OHLCV both come from Alpaca's SIP consolidated feed when credentials
are configured (see load_daily_data/load_minute_data), falling back to yfinance (daily)
or a synthetic path (intraday) otherwise. For days where real minute bars aren't cached
or returned empty (e.g. outside Alpaca's history, or no credentials configured), we fall
back to a synthetic path: a Brownian bridge between the day's real (open, close), scaled
to the day's real (high, low) range via the Parkinson volatility estimator, calibrated
from cached daily OHLCV. Market impact
follows the standard Almgren-Chriss split: a temporary component that only affects the
current slice's execution price, and a permanent component that persists for the rest
of the episode. This is the standard simplification used in execution-RL research when
L2 (order-book) data isn't available -- not a shortcut unique to this project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

_TICKERS = ["TSLA", "NVDA", "AAPL", "SPY"]
_START = "2022-01-01"
_END = "2024-12-31"
_DATA_DIR = Path(__file__).parent.parent / "data_cache"
_SESSION_START = "09:30"
_SESSION_END = "16:00"

# Almgren-Chriss-style impact coefficients, calibrated so a "reasonable" order (~1% ADV)
# costs single-digit bps -- in line with published market-impact magnitudes -- while the
# convexity multiplier below makes outsized orders (tens of % of ADV) blow up superlinearly.
_TEMP_IMPACT_COEF = 0.002
_PERM_IMPACT_COEF = 0.01
_OUTSIZED_CONVEXITY = 20.0


def load_daily_data() -> dict[str, pd.DataFrame]:
    """Loads cached daily OHLCV per ticker (downloads + caches on first run).

    Prefers Alpaca's SIP-consolidated daily bars -- the same feed/source as
    load_minute_data -- so the fields that drive day-sampling and ADV-based impact
    sizing are consistent with the intraday minute bars used for the price path.
    Falls back to yfinance when ALPACA_API_KEY/ALPACA_SECRET_KEY aren't configured.
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
        fetched = _load_daily_from_alpaca(missing)
        for ticker, df in fetched.items():
            df.to_parquet(_DATA_DIR / f"{ticker}.parquet")
            result[ticker] = df

        still_missing = [t for t in missing if t not in fetched]
        if still_missing:
            raw = yf.download(still_missing, start=_START, end=_END, auto_adjust=True, progress=False)
            for ticker in still_missing:
                df = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df.dropna(inplace=True)
                df.to_parquet(_DATA_DIR / f"{ticker}.parquet")
                result[ticker] = df

    return result


def _load_daily_from_alpaca(tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Fetches daily OHLCV for `tickers` from Alpaca's SIP feed in one batched request.
    Returns {} (not a partial result) if ALPACA_API_KEY/ALPACA_SECRET_KEY aren't
    configured, so the caller falls back to yfinance for all of them.
    """
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return {}

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=tickers, timeframe=TimeFrame.Day, start=_START, end=_END, feed=DataFeed.SIP
    )
    bars = client.get_stock_bars(request).df

    result = {}
    for ticker in tickers:
        df = bars.xs(ticker, level="symbol")[["open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df.index.date)
        result[ticker] = df
    return result


def load_minute_data(ticker: str) -> pd.DataFrame | None:
    """Loads cached real minute-bar OHLCV for `ticker` (downloads + caches on first run),
    restricted to regular trading hours (09:30-16:00 ET). Returns None if no cache exists
    and ALPACA_API_KEY/ALPACA_SECRET_KEY aren't configured, so callers can fall back to
    the synthetic path instead of crashing.

    Uses the SIP feed (consolidated tape across all exchanges) rather than IEX-only, for
    closer-to-real fill prices and volume.
    """
    path = _DATA_DIR / f"{ticker}_minute.parquet"
    if path.exists():
        return pd.read_parquet(path)

    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        return None

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=[ticker], timeframe=TimeFrame.Minute, start=_START, end=_END, feed=DataFeed.SIP
    )
    bars = client.get_stock_bars(request).df
    df = bars.xs(ticker, level="symbol")[["open", "high", "low", "close", "volume"]].copy()
    df.index = df.index.tz_convert("America/New_York")
    df = df.between_time(_SESSION_START, _SESSION_END)

    _DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(path)
    return df


def real_intraday_path(minute_df: pd.DataFrame, date: pd.Timestamp, n_slices: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Builds a (price_path, volume_curve) pair for `date` from real minute bars: prices
    at n_slices+1 evenly-spaced slice boundaries within the trading session, and volume
    summed within each slice.

    Returns None if there's no real data for this day within tolerance (e.g. holiday,
    early close, gap in history) -- a partial real day is worse than a fully synthetic
    one, so callers should fall back entirely to generate_intraday_path in that case.
    """
    day_df = minute_df[minute_df.index.date == date.date()]
    if day_df.empty:
        return None

    session_start = day_df.index[0].normalize() + pd.Timedelta(_SESSION_START + ":00")
    session_end = day_df.index[0].normalize() + pd.Timedelta(_SESSION_END + ":00")
    boundaries = pd.date_range(session_start, session_end, periods=n_slices + 1)

    positions = day_df.index.get_indexer(boundaries, method="nearest", tolerance=pd.Timedelta(minutes=2))
    if (positions == -1).any():
        return None
    prices = day_df["close"].iloc[positions].to_numpy()

    volumes = np.zeros(n_slices)
    for i in range(n_slices):
        mask = (day_df.index >= boundaries[i]) & (day_df.index < boundaries[i + 1])
        volumes[i] = day_df.loc[mask, "volume"].sum()
    if volumes.sum() <= 0:
        return None
    volume_curve = volumes / volumes.sum()

    return prices, volume_curve


def intraday_path_and_volume(
    ticker: str,
    date: pd.Timestamp,
    day_row: pd.Series,
    n_slices: int,
    rng: np.random.Generator,
    minute_data: dict[str, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Real-data-first version of (generate_intraday_path, u_shaped_volume_curve): uses
    real minute bars for `ticker`/`date` when available, else falls back to the synthetic
    path and the synthetic U-shaped volume curve.
    """
    minute_df = (minute_data or {}).get(ticker)
    if minute_df is not None:
        real = real_intraday_path(minute_df, date, n_slices)
        if real is not None:
            return real

    return generate_intraday_path(day_row, n_slices, rng), u_shaped_volume_curve(n_slices)


def u_shaped_volume_curve(n_slices: int) -> np.ndarray:
    """Returns normalized per-slice volume weights (sums to 1), U-shaped: heavier at
    open and close, lighter midday -- matches the well-documented real intraday volume
    profile even without real intraday data.

    Curvature tuned so edge slices carry ~2.86x the volume of the midday slice, in line
    with published intraday volume-profile studies showing the first/last ~30min carrying
    roughly 2-3x the midday rate.
    """
    x = np.linspace(-1, 1, n_slices)
    weights = 0.35 + 0.65 * x**2
    return weights / weights.sum()


def generate_intraday_path(day_row: pd.Series, n_slices: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic intraday price path of length n_slices+1 (slice boundaries): a Brownian
    bridge pinned to the real day's (Open, Close) and clipped to the real day's
    (Low, High).

    The bridge's per-slice volatility is derived from the Parkinson estimator (a standard
    OHLC-based volatility estimator using only High/Low), so the path's realized
    volatility is anchored to the day's actual range rather than picked arbitrarily.
    """
    open_, high, low, close = day_row["Open"], day_row["High"], day_row["Low"], day_row["Close"]

    sigma_day = np.sqrt(np.log(high / low) ** 2 / (4 * np.log(2))) if low > 0 else 0.0
    sigma_slice = sigma_day / np.sqrt(n_slices)

    increments = rng.normal(0, sigma_slice * open_, size=n_slices)
    walk = np.concatenate([[0.0], np.cumsum(increments)])
    t = np.arange(n_slices + 1) / n_slices
    bridge = walk - t * walk[-1]  # pinned to 0 at both endpoints

    linear_drift = np.linspace(open_, close, n_slices + 1)
    path = np.clip(linear_drift + bridge, low, high)
    path[0] = open_
    path[-1] = close
    return path


def _convexity_multiplier(participation: float) -> float:
    """Multiplier applied on top of the base impact law. Negligible for normal-sized
    orders (~1-10% ADV) but grows superlinearly for outsized orders, reflecting that
    very large orders can't be executed without disproportionate price concessions.
    """
    return 1.0 + _OUTSIZED_CONVEXITY * participation**2


@dataclass
class ImpactModel:
    adv: float  # average daily volume for this ticker, used to normalize order size

    def temporary_impact(self, qty: float) -> float:
        """Price impact (fraction of price) that affects only this slice's fill, then
        decays away. Follows the standard square-root law for normal order sizes, with
        a convexity multiplier that makes outsized orders blow up superlinearly.
        """
        participation = qty / max(self.adv, 1.0)
        return _TEMP_IMPACT_COEF * np.sqrt(participation) * _convexity_multiplier(participation)

    def permanent_impact(self, qty: float) -> float:
        """Price impact (fraction of price) that persists for the rest of the episode."""
        participation = qty / max(self.adv, 1.0)
        return _PERM_IMPACT_COEF * participation * _convexity_multiplier(participation)


if __name__ == "__main__":
    data = load_daily_data()
    print({k: len(v) for k, v in data.items()})
    print("U-shaped volume curve (10 slices):", u_shaped_volume_curve(10).round(3))
