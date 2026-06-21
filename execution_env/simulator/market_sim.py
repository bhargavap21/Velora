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
_DATA_DIR = Path(__file__).parent.parent / "data_cache"
_SESSION_START = "09:30"
_SESSION_END = "16:00"


def _data_end() -> str:
    """Last calendar day to request from data providers (yesterday ET, avoids partial today)."""
    today = pd.Timestamp.now(tz="America/New_York").normalize()
    return (today - pd.Timedelta(days=1)).strftime("%Y-%m-%d")


# Back-compat alias for config/docs; live fetches use _data_end().
_END = _data_end()


def _alpaca_credentials() -> tuple[str, str] | None:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if api_key and secret_key:
        return api_key, secret_key
    return None


def _daily_cache_stale(df: pd.DataFrame) -> bool:
    """True when cached daily bars are more than a few days behind _data_end()."""
    target = pd.Timestamp(_data_end()).normalize()
    return df.index.max().normalize() < target - pd.Timedelta(days=5)

# Almgren-Chriss-style impact coefficients. Impact is driven by the *participation rate*
# -- the order's share of the volume actually trading in that slice (qty / slice_volume) --
# NOT the order's share of the whole day's ADV. This is the standard market-microstructure
# model and the reason intraday *scheduling* matters: trading the same number of shares in a
# thin midday slice consumes a far larger fraction of that slice's liquidity (high
# participation -> high impact) than trading it into the heavy open/close. A naive
# equal-time schedule over-trades the thin slices and pays for it; a volume-aware schedule
# holds participation roughly constant and minimizes total impact (the classic result).
#
# Calibrated so a ~10% participation rate costs ~10 bps of temporary impact, with the
# convexity multiplier making very high participation (consuming most of a slice's volume)
# blow up superlinearly. NOTE: the coefficient is small because `participation` is now the
# per-slice rate (qty / slice_volume), not the share of whole-day ADV -- so 10% here means
# consuming 10% of a single slice's liquidity. (0.003 * sqrt(0.10) ~ 9.8 bps.)
_TEMP_IMPACT_COEF = 0.003
_PERM_IMPACT_COEF = 0.010
_OUTSIZED_CONVEXITY = 3.0


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
            df = pd.read_parquet(path)
            if _daily_cache_stale(df):
                refreshed = _load_daily_from_alpaca([ticker])
                if ticker in refreshed:
                    df = refreshed[ticker]
                    df.to_parquet(path)
            result[ticker] = df
        else:
            missing.append(ticker)

    if missing:
        fetched = _load_daily_from_alpaca(missing)
        for ticker, df in fetched.items():
            df.to_parquet(_DATA_DIR / f"{ticker}.parquet")
            result[ticker] = df

        still_missing = [t for t in missing if t not in fetched]
        if still_missing:
            raw = yf.download(still_missing, start=_START, end=_data_end(), auto_adjust=True, progress=False)
            for ticker in still_missing:
                df = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.index = pd.to_datetime(df.index)
                df.dropna(inplace=True)
                df.to_parquet(_DATA_DIR / f"{ticker}.parquet")
                result[ticker] = df

    return result


def _load_daily_from_alpaca(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetches daily OHLCV for `tickers` from Alpaca's SIP feed in one batched request.
    Returns {} (not a partial result) if ALPACA_API_KEY/ALPACA_SECRET_KEY aren't
    configured, so the caller falls back to yfinance for all of them.
    """
    creds = _alpaca_credentials()
    if creds is None:
        return {}

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key, secret_key = creds
    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start or _START,
        end=end or _data_end(),
        feed=DataFeed.SIP,
    )
    bars = client.get_stock_bars(request).df

    result = {}
    for ticker in tickers:
        df = bars.xs(ticker, level="symbol")[["open", "high", "low", "close", "volume"]].copy()
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df.index.date)
        result[ticker] = df
    return result


def ensure_daily_date(ticker: str, date: str) -> pd.DataFrame:
    """Return daily OHLCV for `ticker`, refreshing Alpaca SIP cache if `date` is missing."""
    data = load_daily_data()
    df = data[ticker]
    target = pd.Timestamp(date).normalize()
    if not df.index[df.index.normalize() == target].empty:
        return df

    refreshed = _load_daily_from_alpaca([ticker])
    if ticker in refreshed:
        df = refreshed[ticker]
        (_DATA_DIR / f"{ticker}.parquet").parent.mkdir(exist_ok=True)
        df.to_parquet(_DATA_DIR / f"{ticker}.parquet")
        if not df.index[df.index.normalize() == target].empty:
            return df

    raise ValueError(f"No data for {ticker} on {date}")


def _fetch_minute_bars_alpaca(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch minute SIP bars for `ticker` between start/end (YYYY-MM-DD), session hours only."""
    creds = _alpaca_credentials()
    if creds is None:
        return None

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    # Alpaca end is exclusive — bump by one day when requesting a single session.
    end_ts = pd.Timestamp(end)
    start_ts = pd.Timestamp(start)
    if end_ts <= start_ts:
        end_ts = start_ts + pd.Timedelta(days=1)
    elif start == end:
        end_ts = start_ts + pd.Timedelta(days=1)

    api_key, secret_key = creds
    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=[ticker],
        timeframe=TimeFrame.Minute,
        start=start,
        end=end_ts.strftime("%Y-%m-%d"),
        feed=DataFeed.SIP,
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        return None

    df = bars.xs(ticker, level="symbol")[["open", "high", "low", "close", "volume"]].copy()
    df.index = df.index.tz_convert("America/New_York")
    return df.between_time(_SESSION_START, _SESSION_END)


def ensure_minute_day(ticker: str, date: pd.Timestamp) -> pd.DataFrame | None:
    """Ensure minute SIP bars for `date` are in cache; fetch that day on demand if missing."""
    path = _DATA_DIR / f"{ticker}_minute.parquet"
    existing = pd.read_parquet(path) if path.exists() else None

    if existing is not None and not existing[existing.index.date == date.date()].empty:
        return existing

    day_str = date.strftime("%Y-%m-%d")
    fetched = _fetch_minute_bars_alpaca(ticker, day_str, day_str)
    if fetched is None or fetched.empty:
        return existing

    if existing is not None:
        merged = pd.concat([existing, fetched]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = fetched

    _DATA_DIR.mkdir(exist_ok=True)
    merged.to_parquet(path)
    return merged


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

    if _alpaca_credentials() is None:
        return None

    df = _fetch_minute_bars_alpaca(ticker, _START, _data_end())
    if df is None or df.empty:
        return None

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
) -> tuple[np.ndarray, np.ndarray, str]:
    """Real-data-first version of (generate_intraday_path, u_shaped_volume_curve): uses
    real minute bars for `ticker`/`date` when available, else falls back to the synthetic
    path and the synthetic U-shaped volume curve. Returns (path, volume_curve, data_source).
    """
    minute_df = (minute_data or {}).get(ticker)
    if minute_df is not None:
        real = real_intraday_path(minute_df, date, n_slices)
        if real is not None:
            return real[0], real[1], "real_minute"

    minute_df = ensure_minute_day(ticker, date)
    if minute_df is not None:
        real = real_intraday_path(minute_df, date, n_slices)
        if real is not None:
            return real[0], real[1], "real_minute"

    return generate_intraday_path(day_row, n_slices, rng), u_shaped_volume_curve(n_slices), "synthetic"


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
    """Multiplier applied on top of the base impact law. Negligible at low participation
    rates but grows superlinearly as the order consumes a large fraction of a slice's
    volume, reflecting that you can't sweep most of a slice's liquidity without
    disproportionate price concessions.
    """
    return 1.0 + _OUTSIZED_CONVEXITY * participation**2


@dataclass
class ImpactModel:
    adv: float  # average daily volume for this ticker; floor for slice volume

    def _participation(self, qty: float, slice_volume: float | None) -> float:
        """Order's share of the volume actually trading in this slice. Falls back to a
        share of (ADV / typical-slices) when no slice volume is supplied, so the model
        still works if called without per-slice context."""
        denom = slice_volume if slice_volume is not None else self.adv / 26.0
        return qty / max(denom, 1.0)

    def temporary_impact(self, qty: float, slice_volume: float | None = None) -> float:
        """Price impact (fraction of price) that affects only this slice's fill, then
        decays away. Square-root law in the participation rate (qty / slice volume), with
        a convexity multiplier that makes sweeping most of a slice's volume blow up
        superlinearly.
        """
        participation = self._participation(qty, slice_volume)
        return _TEMP_IMPACT_COEF * np.sqrt(participation) * _convexity_multiplier(participation)

    def permanent_impact(self, qty: float, slice_volume: float | None = None) -> float:
        """Price impact (fraction of price) that persists for the rest of the episode."""
        participation = self._participation(qty, slice_volume)
        return _PERM_IMPACT_COEF * participation * _convexity_multiplier(participation)


if __name__ == "__main__":
    data = load_daily_data()
    print({k: len(v) for k, v in data.items()})
    print("U-shaped volume curve (10 slices):", u_shaped_volume_curve(10).round(3))
