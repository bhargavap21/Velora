"""Sandbox configuration helpers: regime sampling, defaults, and constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd

from execution_env.simulator.market_sim import (
    DEFAULT_TICKERS,
    _SESSION_END,
    _SESSION_START,
    _START,
    _data_end,
    load_daily_data,
)

# Curated suggestions for the UI's ticker datalist -- not a validation allowlist.
# Any symbol the data layer can resolve (ensure_daily_data) is a valid request.
_TICKERS = DEFAULT_TICKERS
_N_SLICES = 26
_TOTAL_SHARES = 10_000

REGIME_SAMPLES = [
    {
        "id": "random",
        "label": "Random day",
        "description": "Sample any trading day from history",
    },
    {
        "id": "high_vol",
        "label": "High volatility",
        "description": "Top-decile intraday range — stress-test impact",
    },
    {
        "id": "low_vol",
        "label": "Low volatility",
        "description": "Bottom-decile range — calm tape, tight spreads",
    },
    {
        "id": "rally",
        "label": "Strong rally",
        "description": "Top-decile up day — buying into momentum",
    },
    {
        "id": "selloff",
        "label": "Sharp selloff",
        "description": "Bottom-decile down day — selling into weakness",
    },
]

TIMEFRAME_PRESETS = [
    {"id": "30min", "label": "30-min slices", "n_slices": 13, "minutes_per_slice": 30},
    {"id": "15min", "label": "15-min slices", "n_slices": 26, "minutes_per_slice": 15},
    {"id": "7min", "label": "7-min slices", "n_slices": 52, "minutes_per_slice": 7},
]

CONSTRAINTS = {
    # Ceiling is high enough for institutional orders sized as a fraction of ADV
    # (where participation-driven impact -- and thus scheduling -- actually matters).
    "total_shares": {"min": 1_000, "max": 50_000_000, "step": 1_000, "default": _TOTAL_SHARES},
    "n_slices": {"min": 6, "max": 78, "default": _N_SLICES},
    "capital_usd": {"min": 10_000, "max": 5_000_000_000, "step": 1_000, "default": 1_000_000},
    # Order size as a percentage of average daily volume.
    "adv_pct": {"min": 0.5, "max": 25.0, "step": 0.5, "default": 8.0},
}


def shares_from_adv_pct(adv_pct: float, reference_adv: float) -> int:
    """Convert an order size expressed as a percentage of ADV into whole shares."""
    if reference_adv <= 0:
        raise ValueError("reference_adv must be positive")
    raw = int(reference_adv * adv_pct / 100.0)
    rounded = max(CONSTRAINTS["total_shares"]["min"], (raw // 1_000) * 1_000)
    return min(rounded, CONSTRAINTS["total_shares"]["max"])


def resolve_regime_date(df: pd.DataFrame, regime: str, seed: int | None = None) -> str:
    """Pick a historical date from `df` matching the requested market regime."""
    if regime == "random" or not regime:
        rng = np.random.default_rng(seed)
        idx = int(rng.integers(len(df)))
        return df.index[idx].strftime("%Y-%m-%d")

    work = df.copy()
    work["range_pct"] = (work["High"] - work["Low"]) / work["Open"].clip(lower=0.01)
    work["return_pct"] = (work["Close"] - work["Open"]) / work["Open"].clip(lower=0.01)

    if regime == "high_vol":
        threshold = work["range_pct"].quantile(0.9)
        pool = work[work["range_pct"] >= threshold]
    elif regime == "low_vol":
        threshold = work["range_pct"].quantile(0.1)
        pool = work[work["range_pct"] <= threshold]
    elif regime == "rally":
        threshold = work["return_pct"].quantile(0.9)
        pool = work[work["return_pct"] >= threshold]
    elif regime == "selloff":
        threshold = work["return_pct"].quantile(0.1)
        pool = work[work["return_pct"] <= threshold]
    else:
        raise ValueError(f"Unknown regime {regime!r}")

    if pool.empty:
        pool = work

    rng = np.random.default_rng(seed)
    pick = pool.index[int(rng.integers(len(pool)))]
    return pick.strftime("%Y-%m-%d")


def shares_from_capital(capital_usd: float, reference_price: float) -> int:
    """Convert dollar notional to whole shares, rounded down to nearest 100."""
    if reference_price <= 0:
        raise ValueError("reference_price must be positive")
    raw = int(capital_usd / reference_price)
    rounded = max(CONSTRAINTS["total_shares"]["min"], (raw // 100) * 100)
    return min(rounded, CONSTRAINTS["total_shares"]["max"])


def build_sandbox_config() -> dict:
    """Return defaults and constraints for the sandbox UI."""
    data = load_daily_data()
    tickers: dict[str, dict] = {}
    for ticker in _TICKERS:
        df = data[ticker]
        last_close = float(df["Close"].iloc[-1])
        median_adv = float(df["Volume"].median())
        tickers[ticker] = {
            "date_range": {
                "start": df.index[0].strftime("%Y-%m-%d"),
                "end": df.index[-1].strftime("%Y-%m-%d"),
            },
            "last_close": round(last_close, 2),
            "median_adv": round(median_adv),
            "sample_dates": _curated_sample_dates(df, ticker),
        }

    return {
        "tickers": _TICKERS,
        "ticker_info": tickers,
        "defaults": {
            "ticker": "AAPL",
            "side": "buy",
            "total_shares": _TOTAL_SHARES,
            "n_slices": _N_SLICES,
            "capital_usd": CONSTRAINTS["capital_usd"]["default"],
            "adv_pct": CONSTRAINTS["adv_pct"]["default"],
            "order_mode": "adv_pct",
            "regime": "random",
            "policy": "ppo",
            "baseline": "naive_twap",
        },
        "date_range": {"start": _START, "end": _data_end()},
        "session": {
            "open": _SESSION_START,
            "close": _SESSION_END,
            "timezone": "America/New_York",
        },
        "timeframe_presets": TIMEFRAME_PRESETS,
        "regime_samples": REGIME_SAMPLES,
        "constraints": CONSTRAINTS,
    }


def _curated_sample_dates(df: pd.DataFrame, ticker: str) -> list[dict]:
    """One pinned example date per regime for quick picks in the UI."""
    rng = np.random.default_rng(abs(hash(ticker)) % (2**31))
    samples = []
    for regime in REGIME_SAMPLES:
        if regime["id"] == "random":
            continue
        date = resolve_regime_date(df, regime["id"], seed=int(rng.integers(1_000_000)))
        row = df.loc[pd.Timestamp(date)]
        samples.append(
            {
                "regime": regime["id"],
                "date": date,
                "open": round(float(row["Open"]), 2),
                "close": round(float(row["Close"]), 2),
                "return_pct": round(float((row["Close"] - row["Open"]) / row["Open"] * 100), 2),
            }
        )
    return samples
