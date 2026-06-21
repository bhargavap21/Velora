"""
One-time data-priming step for the expanded training universe (TRAIN_TICKERS).

Fetching ~50+ tickers' daily + full-history minute bars lazily, serialized across 8
parallel training envs, would be slow and likely to trip Alpaca rate limits. This script
pulls everything up front instead: for each ticker, cache daily OHLCV (ensure_daily_data)
and full-history minute bars (load_minute_data, which already fetches+caches the whole
range on first call). Run once -- subsequent training runs read the cache directly.

A bad/delisted/typo'd symbol is logged and skipped, not fatal -- TRAIN_TICKERS is a
best-effort list sanity-checked here against live data, not a guaranteed-valid one.

Usage:
    .venv/bin/python -m execution_env.rl.prime_data_cache

Intended to run inside the Modal job (see modal_train.py) against the data_cache Volume,
but works the same locally if you'd rather prime the cache by hand.
"""

from __future__ import annotations

from execution_env.rl.train_ppo import TRAIN_TICKERS
from execution_env.simulator.market_sim import ensure_daily_data, load_minute_data


def prime(tickers: list[str] = TRAIN_TICKERS) -> tuple[list[str], list[str]]:
    """Returns (succeeded, failed) ticker lists."""
    succeeded, failed = [], []
    for ticker in tickers:
        try:
            daily = ensure_daily_data(ticker)
            minute = load_minute_data(ticker)
            minute_note = f"{len(minute)} minute bars" if minute is not None else "no minute bars (synthetic fallback)"
            print(f"[ok]   {ticker}: {len(daily)} daily bars, {minute_note}")
            succeeded.append(ticker)
        except Exception as exc:  # noqa: BLE001 -- one bad symbol shouldn't kill the run
            print(f"[skip] {ticker}: {exc}")
            failed.append(ticker)
    return succeeded, failed


def main() -> None:
    succeeded, failed = prime()
    print(f"\nPrimed {len(succeeded)}/{len(TRAIN_TICKERS)} tickers.")
    if failed:
        print(f"Skipped (no data from any provider): {failed}")


if __name__ == "__main__":
    main()
