"""
Task definitions for the optimal-execution HUD environment.

TODO(tasks.py): confirm task schema against a real cloned template (see env.py's
module docstring) -- this is written from README descriptions of the task tables in
autonomous-businesses-template / verilog-template, not verified source.
"""

from __future__ import annotations

TASKS = [
    {
        "slug": "buy-10k-aapl",
        "prompt_args": {"ticker": "AAPL", "total_shares": 10_000, "side": "buy"},
        "description": "Buy 10,000 shares of AAPL over one trading day, minimize slippage vs. VWAP.",
    },
    {
        "slug": "buy-10k-tsla",
        "prompt_args": {"ticker": "TSLA", "total_shares": 10_000, "side": "buy"},
        "description": "Buy 10,000 shares of TSLA (higher volatility) over one trading day.",
    },
    {
        "slug": "sell-10k-spy",
        "prompt_args": {"ticker": "SPY", "total_shares": 10_000, "side": "sell"},
        "description": "Sell 10,000 shares of SPY over one trading day.",
    },
]
