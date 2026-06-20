"""
Execution benchmarks and reward for the optimal-execution environment.

The agent is scored against VWAP (volume-weighted average price) -- the standard
benchmark real execution desks are measured against -- not against an arbitrary score.
"""

from __future__ import annotations

import numpy as np


def compute_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
    """Volume-weighted average price over an array of (price, volume) per slice."""
    total_volume = volumes.sum()
    if total_volume <= 0:
        return float(prices.mean())
    return float(np.sum(prices * volumes) / total_volume)


def compute_twap(prices: np.ndarray) -> float:
    """Time-weighted average price -- the naive baseline (equal-sized slices)."""
    return float(prices.mean())


def execution_vwap(exec_prices: np.ndarray, exec_quantities: np.ndarray) -> float:
    """The agent's own volume-weighted average execution price."""
    total = exec_quantities.sum()
    if total <= 0:
        return 0.0
    return float(np.sum(exec_prices * exec_quantities) / total)


def slippage_reward(
    agent_vwap: float,
    benchmark_vwap: float,
    side: str,
    filled_fraction: float,
    unfilled_penalty_coef: float = 1.0,
) -> float:
    """Reward = -slippage vs. benchmark (in basis points), penalized for unfilled
    inventory at episode end.

    side: "buy" or "sell". For a buy, a lower agent_vwap than benchmark is good (positive
    reward); for a sell, a higher agent_vwap is good.

    TODO(benchmark): tune unfilled_penalty_coef -- should dominate the reward if the
    agent fails to complete the order, since an incomplete order is a worse outcome than
    any amount of slippage.
    """
    if benchmark_vwap <= 0:
        return 0.0

    raw_slippage_bps = (benchmark_vwap - agent_vwap) / benchmark_vwap * 10_000
    if side == "sell":
        raw_slippage_bps *= -1

    unfilled_penalty = unfilled_penalty_coef * (1.0 - filled_fraction) * 100
    return raw_slippage_bps - unfilled_penalty


if __name__ == "__main__":
    prices = np.array([100.0, 100.5, 101.0, 100.8])
    volumes = np.array([1000, 1500, 1200, 1300])
    print("VWAP:", compute_vwap(prices, volumes))
    print("TWAP:", compute_twap(prices))

    agent_prices = np.array([100.1, 100.6])
    agent_qty = np.array([5000, 5000])
    agent_vwap = execution_vwap(agent_prices, agent_qty)
    print("Agent VWAP:", agent_vwap)
    print("Reward (full fill):", slippage_reward(agent_vwap, compute_vwap(prices, volumes), "buy", 1.0))
    print("Reward (50% fill):", slippage_reward(agent_vwap, compute_vwap(prices, volumes), "buy", 0.5))
