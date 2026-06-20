"""HUD v6 environment for optimal trade execution (Velora).

Verified against hud-evals/autonomous-businesses-template (HUD v6.6):
  - Environment(name=...)
  - @env.tool for agent-facing tools
  - @env.template() async generators yielding prompt then EvaluationResult
  - tasks.py re-exports env + template handles with .slug assignments

Do NOT add ``from __future__ import annotations`` — it breaks the deploy manifest
TypeAdapter path for @env.template parameters (see template env.py NOTE).
"""

import json
import logging
import sys
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES, _TOTAL_SHARES
from execution_env.simulator.benchmark import compute_vwap, execution_vwap
from hud import Environment
from hud.graders import EvaluationResult

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="[%(levelname)s] %(name)s | %(message)s")
logger = logging.getLogger("velora")

env = Environment(name="velora-execution")

# Per-episode state reset at the top of each template run.
_EPISODE: ExecutionEnv | None = None
_EPISODE_INFO: dict[str, Any] | None = None
_SCHEDULE: list[float] | None = None

_REWARD_FLOOR = -3.0  # slippage_reward clip bound / scale
_REWARD_CEIL = 3.0


def _build_prompt(total_shares: int, n_slices: int, ticker: str, side: str) -> str:
    return f"""You are an execution trader. Your task: {side} {total_shares:,} shares of {ticker}
over {n_slices} time slices today without moving the price against yourself more than
necessary.

1. Call read_market_context() to see the open price and intraday volume curve.
2. Call submit_schedule() with a JSON string containing your plan.

Each schedule entry is a participation multiplier in [0, 2] for that slice — 1.0 means
"trade at exactly the VWAP-implied rate," below 1.0 trades less, above 1.0 trades more.
The order must be fully filled by the end.

You will be scored on slippage vs. VWAP (lower slippage is better) plus a penalty for
any unfilled inventory.

submit_schedule JSON shape:
{{"schedule": [m0, m1, ..., m{n_slices - 1}], "reasoning": "..."}}"""


def _normalize_reward(raw: float) -> float:
    """Map slippage_reward's roughly [-3, 3] range into [0, 1] for HUD."""
    return float(max(0.0, min(1.0, (raw - _REWARD_FLOOR) / (_REWARD_CEIL - _REWARD_FLOOR))))


def run_scenario(
    schedule: list[float],
    total_shares: int = _TOTAL_SHARES,
    n_slices: int = _N_SLICES,
    *,
    ticker: str = "AAPL",
    side: str = "buy",
    seed: int = 42,
) -> dict[str, Any]:
    """Replay a proposed schedule through ExecutionEnv and return the scored result."""
    gym_env = ExecutionEnv(n_slices=n_slices, total_shares=total_shares, side=side, tickers=[ticker])
    gym_env.reset(seed=seed)

    total_reward = 0.0
    for i in range(n_slices):
        multiplier = schedule[i] if i < len(schedule) else 2.0
        _, reward, terminated, _, _ = gym_env.step(np.array([multiplier], dtype=np.float32))
        total_reward += reward
        if terminated:
            break

    return {"reward": total_reward, "done": True}


def _grade_submitted_schedule() -> tuple[float, dict[str, Any]]:
    if _EPISODE is None or _EPISODE_INFO is None:
        return 0.0, {"error": "episode not initialized"}

    if _SCHEDULE is None:
        return 0.0, {"error": "no schedule submitted via submit_schedule()"}

    n_slices = int(_EPISODE_INFO["n_slices"])
    side = _EPISODE._side
    schedule = list(_SCHEDULE)

    if len(schedule) < n_slices:
        schedule.extend([2.0] * (n_slices - len(schedule)))

    total_reward = 0.0
    for i in range(n_slices):
        multiplier = float(np.clip(schedule[i], 0.0, 2.0))
        _, reward, terminated, _, _ = _EPISODE.step(np.array([multiplier], dtype=np.float32))
        total_reward += reward
        if terminated:
            break

    exec_prices = np.array(_EPISODE._exec_prices)
    exec_qty = np.array(_EPISODE._exec_quantities)
    agent_vwap = execution_vwap(exec_prices, exec_qty)
    benchmark_vwap = compute_vwap(_EPISODE._path[1:], _EPISODE._volume_curve * _EPISODE._total_shares)
    filled_fraction = 1.0 - _EPISODE._shares_remaining / _EPISODE._total_shares

    slippage_bps = 0.0
    if benchmark_vwap > 0 and agent_vwap > 0:
        slippage_bps = (benchmark_vwap - agent_vwap) / benchmark_vwap * 10_000
        if side == "sell":
            slippage_bps *= -1

    info = {
        "ticker": _EPISODE_INFO["ticker"],
        "side": side,
        "raw_reward": total_reward,
        "slippage_bps": round(slippage_bps, 2),
        "agent_vwap": round(agent_vwap, 4),
        "benchmark_vwap": round(benchmark_vwap, 4),
        "filled_fraction": round(filled_fraction, 4),
        "n_slices_submitted": len(_SCHEDULE),
        "summary": (
            f"{side} {_EPISODE_INFO['ticker']}: slippage {slippage_bps:.1f} bps vs VWAP, "
            f"filled {filled_fraction:.0%}"
        ),
    }
    return total_reward, info


@env.tool
async def read_market_context() -> dict[str, Any]:
    """Return the current episode's market context: ticker, order size, open price, volume curve."""
    if _EPISODE_INFO is None:
        return {"error": "No active episode. Start a HUD template task first."}
    return {
        "ticker": _EPISODE_INFO["ticker"],
        "total_shares": _EPISODE_INFO["total_shares"],
        "n_slices": _EPISODE_INFO["n_slices"],
        "open_price": _EPISODE_INFO["open_price"],
        "volume_curve": _EPISODE_INFO["volume_curve"],
        "side": _EPISODE._side if _EPISODE else "buy",
    }


@env.tool
async def submit_schedule(schedule_json: str) -> dict[str, Any]:
    """Submit an execution schedule as JSON: {"schedule": [float, ...], "reasoning": "..."}."""
    global _SCHEDULE
    try:
        parsed = json.loads(schedule_json)
    except json.JSONDecodeError as exc:
        return {"accepted": False, "reason": f"Invalid JSON: {exc}"}

    schedule = parsed.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        return {"accepted": False, "reason": "Missing non-empty 'schedule' list."}

    try:
        _SCHEDULE = [float(x) for x in schedule]
    except (TypeError, ValueError):
        return {"accepted": False, "reason": "Schedule values must be numbers."}

    return {
        "accepted": True,
        "n_slices": len(_SCHEDULE),
        "reasoning": parsed.get("reasoning", ""),
    }


async def _run_execution_template(
    prompt: str,
    *,
    ticker: str,
    side: str,
    total_shares: int = _TOTAL_SHARES,
    seed: int = 42,
) -> AsyncGenerator[Any, Any]:
    global _EPISODE, _EPISODE_INFO, _SCHEDULE
    _SCHEDULE = None
    _EPISODE = ExecutionEnv(total_shares=total_shares, side=side, tickers=[ticker])
    _, info = _EPISODE.reset(seed=seed)
    _EPISODE_INFO = info

    yield prompt

    raw_reward, grade_info = _grade_submitted_schedule()
    normalized = _normalize_reward(raw_reward)
    logger.info("%s %s reward=%.3f (raw=%.3f)", side, ticker, normalized, raw_reward)
    yield EvaluationResult(
        reward=normalized,
        content=grade_info.get("summary", ""),
        info=grade_info,
    )


@env.template()
async def buy_10k_aapl(prompt: str) -> AsyncGenerator[Any, Any]:
    """Buy 10,000 AAPL shares over one trading day; minimize slippage vs. VWAP."""
    async for item in _run_execution_template(prompt, ticker="AAPL", side="buy"):
        yield item


@env.template()
async def buy_10k_tsla(prompt: str) -> AsyncGenerator[Any, Any]:
    """Buy 10,000 TSLA shares (higher volatility) over one trading day."""
    async for item in _run_execution_template(prompt, ticker="TSLA", side="buy"):
        yield item


@env.template()
async def sell_10k_spy(prompt: str) -> AsyncGenerator[Any, Any]:
    """Sell 10,000 SPY shares over one trading day."""
    async for item in _run_execution_template(prompt, ticker="SPY", side="sell"):
        yield item


if __name__ == "__main__":
    result = run_scenario([1.0] * _N_SLICES)
    print("VWAP-matching schedule result:", result)
