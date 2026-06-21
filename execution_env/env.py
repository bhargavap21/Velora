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
import random
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import numpy as np

# hud.eval.LocalRuntime spawns this file as a child process with cwd set to
# execution_env/ (this file's own directory), so the repo root isn't on sys.path and the
# absolute `execution_env.*` imports below fail with ModuleNotFoundError unless we put it
# there ourselves. Must run before those imports.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from execution_env.agents.llm_agent import _extract_json
from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES, _TOTAL_SHARES
from execution_env.sandbox_config import shares_from_adv_pct
from execution_env.simulator.benchmark import compute_vwap, execution_vwap, slippage_reward
from execution_env.simulator.market_sim import ensure_daily_data
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
        # A model that never successfully calls submit_schedule() must score worse than
        # any actual fill, not the same as raw_reward=0.0 (which _normalize_reward maps
        # to 0.5 -- i.e. "as good as exactly matching VWAP"). Treat it as a 0%-filled
        # order: slippage_reward's own unfilled-inventory penalty (>> the slippage clip)
        # puts this at the reward floor.
        penalty = slippage_reward(agent_vwap=0.0, benchmark_vwap=1.0, side=_EPISODE._side, filled_fraction=0.0)
        return penalty, {"error": "no schedule submitted via submit_schedule()"}

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
        # Tolerant of prose/code-fences around the JSON object (same parser
        # agents/llm_agent.py uses for LLM-emitted schedules), not a strict json.loads --
        # an RFT'd model mid-training is more likely than a hand-written client to wrap
        # its output in markdown fences or trailing commentary.
        parsed = _extract_json(schedule_json)
    except json.JSONDecodeError as exc:
        return {"accepted": False, "reason": f"Invalid JSON: {exc}"}

    if not isinstance(parsed, dict):
        return {"accepted": False, "reason": "Expected a JSON object, e.g. {\"schedule\": [...]}."}

    schedule = parsed.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        return {"accepted": False, "reason": "Missing non-empty 'schedule' list."}

    try:
        _SCHEDULE = [float(x) for x in schedule]
    except (TypeError, ValueError):
        return {"accepted": False, "reason": "Schedule values must be numbers."}

    response: dict[str, Any] = {
        "accepted": True,
        "n_slices": len(_SCHEDULE),
        "reasoning": parsed.get("reasoning", ""),
    }
    # Length mismatch isn't fatal -- _grade_submitted_schedule() pads a short schedule
    # (forces a full fill) and ignores extra entries -- but surface it so the agent (or a
    # human debugging training) can see the schedule didn't match what was asked for.
    expected_n_slices = _EPISODE_INFO["n_slices"] if _EPISODE_INFO else None
    if expected_n_slices is not None and len(_SCHEDULE) != expected_n_slices:
        response["warning"] = (
            f"Expected {expected_n_slices} slices, got {len(_SCHEDULE)} -- "
            "short schedules are padded to fully fill, extra entries are ignored."
        )
    return response


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


def _sample_random_scenario() -> dict[str, Any]:
    """Sample a random (ticker, side, total_shares, seed) scenario sized as a real
    fraction of ADV -- the regime where intraday scheduling actually moves slippage,
    unlike the fixed 10k-share/seed=42 tasks above which carry negligible impact (see
    issue #15, "make the env trainable"). Sampled fresh on every invocation, not cached
    or pinned, so repeated `hud eval` runs see a genuinely different scenario each time.

    Deferred import of train_ppo: it pulls in matplotlib/stable_baselines3 for its PPO
    training code, which the 3 fixed tasks above have no reason to load at module-import
    time.
    """
    from execution_env.rl.train_ppo import TRAIN_TICKERS, _ORDER_ADV_PCT_RANGE

    ticker = random.choice(TRAIN_TICKERS)
    side = random.choice(["buy", "sell"])
    adv_pct_fraction = random.uniform(*_ORDER_ADV_PCT_RANGE)
    ref_adv = float(ensure_daily_data(ticker)["Volume"].median())
    total_shares = shares_from_adv_pct(adv_pct_fraction * 100, ref_adv)
    seed = random.randint(0, 2**31 - 1)
    return {"ticker": ticker, "side": side, "total_shares": total_shares, "seed": seed}


@env.template()
async def execution_random(prompt: str) -> AsyncGenerator[Any, Any]:
    """Randomized institutional-size execution task: ticker, side, order size (2-10% of
    ADV), and seed are all sampled fresh each invocation -- the trainable task the fixed
    10k-share tasks above can't provide (see issue #15, Phase 1)."""
    scenario = _sample_random_scenario()
    async for item in _run_execution_template(
        prompt,
        ticker=scenario["ticker"],
        side=scenario["side"],
        total_shares=scenario["total_shares"],
        seed=scenario["seed"],
    ):
        yield item


@env.template()
async def execution_fixed(prompt: str, ticker: str, side: str, total_shares: int, seed: int) -> AsyncGenerator[Any, Any]:
    """Same scenario shape as execution_random, but with the (ticker, side, total_shares,
    seed) pinned by the caller instead of sampled inside the generator.

    execution_random deliberately resamples a fresh scenario on every invocation (issue
    #15's "fully dynamic" requirement), but a GRPO training group needs N rollouts of the
    SAME scenario/prompt to compute a meaningful relative advantage -- N independent
    random scenarios isn't a group. Build N Task instances from this template, all with
    the same args (see rl/hud_rft_pipeline.py), to form one group.
    """
    async for item in _run_execution_template(prompt, ticker=ticker, side=side, total_shares=total_shares, seed=seed):
        yield item


if __name__ == "__main__":
    result = run_scenario([1.0] * _N_SLICES)
    print("VWAP-matching schedule result:", result)
