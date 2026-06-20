"""
HTTP API for the optimal-execution simulator, run live on request rather than replayed
from a static fixture.

GET /api/episode runs one full ExecutionEnv episode server-side with the requested
policy and returns its complete trace (price path, volume curve, per-slice fills) as a
single JSON response.

GET /api/episode/stream runs the same episode but streams it slice-by-slice over
Server-Sent Events (SSE): a `meta` event with the static episode info up front, one
`slice` event per executed slice as the server computes it, then a final `done` event.
This is genuinely live -- the frontend renders each slice as the backend produces it,
rather than replaying a precomputed trace.

GET /api/config returns sandbox defaults, constraints, and available tickers/dates.

Usage:
    uvicorn execution_env.server:app --reload --port 8010
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES, _TICKERS, _TOTAL_SHARES
from execution_env.rl.train_ppo import MODEL_PATH, twap_action
from execution_env.sandbox_config import (
    CONSTRAINTS,
    build_sandbox_config,
    resolve_regime_date,
    shares_from_capital,
)
from execution_env.simulator.market_sim import ensure_daily_date, load_daily_data

app = FastAPI(title="Velora execution API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3002"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Default per-slice delay for the SSE stream, so the episode unfolds at a watchable pace
# rather than dumping all slices in a few milliseconds. Overridable via ?tick_ms=.
_DEFAULT_TICK_MS = 250

_ppo_model = None


def _get_ppo_model():
    """Lazily loads the PPO checkpoint on first request; cached after that. Returns
    None if no checkpoint has been trained yet (see train_ppo.py).

    Re-checks for the checkpoint on every call until one is successfully loaded, so a
    server started *before* training finishes will pick up the checkpoint as soon as
    it appears -- without needing a restart."""
    global _ppo_model
    if _ppo_model is None and MODEL_PATH.exists():
        from stable_baselines3 import PPO

        _ppo_model = PPO.load(MODEL_PATH)
    return _ppo_model


def _propose_llm_schedule(info: dict) -> tuple[list[float], dict, str]:
    """Call the Claude LLM agent once to propose a schedule for this episode.

    Returns (multipliers, parsed_json, raw_response). Falls back to TWAP on any error
    so the demo keeps running even when the API key has rate-limit issues.
    """
    from execution_env.agents.llm_agent import (
        propose_schedule,
        twap as llm_twap,
    )

    try:
        return propose_schedule(
            info["ticker"],
            info["total_shares"],
            info["n_slices"],
            info["open_price"],
            info["volume_curve"],
        )
    except Exception as exc:
        fallback = llm_twap(info["n_slices"], np.asarray(info["volume_curve"], dtype=float))
        return fallback, {"primitive": "twap", "reasoning": f"API error — fell back to TWAP: {exc}"}, ""


def _propose_fireworks_schedule(info: dict) -> tuple[list[float], dict, str]:
    """Call the Fireworks (gpt-oss) agent once to propose a schedule for this episode.

    Same shape and TWAP-fallback behaviour as _propose_llm_schedule.
    """
    from execution_env.agents.fireworks_agent import propose_schedule as fw_propose, twap as fw_twap

    try:
        return fw_propose(
            info["ticker"],
            info["total_shares"],
            info["n_slices"],
            info["open_price"],
            info["volume_curve"],
        )
    except Exception as exc:
        fallback = fw_twap(info["n_slices"], np.asarray(info["volume_curve"], dtype=float))
        return fallback, {"primitive": "twap", "reasoning": f"API error — fell back to TWAP: {exc}"}, ""


def _validate_shares(total_shares: int) -> None:
    bounds = CONSTRAINTS["total_shares"]
    if not bounds["min"] <= total_shares <= bounds["max"]:
        raise HTTPException(
            400,
            f"total_shares must be between {bounds['min']:,} and {bounds['max']:,}",
        )


def _validate_slices(n_slices: int) -> None:
    bounds = CONSTRAINTS["n_slices"]
    if not bounds["min"] <= n_slices <= bounds["max"]:
        raise HTTPException(
            400,
            f"n_slices must be between {bounds['min']} and {bounds['max']}",
        )


def _resolve_date(ticker: str, date: str | None, regime: str | None, seed: int | None) -> str | None:
    """Resolve explicit date or regime sample into a YYYY-MM-DD string."""
    if date:
        return date
    if regime and regime != "random":
        df = load_daily_data()[ticker]
        return resolve_regime_date(df, regime, seed)
    return None


@dataclass
class _EpisodeContext:
    """Everything needed to run (and incrementally describe) one episode, shared by the
    JSON and SSE endpoints. Built by _build_context() after all validation/setup."""

    env: ExecutionEnv
    info: dict
    policy: str
    n_slices: int
    schedule_label: str
    regime: str | None
    seed: int | None
    ppo_model: object | None = None
    llm_multipliers: list[float] = field(default_factory=list)
    llm_meta: dict = field(default_factory=dict)
    # Per-slice price tracking for the LLM/fireworks pause-on-adverse-move modifier.
    _current_price: float = 0.0
    _last_exec_price: float | None = None

    def __post_init__(self):
        self._current_price = float(self.info.get("open_price", 0.0))

    def action_for(self, i: int, obs: np.ndarray) -> np.ndarray:
        """Select the action for slice i given the current observation, advancing any
        internal per-slice state (LLM pause tracking)."""
        if self.policy == "ppo":
            action, _ = self.ppo_model.predict(obs, deterministic=True)
            return action

        if self.policy in ("llm", "fireworks"):
            if i > 0:
                # obs[2] = recent_return = (path[i] - path[i-1]) / path[i-1]
                self._current_price *= 1.0 + float(obs[2])

            multiplier = self.llm_multipliers[i]
            if (
                i < self.n_slices - 1
                and self.llm_meta.get("llm_pause_enabled")
                and self._last_exec_price is not None
                and self.llm_meta.get("llm_pause_threshold_bps", 0) > 0
            ):
                price_move_bps = (self._current_price - self._last_exec_price) / self._last_exec_price * 10_000
                if price_move_bps > self.llm_meta["llm_pause_threshold_bps"]:
                    multiplier = 0.0

            if multiplier > 0:
                self._last_exec_price = self._current_price
            return np.array([multiplier], dtype=np.float32)

        return twap_action(i, self.n_slices)

    def meta(self) -> dict:
        """Static episode info known at reset (everything except the per-slice fills,
        the final reward, and filled fraction)."""
        notional_usd = round(self.info["open_price"] * self.env._total_shares, 2)
        return {
            "ticker": self.env._ticker,
            "side": self.env._side,
            "date": self.info["date"],
            "total_shares": self.env._total_shares,
            "notional_usd": notional_usd,
            "n_slices": self.env._n_slices,
            "open_price": self.info["open_price"],
            "close_price": self.info["close_price"],
            "adv": self.info["adv"],
            "data_source": self.info["data_source"],
            "seed": self.seed,
            "regime": self.regime,
            "path": self.env._path.tolist(),
            "volume_curve": self.env._volume_curve.tolist(),
            "schedule_label": self.schedule_label,
            "policy": self.policy,
            **self.llm_meta,
        }


def _build_context(
    policy: str,
    ticker: str,
    side: str,
    total_shares: int | None,
    capital_usd: float | None,
    n_slices: int,
    date: str | None,
    regime: str | None,
    seed: int | None,
) -> _EpisodeContext:
    """Validate inputs, resolve the scenario, set up the env + policy, and propose the
    LLM/fireworks schedule if needed. Raises HTTPException on any invalid input.

    All fallible work happens here (synchronously) so both endpoints fail fast with a
    proper HTTP status before any streaming begins.
    """
    if ticker not in _TICKERS:
        raise HTTPException(400, f"ticker must be one of {_TICKERS}")

    _validate_slices(n_slices)

    if policy == "ppo" and n_slices != _N_SLICES:
        raise HTTPException(
            400,
            f"PPO policy requires n_slices={_N_SLICES} (trained checkpoint is fixed to 15-min slices)",
        )

    ppo_model = None
    if policy == "ppo":
        ppo_model = _get_ppo_model()
        if ppo_model is None:
            raise HTTPException(
                503, "No PPO checkpoint found -- run `python -m execution_env.rl.train_ppo` first."
            )

    if policy == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set -- add it to your .env file.")

    if policy == "fireworks" and not os.environ.get("FIREWORKS_API_KEY"):
        raise HTTPException(503, "FIREWORKS_API_KEY not set -- add it to your .env file.")

    resolved_date = _resolve_date(ticker, date, regime, seed)
    if resolved_date:
        try:
            ensure_daily_date(ticker, resolved_date)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    shares = total_shares if total_shares is not None else _TOTAL_SHARES
    if capital_usd is not None:
        df = load_daily_data()[ticker]
        if resolved_date:
            ref_price = float(df.loc[pd.Timestamp(resolved_date), "Open"])
        else:
            ref_price = float(df["Close"].iloc[-1])
        try:
            shares = shares_from_capital(capital_usd, ref_price)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    _validate_shares(shares)

    reset_options: dict = {"ticker": ticker}
    if resolved_date:
        reset_options["date"] = resolved_date

    env = ExecutionEnv(n_slices=n_slices, total_shares=shares, side=side, tickers=[ticker])
    try:
        _obs, info = env.reset(seed=seed, options=reset_options)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    llm_multipliers: list[float] = []
    llm_meta: dict = {}
    if policy in ("llm", "fireworks"):
        if policy == "llm":
            multipliers, parsed, _raw = _propose_llm_schedule(info)
        else:
            multipliers, parsed, _raw = _propose_fireworks_schedule(info)
        llm_multipliers = multipliers
        pause_cfg = parsed.get("pause_if_adverse_move", {"enabled": False})
        llm_meta = {
            "llm_primitive": parsed.get("primitive", "unknown"),
            "llm_reasoning": parsed.get("reasoning", ""),
            "llm_pause_enabled": bool(pause_cfg.get("enabled", False)),
            "llm_pause_threshold_bps": float(pause_cfg.get("threshold_bps", 0)),
        }

    if policy == "llm":
        schedule_label = f"Claude LLM — {llm_meta.get('llm_primitive', 'unknown')}"
    elif policy == "fireworks":
        schedule_label = f"GPT-OSS (Fireworks) — {llm_meta.get('llm_primitive', 'unknown')}"
    elif policy == "ppo":
        schedule_label = "PPO (trained)"
    else:
        schedule_label = "TWAP (equal participation)"

    return _EpisodeContext(
        env=env,
        info=info,
        policy=policy,
        n_slices=n_slices,
        schedule_label=schedule_label,
        regime=regime,
        seed=seed,
        ppo_model=ppo_model,
        llm_multipliers=llm_multipliers,
        llm_meta=llm_meta,
    )


@app.get("/api/config")
def get_config():
    return build_sandbox_config()


@app.get("/api/episode")
def get_episode(
    policy: Literal["twap", "ppo", "llm", "fireworks"] = Query("twap"),
    ticker: str = Query("AAPL"),
    side: Literal["buy", "sell"] = Query("buy"),
    total_shares: int | None = Query(None),
    capital_usd: float | None = Query(None),
    n_slices: int = Query(_N_SLICES),
    date: str | None = Query(None, description="Historical replay date (YYYY-MM-DD)"),
    regime: str | None = Query(None, description="Market regime sample when date is omitted"),
    seed: int | None = Query(None, description="Seed for reproducible day sampling"),
):
    ctx = _build_context(policy, ticker, side, total_shares, capital_usd, n_slices, date, regime, seed)

    obs = ctx.env._build_obs()
    total_reward = 0.0
    for i in range(ctx.n_slices):
        action = ctx.action_for(i, obs)
        obs, reward, terminated, truncated, _ = ctx.env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    filled_fraction = 1.0 - ctx.env._shares_remaining / ctx.env._total_shares

    return {
        **ctx.meta(),
        "exec_prices": ctx.env._exec_prices,
        "exec_quantities": ctx.env._exec_quantities,
        "final_reward": round(total_reward, 4),
        "filled_fraction": round(filled_fraction, 4),
    }


@app.get("/api/episode/stream")
async def get_episode_stream(
    policy: Literal["twap", "ppo", "llm", "fireworks"] = Query("twap"),
    ticker: str = Query("AAPL"),
    side: Literal["buy", "sell"] = Query("buy"),
    total_shares: int | None = Query(None),
    capital_usd: float | None = Query(None),
    n_slices: int = Query(_N_SLICES),
    date: str | None = Query(None, description="Historical replay date (YYYY-MM-DD)"),
    regime: str | None = Query(None, description="Market regime sample when date is omitted"),
    seed: int | None = Query(None, description="Seed for reproducible day sampling"),
    tick_ms: int = Query(_DEFAULT_TICK_MS, ge=0, le=2000, description="Delay between streamed slices (ms)"),
):
    """Stream one episode slice-by-slice over SSE.

    Emits: one `meta` event, then one `slice` event per executed slice, then a `done`
    event. On a setup error (bad ticker, missing key, etc.) emits a single `error`
    event instead, since EventSource can't surface a non-200 response body.
    """
    import json

    try:
        ctx = _build_context(policy, ticker, side, total_shares, capital_usd, n_slices, date, regime, seed)
    except HTTPException as exc:
        # Bind the detail to a local: Python deletes `exc` when the except block exits,
        # so the generator closure can't reference it directly.
        detail = exc.detail

        async def error_stream():
            yield {"event": "error", "data": json.dumps({"detail": detail})}

        return EventSourceResponse(error_stream())

    delay = tick_ms / 1000.0

    async def event_stream():
        yield {"event": "meta", "data": json.dumps(ctx.meta())}

        obs = ctx.env._build_obs()
        total_reward = 0.0
        for i in range(ctx.n_slices):
            action = ctx.action_for(i, obs)
            obs, reward, terminated, truncated, _ = ctx.env.step(action)
            total_reward += reward

            yield {
                "event": "slice",
                "data": json.dumps(
                    {
                        "i": i,
                        "exec_price": ctx.env._exec_prices[-1],
                        "exec_quantity": ctx.env._exec_quantities[-1],
                        "shares_remaining": round(ctx.env._shares_remaining, 4),
                    }
                ),
            }

            if terminated or truncated:
                break
            if delay > 0:
                await asyncio.sleep(delay)

        filled_fraction = 1.0 - ctx.env._shares_remaining / ctx.env._total_shares
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "final_reward": round(total_reward, 4),
                    "filled_fraction": round(filled_fraction, 4),
                    "exec_prices": ctx.env._exec_prices,
                    "exec_quantities": ctx.env._exec_quantities,
                }
            ),
        }

    return EventSourceResponse(event_stream())


@app.get("/api/policies")
def get_policies():
    available = ["twap"]
    if MODEL_PATH.exists():
        available.append("ppo")
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("llm")
    if os.environ.get("FIREWORKS_API_KEY"):
        available.append("fireworks")
    return {"available": available}
