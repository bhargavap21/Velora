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

from execution_env.rl.execution_gym_env import (
    ExecutionEnv,
    _N_SLICES,
    _OBS_DIM,
    _TOTAL_SHARES,
    naive_twap_action,
)
from execution_env.rl.train_ppo import (
    MODEL_PATH,
    _HOLDOUT_DAY_RANGE,
    twap_action,
)
from execution_env.sandbox_config import (
    CONSTRAINTS,
    build_sandbox_config,
    resolve_regime_date,
    shares_from_adv_pct,
    shares_from_capital,
)
from execution_env.simulator.benchmark import compute_vwap, execution_vwap
from execution_env.simulator.market_sim import ensure_daily_data, ensure_daily_date, load_daily_data

app = FastAPI(title="Velora execution API")

# Extra production origins (e.g. the Vercel deployment) come from ALLOWED_ORIGINS,
# comma-separated, so the same image works in dev and prod without a code change.
_extra_origins = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3002", *_extra_origins],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Default per-slice delay for the SSE stream, so the episode unfolds at a watchable pace
# rather than dumping all slices in a few milliseconds. Overridable via ?tick_ms=.
_DEFAULT_TICK_MS = 250

_ppo_model = None


_ppo_model_error: str | None = None


def _get_ppo_model():
    """Lazily loads the PPO checkpoint on first request; cached after that. Returns
    None if no checkpoint has been trained yet (see train_ppo.py), or if the checkpoint's
    observation-space shape doesn't match the current ExecutionEnv (e.g. after an obs-space
    change like the liquidity/volatility features -- see _OBS_DIM) -- in which case
    `_ppo_model_error()` explains why, so callers can return a clear 503 instead of a raw
    shape-mismatch crash from stable_baselines3.

    Re-checks for the checkpoint on every call until one is successfully loaded, so a
    server started *before* training finishes will pick up the checkpoint as soon as
    it appears -- without needing a restart."""
    global _ppo_model, _ppo_model_error
    if _ppo_model is None and MODEL_PATH.exists():
        from stable_baselines3 import PPO

        loaded = PPO.load(MODEL_PATH)
        loaded_dim = loaded.observation_space.shape[0]
        if loaded_dim != _OBS_DIM:
            _ppo_model_error = (
                f"PPO checkpoint at {MODEL_PATH} was trained on a {loaded_dim}-dim "
                f"observation space, but the current ExecutionEnv produces {_OBS_DIM} dims "
                "-- retrain it (see 'Training on Modal' in execution_env/README.md)."
            )
        else:
            _ppo_model = loaded
    return _ppo_model


def _ppo_model_unavailable_message() -> str:
    return _ppo_model_error or "No PPO checkpoint found -- run `python -m execution_env.rl.train_ppo` first."


# Defensive clip (in bps) on the reported per-episode slippage metric. See _episode_metrics.
_SLIPPAGE_BPS_CLIP = 5_000.0

# Policy display labels, shared with the frontend's understanding of each arm.
POLICY_LABELS = {
    "naive_twap": "TWAP (equal-time)",
    "twap": "VWAP-match",
    "ppo": "PPO (RL agent)",
    "llm": "Claude",
    "fireworks": "GPT-OSS",
}


def _stateless_action(policy: str, env: ExecutionEnv, i: int, obs: np.ndarray, ppo_model) -> np.ndarray:
    """Action for the stateless policies (twap / naive_twap / ppo) used by the
    comparison and batch-eval endpoints. LLM policies are stateful (schedule proposed
    once) and handled separately in _EpisodeContext."""
    if policy == "ppo":
        action, _ = ppo_model.predict(obs, deterministic=True)
        return action
    if policy == "naive_twap":
        return naive_twap_action(env)
    return twap_action(i, env._n_slices)


def _episode_metrics(env: ExecutionEnv) -> dict:
    """Compute the execution-quality metrics for a finished episode: the agent's own
    VWAP, the benchmark VWAP, slippage vs benchmark in bps (sign-adjusted so positive =
    better than benchmark for both buy and sell), and filled fraction."""
    exec_prices = np.array(env._exec_prices, dtype=float)
    exec_quantities = np.array(env._exec_quantities, dtype=float)
    agent_vwap = execution_vwap(exec_prices, exec_quantities)
    benchmark_vwap = compute_vwap(env._path[1:], env._volume_curve * env._total_shares)
    filled_fraction = 1.0 - env._shares_remaining / env._total_shares

    slippage_bps = 0.0
    if benchmark_vwap > 0:
        sign = 1.0 if env._side == "buy" else -1.0
        slippage_bps = sign * (benchmark_vwap - agent_vwap) / benchmark_vwap * 10_000

    # Defensive bound on the *reported* metric. The impact model already saturates
    # participation (see _MAX_PARTICIPATION) so exec prices can't explode, keeping realistic
    # paths well inside this range; this clip is only a safety net so that no single
    # pathological path could ever dominate an aggregate mean we report as evidence.
    # +/-5000 bps (50%) is far beyond any realistic single-day execution slippage.
    slippage_bps = float(np.clip(slippage_bps, -_SLIPPAGE_BPS_CLIP, _SLIPPAGE_BPS_CLIP))

    return {
        "agent_vwap": round(agent_vwap, 4),
        "benchmark_vwap": round(benchmark_vwap, 4),
        "slippage_bps": round(slippage_bps, 2),
        "filled_fraction": round(filled_fraction, 4),
    }


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


def _resolve_date(df: pd.DataFrame, date: str | None, regime: str | None, seed: int | None) -> str | None:
    """Resolve explicit date or regime sample into a YYYY-MM-DD string."""
    if date:
        return date
    if regime and regime != "random":
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

        if self.policy == "naive_twap":
            return naive_twap_action(self.env)

        return twap_action(i, self.n_slices)

    def meta(self) -> dict:
        """Static episode info known at reset (everything except the per-slice fills,
        the final reward, and filled fraction)."""
        notional_usd = round(self.info["open_price"] * self.env._total_shares, 2)
        adv = self.info.get("adv", 0.0)
        order_adv_pct = round(self.env._total_shares / adv * 100, 2) if adv else None
        return {
            "ticker": self.env._ticker,
            "side": self.env._side,
            "date": self.info["date"],
            "total_shares": self.env._total_shares,
            "notional_usd": notional_usd,
            "order_adv_pct": order_adv_pct,
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
    adv_pct: float | None = None,
) -> _EpisodeContext:
    """Validate inputs, resolve the scenario, set up the env + policy, and propose the
    LLM/fireworks schedule if needed. Raises HTTPException on any invalid input.

    All fallible work happens here (synchronously) so both endpoints fail fast with a
    proper HTTP status before any streaming begins.
    """
    try:
        df = ensure_daily_data(ticker)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

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
            raise HTTPException(503, _ppo_model_unavailable_message())

    if policy == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "ANTHROPIC_API_KEY not set -- add it to your .env file.")

    if policy == "fireworks" and not os.environ.get("FIREWORKS_API_KEY"):
        raise HTTPException(503, "FIREWORKS_API_KEY not set -- add it to your .env file.")

    resolved_date = _resolve_date(df, date, regime, seed)
    if resolved_date:
        try:
            df = ensure_daily_date(ticker, resolved_date)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    shares = total_shares if total_shares is not None else _TOTAL_SHARES
    # Precedence: adv_pct (institutional sizing) > capital_usd > explicit total_shares.
    if adv_pct is not None:
        ref_adv = float(df["Volume"].median())
        try:
            shares = shares_from_adv_pct(adv_pct, ref_adv)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    elif capital_usd is not None:
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
        schedule_label = "PPO (trained RL agent)"
    elif policy == "naive_twap":
        schedule_label = "TWAP (equal-time baseline)"
    else:
        schedule_label = "VWAP-match (volume-curve participation)"

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
    policy: Literal["naive_twap", "twap", "ppo", "llm", "fireworks"] = Query("twap"),
    ticker: str = Query("AAPL"),
    side: Literal["buy", "sell"] = Query("buy"),
    total_shares: int | None = Query(None),
    capital_usd: float | None = Query(None),
    adv_pct: float | None = Query(None, description="Order size as a percentage of ADV"),
    n_slices: int = Query(_N_SLICES),
    date: str | None = Query(None, description="Historical replay date (YYYY-MM-DD)"),
    regime: str | None = Query(None, description="Market regime sample when date is omitted"),
    seed: int | None = Query(None, description="Seed for reproducible day sampling"),
):
    ctx = _build_context(policy, ticker, side, total_shares, capital_usd, n_slices, date, regime, seed, adv_pct)

    obs = ctx.env._build_obs()
    total_reward = 0.0
    for i in range(ctx.n_slices):
        action = ctx.action_for(i, obs)
        obs, reward, terminated, truncated, _ = ctx.env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    metrics = _episode_metrics(ctx.env)

    return {
        **ctx.meta(),
        "exec_prices": ctx.env._exec_prices,
        "exec_quantities": ctx.env._exec_quantities,
        "final_reward": round(total_reward, 4),
        **metrics,
    }


@app.get("/api/episode/stream")
async def get_episode_stream(
    policy: Literal["naive_twap", "twap", "ppo", "llm", "fireworks"] = Query("twap"),
    ticker: str = Query("AAPL"),
    side: Literal["buy", "sell"] = Query("buy"),
    total_shares: int | None = Query(None),
    capital_usd: float | None = Query(None),
    adv_pct: float | None = Query(None, description="Order size as a percentage of ADV"),
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
        ctx = _build_context(policy, ticker, side, total_shares, capital_usd, n_slices, date, regime, seed, adv_pct)
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

        metrics = _episode_metrics(ctx.env)
        yield {
            "event": "done",
            "data": json.dumps(
                {
                    "final_reward": round(total_reward, 4),
                    "exec_prices": ctx.env._exec_prices,
                    "exec_quantities": ctx.env._exec_quantities,
                    **metrics,
                }
            ),
        }

    return EventSourceResponse(event_stream())


@app.get("/api/policies")
def get_policies():
    # naive_twap and twap are always available (pure simulator policies, no model/key).
    available = ["naive_twap", "twap"]
    if _get_ppo_model() is not None:
        available.append("ppo")
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("llm")
    if os.environ.get("FIREWORKS_API_KEY"):
        available.append("fireworks")
    labels = {p: POLICY_LABELS.get(p, p.upper()) for p in available}
    return {"available": available, "labels": labels}


@app.get("/api/compare")
def compare_episode(
    policies: str = Query("ppo,naive_twap", description="Comma-separated policy ids to race"),
    ticker: str = Query("AAPL"),
    side: Literal["buy", "sell"] = Query("buy"),
    total_shares: int | None = Query(None),
    capital_usd: float | None = Query(None),
    adv_pct: float | None = Query(None, description="Order size as a percentage of ADV"),
    n_slices: int = Query(_N_SLICES),
    date: str | None = Query(None, description="Historical replay date (YYYY-MM-DD)"),
    regime: str | None = Query(None, description="Market regime sample when date is omitted"),
    seed: int | None = Query(None, description="Seed for reproducible day sampling"),
):
    """Run several policies on the *identical* scenario and price path, so the only
    variable is the policy. This is the apples-to-apples proof: same ticker, same day,
    same seed -> same intraday path (the simulator's path RNG is seeded), so any
    difference in execution quality is attributable to the policy alone.

    Returns the shared scenario (path, volume_curve, meta) once, plus a per-policy
    result (exec trace + metrics). The frontend can then animate all policies racing the
    same path and read off the bps / dollar advantage of the RL agent over the baseline.
    """
    policy_ids = [p.strip() for p in policies.split(",") if p.strip()]
    if not policy_ids:
        raise HTTPException(400, "policies must contain at least one policy id")

    # Pin a seed so every policy samples the exact same day and price path. Without
    # this, a 'random' day request would draw a different day per policy.
    if seed is None:
        seed = int(np.random.default_rng().integers(1_000_000))

    shared_meta: dict | None = None
    results = []
    for policy in policy_ids:
        ctx = _build_context(policy, ticker, side, total_shares, capital_usd, n_slices, date, regime, seed, adv_pct)

        obs = ctx.env._build_obs()
        total_reward = 0.0
        for i in range(ctx.n_slices):
            action = ctx.action_for(i, obs)
            obs, reward, terminated, truncated, _ = ctx.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        meta = ctx.meta()
        if shared_meta is None:
            # Scenario-level fields are identical across policies; capture once.
            shared_meta = {
                k: meta[k]
                for k in (
                    "ticker", "side", "date", "total_shares", "notional_usd",
                    "order_adv_pct", "n_slices", "open_price", "close_price", "adv",
                    "data_source", "seed", "regime", "path", "volume_curve",
                )
            }

        metrics = _episode_metrics(ctx.env)
        results.append(
            {
                "policy": policy,
                "label": POLICY_LABELS.get(policy, policy.upper()),
                "schedule_label": ctx.schedule_label,
                "exec_prices": ctx.env._exec_prices,
                "exec_quantities": ctx.env._exec_quantities,
                "final_reward": round(total_reward, 4),
                "llm_reasoning": meta.get("llm_reasoning", ""),
                "llm_primitive": meta.get("llm_primitive", ""),
                **metrics,
            }
        )

    return {**shared_meta, "policies": results}


@app.get("/api/eval/stream")
async def eval_stream(
    policy: Literal["naive_twap", "twap", "ppo"] = Query("ppo"),
    baseline: Literal["naive_twap", "twap"] = Query("naive_twap"),
    ticker: str = Query("AAPL"),
    side: Literal["buy", "sell"] = Query("buy"),
    total_shares: int | None = Query(None),
    adv_pct: float | None = Query(None, description="Order size as a percentage of ADV"),
    n_slices: int = Query(_N_SLICES),
    n_episodes: int = Query(50, ge=5, le=300),
):
    """Batch evaluation over chronologically held-out days the model never trained on
    (the (0.8, 1.0) split), streaming per-episode results so the frontend can build a
    distribution live, then a final summary with aggregate statistics.

    Each held-out day is run with both `policy` and `baseline` on the *same* seed, so
    they see the identical price path -- a paired comparison. This is the statistical
    backbone of the 'we actually improve execution' claim: win-rate and mean advantage
    over many unseen days, not a single cherry-picked episode.
    """
    import json

    if policy == "ppo" and n_slices != _N_SLICES:
        async def slice_err():
            yield {"event": "error", "data": json.dumps({"detail": f"PPO requires n_slices={_N_SLICES}"})}
        return EventSourceResponse(slice_err())

    ppo_model = None
    if "ppo" in (policy, baseline):
        ppo_model = _get_ppo_model()
        if ppo_model is None:
            message = _ppo_model_unavailable_message()

            async def model_err():
                yield {"event": "error", "data": json.dumps({"detail": message})}
            return EventSourceResponse(model_err())

    try:
        ticker_df = ensure_daily_data(ticker)
    except ValueError as exc:
        async def ticker_err():
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
        return EventSourceResponse(ticker_err())

    if adv_pct is not None:
        ref_adv = float(ticker_df["Volume"].median())
        shares = shares_from_adv_pct(adv_pct, ref_adv)
    else:
        shares = total_shares if total_shares is not None else _TOTAL_SHARES

    def _run(env: ExecutionEnv, which: str, seed: int) -> dict:
        env.reset(seed=seed, options={"ticker": ticker})
        obs = env._build_obs()
        for i in range(env._n_slices):
            action = _stateless_action(which, env, i, obs, ppo_model)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        return _episode_metrics(env)

    async def event_stream():
        # Two independent envs on the holdout split, pinned to the requested ticker.
        env_policy = ExecutionEnv(
            n_slices=n_slices, total_shares=shares, side=side,
            day_range=_HOLDOUT_DAY_RANGE, tickers=[ticker],
        )
        env_baseline = ExecutionEnv(
            n_slices=n_slices, total_shares=shares, side=side,
            day_range=_HOLDOUT_DAY_RANGE, tickers=[ticker],
        )

        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "policy": policy,
                    "baseline": baseline,
                    "policy_label": POLICY_LABELS.get(policy, policy.upper()),
                    "baseline_label": POLICY_LABELS.get(baseline, baseline.upper()),
                    "ticker": ticker,
                    "side": side,
                    "total_shares": shares,
                    "adv_pct": adv_pct,
                    "n_slices": n_slices,
                    "n_episodes": n_episodes,
                    "split": "held-out (last 20% of history, unseen in training)",
                }
            ),
        }

        wins = 0
        notional_sum = 0.0
        policy_bps_list: list[float] = []
        baseline_bps_list: list[float] = []
        base_seed = 10_000
        for k in range(n_episodes):
            seed = base_seed + k
            p_metrics = _run(env_policy, policy, seed)
            b_metrics = _run(env_baseline, baseline, seed)
            p_bps = p_metrics["slippage_bps"]
            b_bps = b_metrics["slippage_bps"]
            advantage = round(p_bps - b_bps, 2)
            policy_bps_list.append(p_bps)
            baseline_bps_list.append(b_bps)
            if advantage > 0:
                wins += 1
            notional_sum += float(env_policy._path[0]) * shares

            yield {
                "event": "episode",
                "data": json.dumps(
                    {
                        "i": k,
                        "policy_bps": p_bps,
                        "baseline_bps": b_bps,
                        "advantage_bps": advantage,
                    }
                ),
            }
            # Yield control so the client renders progressively.
            await asyncio.sleep(0)

        p_arr = np.array(policy_bps_list)
        b_arr = np.array(baseline_bps_list)
        adv = p_arr - b_arr
        mean_notional = notional_sum / max(1, n_episodes)
        # Dollar advantage per order = mean bps advantage * mean notional.
        usd_per_order = float(adv.mean() / 10_000 * mean_notional)
        # Paired t-like effect: mean / standard error.
        se = float(adv.std(ddof=1) / np.sqrt(len(adv))) if len(adv) > 1 else 0.0
        t_stat = float(adv.mean() / se) if se > 0 else 0.0

        yield {
            "event": "summary",
            "data": json.dumps(
                {
                    "n_episodes": n_episodes,
                    "policy_mean_bps": round(float(p_arr.mean()), 2),
                    "policy_std_bps": round(float(p_arr.std()), 2),
                    "baseline_mean_bps": round(float(b_arr.mean()), 2),
                    "baseline_std_bps": round(float(b_arr.std()), 2),
                    "mean_advantage_bps": round(float(adv.mean()), 2),
                    "median_advantage_bps": round(float(np.median(adv)), 2),
                    "win_rate": round(wins / max(1, n_episodes), 4),
                    "mean_notional_usd": round(mean_notional, 2),
                    "usd_saved_per_order": round(usd_per_order, 2),
                    "t_stat": round(t_stat, 2),
                }
            ),
        }

    return EventSourceResponse(event_stream())


# ---------------------------------------------------------------------------
# RFT live eval -- a real LLM (HUD-gateway, Tinker-hosted) graded per episode, not a
# deterministic in-process policy like the ones above. Each episode is two live tool-
# calling rollouts (base zero-shot model + the trained RFT checkpoint), so this is slow
# (~10-30s per rollout, observed) and costs real inference -- capped tightly below.
# ---------------------------------------------------------------------------

_RFT_BASE_MODEL = "Qwen/Qwen3-8B"  # pristine, un-forked -- the zero-shot comparison point
_RFT_BASE_SEED = 77_000  # distinct from /api/eval/stream's 10_000 and training's 20260621 pool


@app.get("/api/rft/stream")
async def rft_stream(
    ticker: str = Query("AAPL"),
    side: Literal["buy", "sell"] = Query("buy"),
    adv_pct: float = Query(8.0, ge=1.0, le=20.0),
    n_episodes: int = Query(8, ge=1, le=8),
):
    """Live, re-runnable version of the static RFT panel on the Proof page: streams one
    `episode` event per (base, rft) paired rollout as each completes, then a `summary`
    event with the aggregate stats -- same shape as /api/eval/stream, but each episode is
    a real LLM call through HUD's gateway instead of an instant local policy step.
    """
    import json

    import hud.agents as hud_agents
    from hud.eval import LocalRuntime
    from hud.eval import rollout as hud_rollout

    from execution_env.env import execution_fixed
    from execution_env.rl.hud_rft_pipeline import MODEL as _RFT_MODEL, _COMPLETION_KWARGS, _ENV_PY, _PROMPT

    try:
        ticker_df = ensure_daily_data(ticker)
    except ValueError as exc:
        async def ticker_err():
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
        return EventSourceResponse(ticker_err())

    ref_adv = float(ticker_df["Volume"].median())
    shares = shares_from_adv_pct(adv_pct, ref_adv)
    runtime = LocalRuntime(_ENV_PY)

    async def run_one(model: str, seed: int) -> float:
        task = execution_fixed(prompt=_PROMPT, ticker=ticker, side=side, total_shares=shares, seed=seed)
        agent = hud_agents.create_agent(model, completion_kwargs=_COMPLETION_KWARGS)
        run = await hud_rollout(task, agent, runtime=runtime)
        return float(run.reward)

    async def event_stream():
        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "model": _RFT_MODEL,
                    "baseline": "zero-shot " + _RFT_BASE_MODEL,
                    "ticker": ticker,
                    "side": side,
                    "total_shares": shares,
                    "adv_pct": adv_pct,
                    "n_episodes": n_episodes,
                }
            ),
        }

        deltas: list[float] = []
        for k in range(n_episodes):
            seed = _RFT_BASE_SEED + k
            try:
                base_reward = await run_one(_RFT_BASE_MODEL, seed)
                rft_reward = await run_one(_RFT_MODEL, seed)
            except Exception as exc:
                yield {"event": "episode", "data": json.dumps({"i": k, "error": str(exc)})}
                continue

            delta = rft_reward - base_reward
            deltas.append(delta)
            yield {
                "event": "episode",
                "data": json.dumps(
                    {"i": k, "base_reward": round(base_reward, 4), "rft_reward": round(rft_reward, 4), "delta": round(delta, 4)}
                ),
            }
            await asyncio.sleep(0)

        if deltas:
            arr = np.array(deltas)
            wins = int((arr > 0).sum())
            ties = int((arr == 0).sum())
            losses = int((arr < 0).sum())
            se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            t_stat = float(arr.mean() / se) if se > 0 else 0.0
            yield {
                "event": "summary",
                "data": json.dumps(
                    {
                        "n": len(arr),
                        "mean_delta": round(float(arr.mean()), 4),
                        "wins": wins,
                        "ties": ties,
                        "losses": losses,
                        "t_stat": round(t_stat, 3),
                        "is_significant": bool(abs(t_stat) > 1.96),
                    }
                ),
            }
        else:
            yield {"event": "summary", "data": json.dumps({"n": 0, "error": "all episodes failed"})}

    return EventSourceResponse(event_stream())
