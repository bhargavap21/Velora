"""
HTTP API for the optimal-execution simulator, run live on request rather than replayed
from a static fixture.

GET /api/episode runs one fresh ExecutionEnv episode server-side with the requested
policy and returns its trace (price path, volume curve, per-slice fills) for the
frontend to animate.

Supported policies:
  - twap: equal quantity every slice (baseline)
  - ppo:  trained PPO checkpoint (requires prior run of train_ppo.py)
  - llm:  Claude sets a coarse execution schedule once at episode start
          (requires ANTHROPIC_API_KEY in environment)

Usage:
    uvicorn execution_env.server:app --reload --port 8000
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES, _TICKERS, _TOTAL_SHARES
from execution_env.rl.train_ppo import MODEL_PATH, twap_action

app = FastAPI(title="Velora execution API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3002"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_ppo_model = None
_ppo_load_attempted = False


def _get_ppo_model():
    """Lazily loads the PPO checkpoint on first request; cached after that. Returns
    None if no checkpoint has been trained yet (see train_ppo.py)."""
    global _ppo_model, _ppo_load_attempted
    if not _ppo_load_attempted:
        _ppo_load_attempted = True
        if MODEL_PATH.exists():
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


@app.get("/api/episode")
def get_episode(
    policy: Literal["twap", "ppo", "llm", "fireworks"] = Query("twap"),
    ticker: str | None = Query(None),
    side: Literal["buy", "sell"] = Query("buy"),
):
    if ticker is not None and ticker not in _TICKERS:
        raise HTTPException(400, f"ticker must be one of {_TICKERS}")

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

    env = ExecutionEnv(
        n_slices=_N_SLICES,
        total_shares=_TOTAL_SHARES,
        side=side,
        tickers=[ticker] if ticker else None,
    )
    obs, info = env.reset()

    # For LLM policy: call Claude once upfront to get the full schedule.
    # For Fireworks policy: call Llama once upfront to get the full schedule.
    llm_multipliers: list[float] = []
    llm_meta: dict = {}
    if policy in ("llm", "fireworks"):
        if policy == "llm":
            multipliers, parsed, _raw = _propose_llm_schedule(info)
        else:
            from execution_env.agents.fireworks_agent import propose_schedule as fw_propose, twap as fw_twap
            try:
                multipliers, parsed, _raw = fw_propose(
                    info["ticker"], info["total_shares"], info["n_slices"],
                    info["open_price"], info["volume_curve"],
                )
            except Exception as exc:
                multipliers = fw_twap(info["n_slices"], np.asarray(info["volume_curve"], dtype=float))
                parsed = {"primitive": "twap", "reasoning": f"API error — fell back to TWAP: {exc}"}
        llm_multipliers = multipliers
        pause_cfg = parsed.get("pause_if_adverse_move", {"enabled": False})
        llm_meta = {
            "llm_primitive": parsed.get("primitive", "unknown"),
            "llm_reasoning": parsed.get("reasoning", ""),
            "llm_pause_enabled": bool(pause_cfg.get("enabled", False)),
            "llm_pause_threshold_bps": float(pause_cfg.get("threshold_bps", 0)),
        }

    total_reward = 0.0
    current_price: float = float(info.get("open_price", 0.0))
    last_exec_price: float | None = None

    for i in range(_N_SLICES):
        if policy == "ppo":
            action, _ = ppo_model.predict(obs, deterministic=True)

        elif policy in ("llm", "fireworks"):
            if i > 0:
                # obs[2] = recent_return = (path[i] - path[i-1]) / path[i-1]
                current_price *= 1.0 + float(obs[2])

            multiplier = llm_multipliers[i]

            # Apply pause modifier: skip slice if price rose adversely since last fill.
            # Never skips the final slice so inventory always fully fills.
            if (
                i < _N_SLICES - 1
                and llm_meta["llm_pause_enabled"]
                and last_exec_price is not None
                and llm_meta["llm_pause_threshold_bps"] > 0
            ):
                price_move_bps = (current_price - last_exec_price) / last_exec_price * 10_000
                if price_move_bps > llm_meta["llm_pause_threshold_bps"]:
                    multiplier = 0.0

            if multiplier > 0:
                last_exec_price = current_price

            action = np.array([multiplier], dtype=np.float32)

        else:
            action = twap_action(i, _N_SLICES)

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    filled_fraction = 1.0 - env._shares_remaining / env._total_shares

    if policy == "llm":
        primitive = llm_meta.get("llm_primitive", "unknown")
        schedule_label = f"Claude LLM — {primitive}"
    elif policy == "fireworks":
        primitive = llm_meta.get("llm_primitive", "unknown")
        schedule_label = f"Llama (Fireworks) — {primitive}"
    elif policy == "ppo":
        schedule_label = "PPO (trained)"
    else:
        schedule_label = "TWAP (equal participation)"

    return {
        "ticker": env._ticker,
        "side": env._side,
        "total_shares": env._total_shares,
        "n_slices": env._n_slices,
        "path": env._path.tolist(),
        "volume_curve": env._volume_curve.tolist(),
        "exec_prices": env._exec_prices,
        "exec_quantities": env._exec_quantities,
        "final_reward": round(total_reward, 4),
        "filled_fraction": round(filled_fraction, 4),
        "schedule_label": schedule_label,
        "policy": policy,
        **llm_meta,
    }


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
