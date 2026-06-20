"""
HTTP API for the optimal-execution simulator, run live on request rather than replayed
from a static fixture.

GET /api/episode runs one fresh ExecutionEnv episode server-side with the requested
policy and returns its trace (price path, volume curve, per-slice fills) for the
frontend to animate.

Usage:
    uvicorn execution_env.server:app --reload --port 8000
"""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES, _TICKERS, _TOTAL_SHARES
from execution_env.rl.train_ppo import MODEL_PATH, twap_action

app = FastAPI(title="Velora execution API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
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


@app.get("/api/episode")
def get_episode(
    policy: Literal["twap", "ppo"] = Query("twap"),
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

    env = ExecutionEnv(
        n_slices=_N_SLICES,
        total_shares=_TOTAL_SHARES,
        side=side,
        tickers=[ticker] if ticker else None,
    )
    obs, _ = env.reset()

    total_reward = 0.0
    i = 0
    while True:
        if policy == "ppo":
            action, _ = ppo_model.predict(obs, deterministic=True)
        else:
            action = twap_action(i, _N_SLICES)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        i += 1
        if terminated or truncated:
            break

    filled_fraction = 1.0 - env._shares_remaining / env._total_shares

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
        "schedule_label": "PPO (trained)" if policy == "ppo" else "VWAP-matching (participation = 1.0)",
        "policy": policy,
    }


@app.get("/api/policies")
def get_policies():
    return {"available": ["twap"] + (["ppo"] if MODEL_PATH.exists() else [])}
