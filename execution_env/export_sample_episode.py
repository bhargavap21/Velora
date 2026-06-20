"""
Runs one real episode through ExecutionEnv with a corrected TWAP schedule (equal
absolute quantity per slice, not equal fraction-of-remaining -- see the bug noted in
execution_env/README.md) and exports the raw trace as JSON for the frontend to animate.

This is a stopgap until the real backend SSE endpoint (issue #8) exists -- the frontend
plays this fixture back client-side at intervals to preview the live-execution chart
against real simulator output, not fabricated numbers.

Usage:
    python -m execution_env.export_sample_episode
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES, _TOTAL_SHARES

_OUT_PATH = Path(__file__).parent.parent / "frontend" / "src" / "data" / "sample_episode.json"


def corrected_twap_schedule(n_slices: int) -> list[float]:
    """Equal absolute quantity per slice: f_i = 1 / (n_slices - i) of whatever remains."""
    return [1.0 / (n_slices - i) for i in range(n_slices)]


def main() -> None:
    env = ExecutionEnv(n_slices=_N_SLICES, total_shares=_TOTAL_SHARES, side="buy")
    obs, _ = env.reset(seed=42)

    schedule = corrected_twap_schedule(_N_SLICES)
    total_reward = 0.0
    for frac in schedule:
        obs, reward, terminated, truncated, _ = env.step(np.array([frac], dtype=np.float32))
        total_reward += reward
        if terminated:
            break

    filled_fraction = 1.0 - env._shares_remaining / env._total_shares

    payload = {
        "ticker": "AAPL",
        "side": env._side,
        "total_shares": env._total_shares,
        "n_slices": env._n_slices,
        "path": env._path.tolist(),
        "volume_curve": env._volume_curve.tolist(),
        "exec_prices": env._exec_prices,
        "exec_quantities": env._exec_quantities,
        "final_reward": round(total_reward, 4),
        "filled_fraction": round(filled_fraction, 4),
        "schedule_label": "corrected TWAP (equal absolute quantity per slice)",
    }

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {_OUT_PATH}")
    print(f"Filled fraction: {filled_fraction:.4f}  |  Total reward: {total_reward:.4f}")


if __name__ == "__main__":
    main()
