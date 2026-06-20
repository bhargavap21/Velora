"""
HUD v6 environment definition for the optimal-execution task.

TODO(env.py): the decorator/import API here (@env.initialize, @env.template) is written
from the *README descriptions* of the hud-evals template repos (autonomous-businesses-
template, robotics-template), not from their verified source. Before relying on this,
clone one template locally and diff its actual env.py against this file -- the decorator
names or signatures may not match exactly:

    git clone https://github.com/hud-evals/autonomous-businesses-template /tmp/hud-ref

This file currently wires the simulator into a single-shot scenario: the agent gets the
task prompt + initial market state, proposes a full execution schedule in one response,
and is graded on the resulting slippage. A multi-turn version (re-prompting per slice)
is possible but spends more LLM calls per episode -- decide which based on how the
grading rubric in tasks.py ends up shaped.
"""

from __future__ import annotations

import numpy as np

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES, _TOTAL_SHARES

# from hud import Environment  # TODO: confirm against a real template's env.py
# env = Environment("velora-execution")


def _build_prompt(total_shares: int, n_slices: int, ticker: str) -> str:
    return f"""You are an execution trader. Your task: buy {total_shares} shares of {ticker}
over {n_slices} time slices today without moving the price against yourself more than
necessary.

Propose a full execution schedule as JSON: a list of {n_slices} participation
multipliers (each in [0, 2]), where each multiplier scales the volume-curve-implied
"fair share" of remaining inventory to trade in that slice -- a multiplier of 1.0 means
"trade at exactly the VWAP-implied rate," below 1.0 trades less than that, above 1.0
trades more. The order must be fully filled by the end.

You will be scored on slippage vs. VWAP (lower is better) plus a penalty for any
unfilled inventory.

Return ONLY a JSON object: {{"schedule": [m0, m1, ..., m{n_slices - 1}], "reasoning": "..."}}"""


def run_scenario(schedule: list[float], total_shares: int = _TOTAL_SHARES, n_slices: int = _N_SLICES) -> dict:
    """Replays a proposed schedule through ExecutionEnv and returns the scored result.

    TODO(env.py): wrap this in the actual HUD @env.template/@env.scenario decorator once
    the v6 API is confirmed (see module docstring).
    """
    env = ExecutionEnv(n_slices=n_slices, total_shares=total_shares)
    obs, _ = env.reset()

    total_reward = 0.0
    for i in range(n_slices):
        multiplier = schedule[i] if i < len(schedule) else 2.0  # force max participation on schedule underrun
        obs, reward, terminated, truncated, _ = env.step(np.array([multiplier], dtype=np.float32))
        total_reward += reward
        if terminated:
            break

    return {"reward": total_reward, "done": True}


if __name__ == "__main__":
    vwap_schedule = [1.0] * _N_SLICES  # VWAP-matching baseline: participation = 1.0 every slice
    result = run_scenario(vwap_schedule)
    print("VWAP-matching schedule result:", result)
