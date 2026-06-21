"""Run the trained PPO policy as an agent against the Velora HUD environment.

The HUD env (../env.py) is an MCP environment: an agent calls read_market_context(),
then submit_schedule() with a list of per-slice participation multipliers in [0, 2].
PPO's action space *is* that multiplier, so we drive it by reconstructing the same
deterministic episode the grader will replay (identical ticker/side/size/seed -> identical
price path), rolling the policy out, and submitting the multipliers it chooses. Because the
episode is deterministic and PPO is evaluated deterministically, the submitted static
schedule reproduces PPO's exact trajectory when the grader replays it.

Usage (produces a real hud.ai job; needs HUD_API_KEY):
    PYTHONPATH=. .venv/bin/python -m execution_env.rl.hud_ppo_agent
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from hud.agents.base import Agent
from hud.eval import LocalRuntime, Taskset

from execution_env.rl.execution_gym_env import ExecutionEnv
from execution_env.rl.train_ppo import MODEL_PATH

# The HUD templates call _run_execution_template with the default seed=42 (see env.py),
# so every task's grading episode is pinned to this seed. We mirror it exactly.
_TASK_SEED = 42

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASKS_PY = _REPO_ROOT / "execution_env" / "tasks.py"
_ENV_PY = _REPO_ROOT / "execution_env" / "env.py"


def _parse_context(result) -> dict:
    """Pull the market-context dict out of an MCP tool result, tolerating the
    structured-content / text-content shape differences across MCP backends."""
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        if "ticker" in sc:
            return sc
        if isinstance(sc.get("result"), dict) and "ticker" in sc["result"]:
            return sc["result"]
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "ticker" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    raise RuntimeError(f"Could not parse market context from tool result: {result!r}")


class PPOScheduleAgent(Agent):
    """Drives the HUD run by rolling out the trained PPO policy to a schedule."""

    def __init__(self, model: PPO, seed: int = _TASK_SEED) -> None:
        self.model = model
        self.seed = seed

    async def __call__(self, run) -> None:  # noqa: ANN001 (hud.eval.run.Run)
        mcp = await run.client.open("mcp")
        ctx = _parse_context(await mcp.call_tool("read_market_context", {}))

        env = ExecutionEnv(
            total_shares=int(ctx["total_shares"]),
            side=ctx["side"],
            tickers=[ctx["ticker"]],
        )
        obs, _ = env.reset(seed=self.seed)

        schedule: list[float] = []
        for _ in range(env._n_slices):
            action, _ = self.model.predict(obs, deterministic=True)
            schedule.append(round(float(np.clip(action[0], 0.0, 2.0)), 4))
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        payload = json.dumps(
            {
                "schedule": schedule,
                "reasoning": (
                    f"Deterministic PPO rollout for {ctx['side']} {ctx['ticker']} "
                    f"over {len(schedule)} slices."
                ),
            }
        )
        await mcp.call_tool("submit_schedule", {"schedule_json": payload})
        run.trace.content = payload


async def main() -> None:
    model = PPO.load(MODEL_PATH)
    agent = PPOScheduleAgent(model)

    taskset = Taskset.from_file(_TASKS_PY)
    runtime = LocalRuntime(_ENV_PY)

    print(f"Running {len(taskset)} HUD tasks with the PPO agent...")
    job = await taskset.run(agent, runtime=runtime)

    print("\n=== HUD job ===")
    for attr in ("id", "name", "url"):
        if hasattr(job, attr):
            print(f"{attr}: {getattr(job, attr)}")
    runs = getattr(job, "runs", None) or getattr(job, "results", None) or []
    for r in runs:
        slug = getattr(r, "task_id", getattr(r, "id", "?"))
        grade = getattr(r, "grade", None)
        reward = getattr(grade, "reward", None) if grade is not None else getattr(r, "reward", None)
        print(f"  {slug}: reward={reward}")
    print("\nRaw job:", repr(job))


if __name__ == "__main__":
    asyncio.run(main())
