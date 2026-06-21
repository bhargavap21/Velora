"""
Scaffold for issue #15 Path A (HUD-native GRPO training). `collect_group()` below is
real, runnable code -- not invoked by this module's __main__, since it drives a live LLM
agent through HUD (real inference cost) and I'm holding off running it myself until
explicitly asked, same as the training-call boundary below.

What's real, found by inspecting the installed hud-python (v0.6.6), correcting issue
#15's assumption that Path A is GUI-only ("no `hud train` CLI; training is GUI-driven"):

  hud.agents.create_agent(model: str, **kwargs) -> GatewayAgent
      The same built-in agent `hud eval tasks.py claude` uses -- no custom Agent
      subclass needed (contrast hud_ppo_agent.py's PPOScheduleAgent, which exists only
      because PPO isn't a tool-calling LLM).

  hud.eval.rollout(task, agent, *, runtime, group_id=None, ...) -> Run
      Runs one rollout of `agent` against `task` through the real MCP env (the same
      read_market_context -> submit_schedule flow). `group_id` is the GRPO grouping
      primitive: multiple rollouts sharing a group_id are the "N samples of the same
      prompt" GRPO needs to compute a relative advantage.

  hud.train.TrainingClient(model: str, *, api_key=None, base_url=None)
      .step(trajectories: Sequence[str | Run | TrajectoryPayload], *, learning_rate,
            loss_fn='importance_sampling', group_size=None, reward_scale=1.0, ...)
      Takes Run objects (rollout() output) DIRECTLY -- no manual TrainingDatum assembly
      needed for the common case. `available_losses()` reports what's enabled for the
      target model. `.forward`/`.forward_backward`/`.optim_step` are the lower-level
      Tinker-style split `.step()` composes. `.checkpoints()`/`.set_head()`/`.head()`
      manage trained checkpoints.

GRPO grouping design question -- RESOLVED: env.py::execution_random deliberately
resamples a fresh scenario every invocation (issue #15's "fully dynamic" requirement),
which can't form a GRPO group (N rollouts of the SAME prompt). env.py::execution_fixed
takes the scenario as explicit args instead -- build N Task instances with identical
args (see collect_group() below) to form one group.

Usage:
    .venv/bin/python -m execution_env.rl.hud_rft_pipeline   # prints the plan only.
    # To actually collect a group (real LLM inference cost via HUD):
    #   import asyncio; from execution_env.rl.hud_rft_pipeline import collect_group
    #   runs = asyncio.run(collect_group(scenario, model="claude-sonnet-4-6"))
    # Real training (TrainingClient.step) is intentionally not wired up here -- needs an
    # explicit model-selection decision and costs real training compute/credits.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

GROUP_SIZE = 8  # rollouts per GRPO group -- a starting guess, not tuned
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PY = _REPO_ROOT / "execution_env" / "env.py"

_PROMPT = (
    "You are an institutional execution trader. Minimize VWAP slippage on the order "
    "described in the task. Use read_market_context() to inspect the market, then "
    "submit_schedule() with your full participation plan before the session ends."
)


async def collect_group(scenario: dict[str, Any], model: str, group_size: int = GROUP_SIZE) -> list[Any]:
    """group_size rollouts of the SAME pinned scenario (via execution_fixed), using
    HUD's built-in agent for `model`. Returns the Run list -- exactly what
    TrainingClient.step() consumes directly, no further assembly needed.

    Real LLM inference cost per call (group_size x one agent rollout through HUD) -- not
    a training call, but not free either. Intentionally not invoked anywhere in this
    module; call it yourself once ready to spend that cost.
    """
    import hud.agents as hud_agents
    import hud.eval as hud_eval
    from hud.eval import LocalRuntime

    from execution_env.env import execution_fixed

    task = execution_fixed(
        prompt=_PROMPT,
        ticker=scenario["ticker"],
        side=scenario["side"],
        total_shares=scenario["total_shares"],
        seed=scenario["seed"],
    )
    agent = hud_agents.create_agent(model)
    runtime = LocalRuntime(_ENV_PY)
    group_id = str(uuid.uuid4())

    runs = []
    for _ in range(group_size):
        run = await hud_eval.rollout(task, agent, runtime=runtime, group_id=group_id)
        runs.append(run)
    return runs


def describe_pipeline() -> None:
    print(__doc__)
    print(
        "\nNext steps once ready to spend real cost:\n"
        "  1. Pick a target model + confirm it's enabled for HUD training "
        "(TrainingClient(model).available_losses()).\n"
        "  2. runs = await collect_group(scenario, model=...) -- one real group "
        f"({GROUP_SIZE} rollouts), real LLM inference cost via HUD.\n"
        "  3. client.step(runs, learning_rate=..., loss_fn='importance_sampling', "
        f"group_size={GROUP_SIZE}) -- a real training call, not wired up here.\n"
        "  4. Re-eval the new checkpoint (client.head()) against "
        "sanity_eval_random_task.py's task distribution; compare to the base-model gap "
        "already recorded there."
    )


if __name__ == "__main__":
    describe_pipeline()
