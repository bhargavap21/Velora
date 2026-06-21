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

Backend/model decision -- RESOLVED (Path A, HUD-native): forked a team-owned trainable
copy of Qwen3 8B (Tinker) via `hud models fork`, named `velora-execution-rft-qwen3-8b`.
Chosen over DeepSeek V3.1/Qwen3-30B/GPT-OSS-20B (also available, Trainable=true on this
account) for being the cheapest/fastest to iterate on for a hackathon timeline -- if RFT
can beat zero-shot's 0.0 bps baseline at all, an 8B model should show it fastest. Verified
real and trainable: `TrainingClient("velora-execution-rft-qwen3-8b").available_losses()`
-> ['cispo', 'cross_entropy', 'dro', 'importance_sampling', 'ppo'].

Usage:
    .venv/bin/python -m execution_env.rl.hud_rft_pipeline   # prints the plan only.
    # To actually collect a group (real LLM inference cost via HUD):
    #   import asyncio; from execution_env.rl.hud_rft_pipeline import collect_group
    #   runs = asyncio.run(collect_group(scenario, model=MODEL))
    # Real training (TrainingClient.step) is intentionally not wired up here -- costs real
    # training compute/credits, gated on validating collect_group() end-to-end first.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

MODEL = "velora-execution-rft-qwen3-8b"  # forked, team-owned trainable copy of Qwen3 8B (Tinker)
GROUP_SIZE = 8  # rollouts per GRPO group -- a starting guess, not tuned
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_PY = _REPO_ROOT / "execution_env" / "env.py"

_PROMPT = (
    "You are an institutional execution trader. Minimize VWAP slippage on the order "
    "described in the task. Use read_market_context() to inspect the market, then "
    "submit_schedule() with your full participation plan before the session ends."
)


# Qwen3 8B (Tinker) is a thinking model that burns a large completion budget reasoning
# through the volume curve before acting. Two distinct, independently-confirmed failure
# modes were traced via a tool-call-argument-logging debug harness (3 sequential rollouts,
# same pinned ROKU scenario):
#   1. Token-budget truncation mid-reasoning: the model reasons *correctly* (one run
#      derived "multiplier = 1.0 for every slice", which is right since the volume curve
#      already sums to 1.0) but runs out of tokens before emitting submit_schedule ->
#      reward floor. This is what max_tokens addresses; bumped 4096 -> 8192 to give the
#      reasoning room to finish and still call the tool.
#   2. Literal volume-curve echo: the model submits the raw volume_curve array verbatim
#      (down to the exact float 0.14905090274683408) as the schedule, despite an explicit
#      prompt warning against it -- reproduced bit-for-bit across independent rollouts, so
#      it's the model's deterministic default, not sampling noise. Two prompt rewrites did
#      NOT fix it; this is a reasoning-depth limitation of the 8B model and is exactly the
#      structured failure RFT training is meant to correct, not a prompting problem.
# enable_thinking=False is the standard vLLM/SGLang chat_template_kwargs toggle for Qwen3's
# thinking mode (passthrough extra_body, not a HUD-specific guarantee).
#
# return_token_ids=True is REQUIRED for TrainingClient.step() to work at all -- without it
# the gateway never attaches token ids/logprobs to the response, so no AgentStep.sample ever
# gets output_token_ids populated (confirmed: every run showed n_with_output_tokens=0) and
# the server has nothing to resolve a trace_id's turns against either, failing every
# forward_backward call with "no trainable turns in the provided inputs" regardless of
# reward or group composition. Found via hud-python's own cookbooks/rl-training/simple_train.py
# reference example -- not documented in the TrainingClient docstring itself.
_COMPLETION_KWARGS = {
    "max_tokens": 8192,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": False}, "return_token_ids": True},
}


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
    agent = hud_agents.create_agent(model, completion_kwargs=_COMPLETION_KWARGS)
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
        f"\nModel decision made: {MODEL} (forked Qwen3 8B (Tinker), available_losses "
        "verified real).\n"
        "Next steps once ready to spend real cost:\n"
        f"  1. runs = await collect_group(scenario, model=MODEL) -- one real group "
        f"({GROUP_SIZE} rollouts), real LLM inference cost via HUD.\n"
        "  2. client.step(runs, learning_rate=..., loss_fn='importance_sampling', "
        f"group_size={GROUP_SIZE}) -- a real training call, not wired up here.\n"
        "  3. Re-eval the new checkpoint (client.head()) against "
        "sanity_eval_random_task.py's task distribution; compare to the base-model gap "
        "already recorded there."
    )


if __name__ == "__main__":
    describe_pipeline()
