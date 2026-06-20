"""
HUD scenario wrapper for StratRL.

All episode logic (seeds, observation building, mutation/backtest/reward) lives in
`episode_core.py`, which has no dependency on `hud` — that keeps it importable from
the Modal remote container (which never installs hud-python) and from a `modal`-only
local venv (hud-python's a2a-sdk requires protobuf>=5.29.5, which conflicts with
modal's protobuf<5.0 — they cannot share a venv).
"""

from __future__ import annotations

import copy

from hud import Environment
from hud.tools.types import ScenarioResult

from episode_core import (
    MAX_CONSECUTIVE_INVALID,
    MAX_TURNS,
    SEED_STRATEGIES,
    build_final_observation,
    build_initial_observation,
    build_turn_observation,
    fetch_news_context,
    process_turn,
)
from backtester import run_backtest

env = Environment("optimize")


@env.scenario(chat=True)
async def optimize(messages: list | None = None):
    messages = messages or []

    initial_strategy = copy.deepcopy(SEED_STRATEGIES[0])
    initial_metrics = run_backtest(initial_strategy)
    news_context = fetch_news_context(initial_strategy["ticker"])

    current_strategy = copy.deepcopy(initial_strategy)
    current_metrics = copy.deepcopy(initial_metrics)
    history: list = []
    turn = 0
    consecutive_invalid = 0
    all_rewards: list[float] = []

    # Replay all prior assistant messages to reconstruct episode state
    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        turn += 1
        result = process_turn(
            current_strategy, current_metrics, initial_metrics,
            history, _extract_text(msg), consecutive_invalid, turn,
            news_context=news_context,
        )
        if result["valid"]:
            current_strategy = result["strategy"]
            current_metrics = result["metrics"]
            history = result["history"]
            all_rewards.append(result["reward"])
            consecutive_invalid = 0
        else:
            consecutive_invalid = result["consecutive_invalid"]

    avg_reward = round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else 0.0
    terminated = turn >= MAX_TURNS or consecutive_invalid >= MAX_CONSECUTIVE_INVALID

    if terminated:
        reason = "max turns reached" if turn >= MAX_TURNS else "too many invalid mutations"
        final_obs = build_final_observation(reason, initial_metrics, current_metrics, history)
        yield final_obs
        yield ScenarioResult(reward=avg_reward, done=True, content=final_obs)
        return

    # Yield observation for the current (upcoming) turn
    if turn == 0:
        obs = build_initial_observation(initial_strategy, initial_metrics, news_context)
    else:
        last = history[-1]
        obs = build_turn_observation(
            turn, current_strategy, current_metrics,
            last["before_metrics"], last["after_metrics"],
            last["reward"], last["mutation"], last["reasoning"], history,
            news_context=news_context,
        )

    answer = yield obs

    # Process the agent's response
    turn += 1
    result = process_turn(
        current_strategy, current_metrics, initial_metrics,
        history, str(answer) if answer else "", consecutive_invalid, turn,
        news_context=news_context,
    )

    if result["valid"]:
        all_rewards.append(result["reward"])
        avg_reward = round(sum(all_rewards) / len(all_rewards), 4)
        done = result["done"]
        reward_to_yield = avg_reward if done else result["reward"]
        yield ScenarioResult(reward=reward_to_yield, done=done, content=result["observation"])
    else:
        consecutive_invalid = result["consecutive_invalid"]
        done = result["done"]
        reward_to_yield = avg_reward if done else 0.0
        yield ScenarioResult(reward=reward_to_yield, done=done, content=result["observation"])


def _extract_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)
