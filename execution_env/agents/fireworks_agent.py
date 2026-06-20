"""
Open-source-model execution-schedule agent via Fireworks AI.

Uses the same schedule-primitive approach as llm_agent.py (front_load, back_load,
follow_volume_curve, twap) but calls a Fireworks-hosted open-weight model (OpenAI's
gpt-oss-120b) via their OpenAI-compatible chat completions endpoint instead of the
Anthropic SDK -- the open-source comparison arm against the closed Claude agent.

Requires FIREWORKS_API_KEY in the environment.
"""

from __future__ import annotations

import os

import numpy as np
from openai import OpenAI

from execution_env.agents.llm_agent import (
    build_observation,
    schedule_from_response,
    twap,
    _SYSTEM_PROMPT,
)

_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"
# OpenAI's open-weight model on Fireworks serverless -- the open-source comparison arm
# against the closed Claude agent. gpt-oss emits chain-of-thought into a separate
# `reasoning_content` field and the final answer into `content`, so we give it enough
# token budget to finish reasoning *and* emit the JSON.
_MODEL = "accounts/fireworks/models/gpt-oss-120b"
_MAX_TOKENS = 2048


def _get_client() -> OpenAI:
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise EnvironmentError("FIREWORKS_API_KEY not set in environment.")
    return OpenAI(api_key=api_key, base_url=_FIREWORKS_BASE_URL)


def propose_schedule(
    ticker: str,
    total_shares: int,
    n_slices: int,
    open_price: float,
    volume_curve: list[float],
) -> tuple[list[float], dict, str]:
    """Call a Fireworks-hosted Llama model once at episode start to choose a schedule.

    Returns (multipliers, parsed_json, raw_response_text). Same signature as
    llm_agent.propose_schedule so callers can swap agents without changing call sites.
    """
    client = _get_client()
    user_message = build_observation(ticker, total_shares, n_slices, open_price, volume_curve)

    response = client.chat.completions.create(
        model=_MODEL,
        max_tokens=_MAX_TOKENS,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    message = response.choices[0].message
    # gpt-oss puts the answer in `content`; some reasoning models leave `content`
    # empty and emit everything (including the JSON) into `reasoning_content`. Fall
    # back to the latter so schedule_from_response can still find the JSON object.
    raw = message.content or getattr(message, "reasoning_content", None) or ""
    multipliers, parsed = schedule_from_response(raw, n_slices, np.asarray(volume_curve, dtype=float))
    return multipliers, parsed, raw


def run_episode(env, *, verbose: bool = False) -> dict:
    """Reset env, call the Fireworks Llama model once for a schedule, replay slice-by-slice.

    Same interface as llm_agent.run_episode so the two agents can be compared directly.
    Falls back to TWAP on any API or parse error.
    """
    obs, info = env.reset()
    ticker: str = info["ticker"]
    open_price: float = info["open_price"]
    volume_curve: list[float] = info["volume_curve"]
    n_slices: int = info["n_slices"]
    total_shares: int = info["total_shares"]

    try:
        multipliers, parsed, raw_response = propose_schedule(
            ticker, total_shares, n_slices, open_price, volume_curve
        )
    except Exception as exc:
        multipliers = twap(n_slices, np.asarray(volume_curve, dtype=float))
        parsed = {"primitive": "twap", "reasoning": f"fallback: {exc}"}
        raw_response = ""

    if verbose:
        print(
            f"[{ticker}] primitive={parsed.get('primitive')}  "
            f"pct={parsed.get('pct', '—')}"
        )
        print(f"  reasoning: {parsed.get('reasoning', '')}")

    pause_cfg: dict = parsed.get("pause_if_adverse_move", {"enabled": False})
    pause_enabled: bool = bool(pause_cfg.get("enabled", False))
    pause_threshold_bps: float = float(pause_cfg.get("threshold_bps", 0))

    total_reward = 0.0
    current_price: float = open_price
    last_exec_price: float | None = None

    for i in range(n_slices):
        if i > 0:
            current_price *= 1.0 + float(obs[2])

        multiplier = multipliers[i]

        if i < n_slices - 1 and pause_enabled and last_exec_price is not None and pause_threshold_bps > 0:
            price_move_bps = (current_price - last_exec_price) / last_exec_price * 10_000
            if price_move_bps > pause_threshold_bps:
                multiplier = 0.0

        obs, reward, terminated, truncated, _ = env.step(
            np.array([multiplier], dtype=np.float32)
        )
        total_reward += reward

        if multiplier > 0:
            last_exec_price = current_price

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "ticker": ticker,
        "open_price": open_price,
        "primitive": parsed.get("primitive"),
        "pct": parsed.get("pct"),
        "pause_enabled": pause_enabled,
        "reasoning": parsed.get("reasoning", ""),
        "raw_response": raw_response,
    }
