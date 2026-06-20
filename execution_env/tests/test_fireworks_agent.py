"""
Tests for fireworks_agent.py.

All tests mock the OpenAI client so no real Fireworks API key is needed. The agent
shares schedule-primitive logic with llm_agent, so these tests focus on the Fireworks-
specific plumbing: the OpenAI client call, model/base-url configuration, and the
fallback-to-TWAP path on errors.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from execution_env.agents import fireworks_agent
from execution_env.rl.execution_gym_env import ExecutionEnv


@pytest.fixture
def episode_info():
    env = ExecutionEnv()
    _, info = env.reset(seed=42)
    return info


def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal object that looks like an openai ChatCompletion response."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


def test_propose_schedule_returns_correct_length(episode_info):
    """propose_schedule returns multipliers with the right length for the episode."""
    response_json = json.dumps(
        {"primitive": "twap", "pause_if_adverse_move": {"enabled": False}, "reasoning": "test"}
    )
    mock_response = _make_openai_response(response_json)

    with patch.dict("os.environ", {"FIREWORKS_API_KEY": "dummy"}):
        with patch("execution_env.agents.fireworks_agent.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            multipliers, parsed, raw = fireworks_agent.propose_schedule(
                ticker=episode_info["ticker"],
                total_shares=episode_info["total_shares"],
                n_slices=episode_info["n_slices"],
                open_price=episode_info["open_price"],
                volume_curve=episode_info["volume_curve"],
            )

    assert len(multipliers) == episode_info["n_slices"]
    assert parsed["primitive"] == "twap"


def test_propose_schedule_uses_fireworks_base_url(episode_info):
    """OpenAI client is constructed with the Fireworks base URL."""
    response_json = json.dumps({"primitive": "twap", "pause_if_adverse_move": {"enabled": False}, "reasoning": "x"})
    mock_response = _make_openai_response(response_json)

    with patch.dict("os.environ", {"FIREWORKS_API_KEY": "fw-testkey"}):
        with patch("execution_env.agents.fireworks_agent.OpenAI") as mock_cls:
            mock_client = MagicMock()
            mock_cls.return_value = mock_client
            mock_client.chat.completions.create.return_value = mock_response

            fireworks_agent.propose_schedule(
                ticker=episode_info["ticker"],
                total_shares=episode_info["total_shares"],
                n_slices=episode_info["n_slices"],
                open_price=episode_info["open_price"],
                volume_curve=episode_info["volume_curve"],
            )

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args.kwargs
    assert call_kwargs["base_url"] == fireworks_agent._FIREWORKS_BASE_URL
    assert call_kwargs["api_key"] == "fw-testkey"


def test_propose_schedule_raises_without_api_key():
    """propose_schedule raises EnvironmentError when FIREWORKS_API_KEY is missing."""
    with patch.dict("os.environ", {}, clear=True):
        import os
        os.environ.pop("FIREWORKS_API_KEY", None)
        with pytest.raises(EnvironmentError, match="FIREWORKS_API_KEY"):
            fireworks_agent.propose_schedule("AAPL", 10_000, 26, 150.0, [1 / 26] * 26)


def test_run_episode_falls_back_to_twap_on_api_error():
    """run_episode completes and falls back to TWAP when propose_schedule raises."""
    env = ExecutionEnv()

    with patch("execution_env.agents.fireworks_agent.propose_schedule", side_effect=RuntimeError("boom")):
        result = fireworks_agent.run_episode(env)

    assert "reward" in result
    assert result["primitive"] == "twap"
    assert "boom" in result["reasoning"]


def test_run_episode_completes_with_mocked_propose(episode_info):
    """run_episode drives a full episode to completion with a mocked schedule."""
    env = ExecutionEnv()
    n = episode_info["n_slices"]
    vc = np.asarray(episode_info["volume_curve"])

    from execution_env.agents.llm_agent import front_load
    multipliers = front_load(0.6, n, vc)

    parsed = {"primitive": "front_load", "pct": 0.6, "pause_if_adverse_move": {"enabled": False}, "reasoning": "unit test"}

    with patch("execution_env.agents.fireworks_agent.propose_schedule", return_value=(multipliers, parsed, "")):
        result = fireworks_agent.run_episode(env)

    assert result["primitive"] == "front_load"
    assert isinstance(result["reward"], float)
