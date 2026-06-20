import json

import numpy as np
import pytest

from execution_env.agents import llm_agent
from execution_env.rl.execution_gym_env import ExecutionEnv


@pytest.fixture
def volume_curve():
    env = ExecutionEnv()
    _, info = env.reset(seed=0)
    return np.array(info["volume_curve"]), info["n_slices"]


def test_follow_volume_curve_collapses_to_constant_one(volume_curve):
    vc, n = volume_curve
    multipliers = llm_agent.follow_volume_curve(n, vc)
    assert np.allclose(multipliers, 1.0, atol=1e-6)


def test_primitives_stay_within_action_bounds(volume_curve):
    vc, n = volume_curve
    for multipliers in [
        llm_agent.front_load(0.6, n, vc),
        llm_agent.back_load(0.6, n, vc),
        llm_agent.twap(n, vc),
        llm_agent.follow_volume_curve(n, vc),
    ]:
        assert len(multipliers) == n
        assert all(0.0 <= m <= 2.0 for m in multipliers)
        assert multipliers[-1] == 1.0  # forced full-fill on the last slice


def test_primitives_fully_fill_order_through_real_env(volume_curve):
    vc, n = volume_curve
    env = ExecutionEnv()
    for multipliers in [
        llm_agent.front_load(0.6, n, vc),
        llm_agent.back_load(0.6, n, vc),
        llm_agent.twap(n, vc),
    ]:
        env.reset(seed=1)
        for m in multipliers:
            _, _, terminated, truncated, _ = env.step(np.array([m], dtype=np.float32))
            if terminated or truncated:
                break
        assert abs(env._shares_remaining) < 1e-6


def test_extract_json_strips_markdown_fences():
    text = '```json\n{"primitive": "twap"}\n```'
    assert llm_agent._extract_json(text) == {"primitive": "twap"}


def test_schedule_from_response_unknown_primitive_raises(volume_curve):
    vc, n = volume_curve
    with pytest.raises(ValueError):
        llm_agent.schedule_from_response(json.dumps({"primitive": "bogus"}), n, vc)


def test_schedule_from_response_valid_primitive(volume_curve):
    vc, n = volume_curve
    multipliers, parsed = llm_agent.schedule_from_response(
        json.dumps({"primitive": "follow_volume_curve"}), n, vc
    )
    assert len(multipliers) == n
    assert parsed["primitive"] == "follow_volume_curve"


def test_run_episode_completes_with_mocked_propose_schedule(monkeypatch, volume_curve):
    vc, n = volume_curve

    def fake_propose_schedule(ticker, total_shares, n_slices, open_price, volume_curve):
        multipliers = llm_agent.twap(n_slices, np.asarray(volume_curve, dtype=float))
        parsed = {"primitive": "twap", "pause_if_adverse_move": {"enabled": False}, "reasoning": "test"}
        return multipliers, parsed, "{}"

    monkeypatch.setattr(llm_agent, "propose_schedule", fake_propose_schedule)

    env = ExecutionEnv()
    result = llm_agent.run_episode(env)

    assert result["primitive"] == "twap"
    assert isinstance(result["reward"], float)


def test_run_episode_pause_modifier_never_blocks_final_fill(monkeypatch, volume_curve):
    """Regression test: an adverse-move pause on the very last slice must not leave
    inventory unfilled -- the forced final-slice multiplier always takes priority."""
    vc, n = volume_curve

    def fake_propose_schedule(ticker, total_shares, n_slices, open_price, volume_curve):
        multipliers = llm_agent.twap(n_slices, np.asarray(volume_curve, dtype=float))
        parsed = {
            "primitive": "twap",
            # threshold of 0 means even a tiny favorable-for-pause price tick triggers it
            "pause_if_adverse_move": {"enabled": True, "threshold_bps": 0.0001},
            "reasoning": "test",
        }
        return multipliers, parsed, "{}"

    monkeypatch.setattr(llm_agent, "propose_schedule", fake_propose_schedule)

    env = ExecutionEnv()
    llm_agent.run_episode(env)

    assert abs(env._shares_remaining) < 1e-6


def test_run_episode_falls_back_to_twap_on_propose_schedule_error(monkeypatch):
    def raising_propose_schedule(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(llm_agent, "propose_schedule", raising_propose_schedule)

    env = ExecutionEnv()
    result = llm_agent.run_episode(env)

    assert result["primitive"] == "twap"
    assert "fallback" in result["reasoning"]
