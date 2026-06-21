"""Tests for the HUD environment tools/templates in execution_env/env.py.

Runs fully offline against the committed parquet cache (DEFAULT_TICKERS only -- the
random sampler can draw from the full TRAIN_TICKERS universe, but tests pin it to a
cached ticker via monkeypatch so no network call is needed).
"""

import asyncio
import json

import pytest

import execution_env.env as env_mod


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset_episode_state():
    """env.py's episode state is module-global; isolate each test."""
    env_mod._EPISODE = None
    env_mod._EPISODE_INFO = None
    env_mod._SCHEDULE = None
    yield
    env_mod._EPISODE = None
    env_mod._EPISODE_INFO = None
    env_mod._SCHEDULE = None


def test_sample_random_scenario_lands_in_valid_ranges(monkeypatch):
    monkeypatch.setattr(
        "execution_env.rl.train_ppo.TRAIN_TICKERS", ["AAPL", "NVDA", "TSLA", "SPY"]
    )
    for _ in range(10):
        scenario = env_mod._sample_random_scenario()
        assert scenario["ticker"] in ("AAPL", "NVDA", "TSLA", "SPY")
        assert scenario["side"] in ("buy", "sell")
        assert scenario["total_shares"] > 0
        assert 0 <= scenario["seed"] < 2**31


def test_execution_fixed_produces_identical_tasks_for_same_args():
    """The GRPO-grouping entrypoint: N calls with the same scenario args must produce
    equal Task objects (same prompt/scenario), unlike execution_random's per-call
    resampling -- this is what lets a training script form one group."""
    kwargs = dict(prompt="p", ticker="AAPL", side="buy", total_shares=500_000, seed=42)
    tasks = [env_mod.execution_fixed(**kwargs) for _ in range(3)]
    assert all(t == tasks[0] for t in tasks)


def test_execution_random_task_is_registered():
    from execution_env.tasks import tasks

    slugs = [t.slug for t in tasks]
    assert "execution-random" in slugs
    # The original 3 fixed tasks must still be present, untouched.
    assert {"buy-10k-aapl", "buy-10k-tsla", "sell-10k-spy"}.issubset(set(slugs))


def test_submit_schedule_accepts_fenced_json():
    """The hardened parser tolerates markdown fences + prose the old strict json.loads
    would reject outright."""
    _run(_drain_one(env_mod._run_execution_template("p", ticker="AAPL", side="buy", total_shares=10_000, seed=1)))
    fenced = 'Here is my plan:\n```json\n{"schedule": [1.0]*26, "reasoning": "x"}\n```'.replace(
        "[1.0]*26", str([1.0] * 26)
    )
    result = _run(env_mod.submit_schedule(fenced))
    assert result["accepted"] is True


def test_submit_schedule_rejects_invalid_json():
    result = _run(env_mod.submit_schedule("not json at all {{{"))
    assert result["accepted"] is False


def test_submit_schedule_rejects_non_object_json():
    """A bare JSON array (valid JSON, wrong shape) must not crash with AttributeError."""
    result = _run(env_mod.submit_schedule("[1.0, 2.0, 3.0]"))
    assert result["accepted"] is False
    assert "object" in result["reason"].lower()


def test_submit_schedule_warns_on_length_mismatch():
    _run(_drain_one(env_mod._run_execution_template("p", ticker="AAPL", side="buy", total_shares=10_000, seed=1)))
    result = _run(env_mod.submit_schedule(json.dumps({"schedule": [1.0] * 5})))
    assert result["accepted"] is True
    assert "warning" in result


def test_no_submission_scores_at_floor_not_midpoint():
    """Regression test: a model that never calls submit_schedule() must score at the
    reward floor (0.0 normalized), not 0.5 (which used to mean "as good as VWAP")."""

    async def scenario():
        gen = env_mod._run_execution_template("p", ticker="AAPL", side="buy", total_shares=10_000, seed=1)
        await gen.__anext__()
        return await gen.__anext__()

    final = _run(scenario())
    assert final.reward == pytest.approx(0.0, abs=1e-6)


async def _drain_one(gen):
    return await gen.__anext__()
