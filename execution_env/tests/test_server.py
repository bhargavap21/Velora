"""API-layer tests for the execution server, including the live SSE stream.

These run fully offline: they only exercise the `twap` policy (no API keys) and rely
on the committed parquet cache in execution_env/data_cache/, so no network is needed.
"""

import json

import pytest
from fastapi.testclient import TestClient

from execution_env.server import app

client = TestClient(app)

# A historical day present in the committed AAPL minute-bar cache.
_DATE = "2024-03-15"


def _parse_sse(text):
    """Parse an SSE response body into a list of (event, data_dict) tuples."""
    # SSE uses CRLF line endings; normalize so event blocks split cleanly.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    events = []
    for block in text.strip().split("\n\n"):
        event_name = None
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if event_name is None:
            continue
        payload = json.loads("\n".join(data_lines)) if data_lines else {}
        events.append((event_name, payload))
    return events


def test_policies_lists_twap():
    r = client.get("/api/policies")
    assert r.status_code == 200
    available = r.json()["available"]
    assert "twap" in available


def _mock_mismatched_ppo_model(monkeypatch):
    """Simulates a checkpoint trained on a different observation-space shape than the
    current ExecutionEnv (e.g. before the liquidity/volatility obs features were added),
    independent of whatever the actually-committed checkpoint's shape happens to be at
    any given time -- _get_ppo_model()'s mismatch-detection branch is what's under test."""
    import execution_env.server as server_module
    from gymnasium import spaces

    class _FakeModel:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,))

    class _FakePath:
        def exists(self):
            return True

    monkeypatch.setattr(server_module, "_ppo_model", None)
    monkeypatch.setattr(server_module, "_ppo_model_error", None)
    monkeypatch.setattr(server_module, "MODEL_PATH", _FakePath())
    monkeypatch.setattr("stable_baselines3.PPO.load", lambda *a, **k: _FakeModel())


def test_policies_excludes_ppo_when_checkpoint_obs_space_mismatched(monkeypatch):
    _mock_mismatched_ppo_model(monkeypatch)
    r = client.get("/api/policies")
    assert "ppo" not in r.json()["available"]


def test_episode_ppo_with_incompatible_checkpoint_returns_clean_503(monkeypatch):
    _mock_mismatched_ppo_model(monkeypatch)
    r = client.get("/api/episode?policy=ppo&ticker=AAPL&total_shares=10000")
    assert r.status_code == 503
    assert "observation space" in r.json()["detail"]


def test_config_has_defaults_and_constraints():
    r = client.get("/api/config")
    assert r.status_code == 200
    cfg = r.json()
    assert "defaults" in cfg
    assert "constraints" in cfg
    assert "AAPL" in cfg["tickers"]


def test_episode_json_twap_fills_completely():
    r = client.get(f"/api/episode?policy=twap&ticker=AAPL&date={_DATE}&total_shares=10000")
    assert r.status_code == 200
    ep = r.json()
    assert ep["policy"] == "twap"
    assert ep["ticker"] == "AAPL"
    assert ep["date"] == _DATE
    # Path has one more point than slices (initial price + one per slice).
    assert len(ep["path"]) == ep["n_slices"] + 1
    assert len(ep["volume_curve"]) == ep["n_slices"]
    assert len(ep["exec_prices"]) == ep["n_slices"]
    assert len(ep["exec_quantities"]) == ep["n_slices"]
    assert ep["filled_fraction"] == pytest.approx(1.0, abs=1e-6)


def test_episode_json_bad_ticker_400(monkeypatch):
    def _fail(ticker):
        raise ValueError(f"No market data found for {ticker!r}")

    monkeypatch.setattr("execution_env.server.ensure_daily_data", _fail)

    r = client.get("/api/episode?policy=twap&ticker=BOGUS")
    assert r.status_code == 400


def test_stream_emits_meta_slices_then_done():
    r = client.get(f"/api/episode/stream?policy=twap&ticker=AAPL&date={_DATE}&total_shares=10000&tick_ms=0")
    assert r.status_code == 200
    events = _parse_sse(r.text)

    names = [name for name, _ in events]
    assert names[0] == "meta"
    assert names[-1] == "done"

    meta = events[0][1]
    n_slices = meta["n_slices"]
    assert meta["ticker"] == "AAPL"
    assert len(meta["path"]) == n_slices + 1
    assert "exec_prices" not in meta  # fills stream incrementally, not in meta

    slices = [data for name, data in events if name == "slice"]
    assert len(slices) == n_slices
    # Slice indices arrive in order, each carrying a real fill.
    for i, s in enumerate(slices):
        assert s["i"] == i
        assert s["exec_price"] > 0
        assert s["exec_quantity"] >= 0

    done = events[-1][1]
    assert done["filled_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert len(done["exec_prices"]) == n_slices


def test_stream_bad_ticker_emits_error_event(monkeypatch):
    # ticker is now resolved dynamically (ensure_daily_data), which would otherwise hit
    # real Alpaca/yfinance for an unknown symbol -- patch it so this stays offline and
    # deterministic, same as the "no keys, no network" promise for this test file.
    def _fail(ticker):
        raise ValueError(f"No market data found for {ticker!r}")

    monkeypatch.setattr("execution_env.server.ensure_daily_data", _fail)

    r = client.get("/api/episode/stream?policy=twap&ticker=BOGUS")
    # SSE always returns 200; the failure is delivered as an `error` event.
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert len(events) == 1
    name, data = events[0]
    assert name == "error"
    assert "BOGUS" in data["detail"]


def test_stream_json_consistency():
    """The streamed fills should match what the JSON endpoint computes for the
    same deterministic scenario."""
    params = f"policy=twap&ticker=AAPL&date={_DATE}&total_shares=10000"
    json_ep = client.get(f"/api/episode?{params}").json()
    stream_events = _parse_sse(client.get(f"/api/episode/stream?{params}&tick_ms=0").text)
    done = stream_events[-1][1]
    assert done["exec_prices"] == pytest.approx(json_ep["exec_prices"], rel=1e-9)
    assert done["final_reward"] == pytest.approx(json_ep["final_reward"], abs=1e-6)
