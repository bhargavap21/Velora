import numpy as np
import pandas as pd
import pytest

from execution_env.sandbox_config import resolve_regime_date, shares_from_capital


def _make_df(n=100):
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 + rng.normal(0, 1, n).cumsum()
    open_ = close + rng.normal(0, 0.5, n)
    high = np.maximum(open_, close) + rng.uniform(0.5, 2, n)
    low = np.minimum(open_, close) - rng.uniform(0.5, 2, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": rng.integers(1e6, 5e6, n)}, index=idx)


def test_reset_pins_date():
    from execution_env.rl.execution_gym_env import ExecutionEnv

    env = ExecutionEnv(tickers=["AAPL"])
    env._data = {"AAPL": _make_df()}
    env._minute_data = {"AAPL": None}
    obs, info = env.reset(seed=42, options={"ticker": "AAPL", "date": "2023-01-03"})
    assert info["date"] == "2023-01-03"
    assert info["ticker"] == "AAPL"
    assert "data_source" in info


def test_reset_unknown_date_raises():
    from execution_env.rl.execution_gym_env import ExecutionEnv

    env = ExecutionEnv(tickers=["AAPL"])
    env._data = {"AAPL": _make_df()}
    with pytest.raises(ValueError, match="No data"):
        env.reset(options={"ticker": "AAPL", "date": "2099-01-01"})


def test_same_seed_same_regime_date():
    df = _make_df()
    d1 = resolve_regime_date(df, "high_vol", seed=7)
    d2 = resolve_regime_date(df, "high_vol", seed=7)
    assert d1 == d2


def test_shares_from_capital_rounds_to_hundreds():
    shares = shares_from_capital(1_000_000, 150.0)
    assert shares % 100 == 0
    assert shares == 6600
