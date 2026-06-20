"""
Gymnasium wrapper around the optimal-execution simulator.

One episode = one (ticker, historical day) pair. The agent must execute a fixed total
quantity over a fixed number of slices; action is a participation multiplier (continuous,
bounded [0, 2]) on the volume-curve-implied "fair share" of *remaining* inventory for this
slice, so the action space stays well-posed regardless of how much inventory is left, and
`action == 1.0` exactly reproduces the VWAP benchmark schedule the agent is scored against.
See simulator/market_sim.py and simulator/benchmark.py for the pieces this composes.
"""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import pandas as pd

from execution_env.simulator.benchmark import _MAX_SLIPPAGE_BPS, compute_vwap, execution_vwap, slippage_reward
from execution_env.simulator.market_sim import (
    ImpactModel,
    intraday_path_and_volume,
    load_daily_data,
    load_minute_data,
)

_TICKERS = ["TSLA", "NVDA", "AAPL", "SPY"]
_N_SLICES = 26  # e.g. 15-min slices across a 6.5h trading day
_TOTAL_SHARES = 10_000


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        n_slices: int = _N_SLICES,
        total_shares: int = _TOTAL_SHARES,
        side: str = "buy",
        day_range: tuple[float, float] = (0.0, 1.0),
        tickers: list[str] | None = None,
    ):
        """day_range restricts which (ticker, day) pairs reset() can sample, as a
        fraction of each ticker's chronological history -- e.g. (0.0, 0.8) for a
        training split, (0.8, 1.0) for a held-out evaluation split. Chronological, not
        random, so the holdout set is genuinely unseen future data, not just a random
        subsample of the same period.

        tickers restricts which tickers reset() can sample from (defaults to all of
        _TICKERS) -- e.g. so a caller can pin an episode to a single requested ticker.
        """
        super().__init__()
        self._n_slices = n_slices
        self._total_shares = total_shares
        self._side = side
        self._day_range = day_range
        self._tickers = tickers or _TICKERS
        self._data = load_daily_data()
        self._minute_data = {ticker: load_minute_data(ticker) for ticker in self._tickers}

        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._path: np.ndarray = np.array([])
        self._volume_curve: np.ndarray = np.array([])
        self._impact: ImpactModel | None = None
        self._slice_idx = 0
        self._shares_remaining = 0.0
        self._exec_prices: list[float] = []
        self._exec_quantities: list[float] = []
        self._permanent_offset = 0.0

    def _build_obs(self) -> np.ndarray:
        time_remaining_frac = 1.0 - self._slice_idx / self._n_slices
        shares_remaining_frac = self._shares_remaining / self._total_shares
        recent_return = 0.0
        if self._slice_idx > 0:
            recent_return = (self._path[self._slice_idx] - self._path[self._slice_idx - 1]) / self._path[self._slice_idx - 1]
        volume_curve_position = self._volume_curve[min(self._slice_idx, self._n_slices - 1)]

        interim_slippage_norm = 0.0
        if self._slice_idx > 0:
            interim_agent_vwap = execution_vwap(np.array(self._exec_prices), np.array(self._exec_quantities))
            interim_benchmark_vwap = compute_vwap(
                self._path[1 : self._slice_idx + 1], self._volume_curve[: self._slice_idx] * self._total_shares
            )
            raw_bps = (interim_benchmark_vwap - interim_agent_vwap) / interim_benchmark_vwap * 10_000
            if self._side == "sell":
                raw_bps *= -1
            interim_slippage_norm = float(np.clip(raw_bps, -_MAX_SLIPPAGE_BPS, _MAX_SLIPPAGE_BPS) / _MAX_SLIPPAGE_BPS)

        return np.array(
            [
                shares_remaining_frac,
                time_remaining_frac,
                np.clip(recent_return, -1, 1),
                volume_curve_position,
                interim_slippage_norm,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "ticker" in options:
            ticker = options["ticker"]
            if ticker not in self._tickers:
                raise ValueError(f"ticker {ticker!r} not in {self._tickers}")
        else:
            ticker = self._tickers_choice()
        self._ticker = ticker

        df = self._data[ticker]
        if "date" in options:
            target = pd.Timestamp(options["date"]).normalize()
            matches = df.index[df.index.normalize() == target]
            if matches.empty:
                raise ValueError(f"No data for {ticker} on {options['date']}")
            day_ts = matches[0]
            day_row = df.loc[day_ts]
        else:
            lo = int(len(df) * self._day_range[0])
            hi = max(lo + 1, int(len(df) * self._day_range[1]))
            day_idx = lo + int(self.np_random.integers(hi - lo))
            day_ts = df.index[day_idx]
            day_row = df.iloc[day_idx]

        rng = np.random.default_rng(seed)
        self._path, self._volume_curve, data_source = intraday_path_and_volume(
            ticker, day_ts, day_row, self._n_slices, rng, self._minute_data
        )

        adv = float(day_row["Volume"])
        self._impact = ImpactModel(adv=adv)

        self._slice_idx = 0
        self._shares_remaining = float(self._total_shares)
        self._exec_prices = []
        self._exec_quantities = []
        self._permanent_offset = 0.0

        info = {
            "ticker": ticker,
            "date": day_ts.strftime("%Y-%m-%d"),
            "side": self._side,
            "open_price": float(self._path[0]),
            "close_price": float(day_row["Close"]),
            "adv": adv,
            "data_source": data_source,
            "volume_curve": self._volume_curve.tolist(),
            "n_slices": self._n_slices,
            "total_shares": self._total_shares,
            "seed": seed,
        }
        return self._build_obs(), info

    def _tickers_choice(self) -> str:
        return self._tickers[int(self.np_random.integers(len(self._tickers)))]

    def step(self, action: np.ndarray):
        remaining_weights = self._volume_curve[self._slice_idx :]
        fair_share_qty = (self._volume_curve[self._slice_idx] / remaining_weights.sum()) * self._shares_remaining
        participation = float(np.clip(action[0], 0.0, 2.0))
        qty = float(np.clip(participation * fair_share_qty, 0.0, self._shares_remaining))

        base_price = self._path[self._slice_idx] * (1 + self._permanent_offset)
        temp_impact = self._impact.temporary_impact(qty)
        perm_impact = self._impact.permanent_impact(qty)
        sign = 1 if self._side == "buy" else -1
        exec_price = base_price * (1 + sign * temp_impact)
        self._permanent_offset += sign * perm_impact

        self._exec_prices.append(exec_price)
        self._exec_quantities.append(qty)
        self._shares_remaining -= qty
        self._slice_idx += 1

        terminated = self._slice_idx >= self._n_slices
        reward = 0.0
        if terminated:
            agent_vwap = execution_vwap(np.array(self._exec_prices), np.array(self._exec_quantities))
            benchmark_vwap = compute_vwap(self._path[1:], self._volume_curve * self._total_shares)
            filled_fraction = 1.0 - self._shares_remaining / self._total_shares
            reward = slippage_reward(agent_vwap, benchmark_vwap, self._side, filled_fraction)

        return self._build_obs(), reward, terminated, False, {}


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = ExecutionEnv()
    check_env(env, skip_render_check=True)
    print("check_env passed.")

    obs, info = env.reset()
    total_reward = 0.0
    for _ in range(_N_SLICES):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Random-policy episode reward: {total_reward:.4f}")
