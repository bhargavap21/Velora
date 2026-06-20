"""
Gymnasium wrapper around the optimal-execution simulator.

One episode = one (ticker, historical day) pair. The agent must execute a fixed total
quantity over a fixed number of slices; action is the fraction of *remaining* inventory
to execute this slice (continuous, bounded [0, 1]) so the action space stays well-posed
regardless of how much inventory is left.

TODO(execution_gym_env): this skeleton wires the interfaces together but has not been
run through gymnasium's check_env yet -- do that before training. See
simulator/market_sim.py and simulator/benchmark.py for the pieces this composes.
"""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from execution_env.simulator.benchmark import compute_vwap, execution_vwap, slippage_reward
from execution_env.simulator.market_sim import ImpactModel, generate_intraday_path, load_daily_data, u_shaped_volume_curve

_TICKERS = ["TSLA", "NVDA", "AAPL", "SPY"]
_N_SLICES = 26  # e.g. 15-min slices across a 6.5h trading day
_TOTAL_SHARES = 10_000


class ExecutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, n_slices: int = _N_SLICES, total_shares: int = _TOTAL_SHARES, side: str = "buy"):
        super().__init__()
        self._n_slices = n_slices
        self._total_shares = total_shares
        self._side = side
        self._data = load_daily_data()

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
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
        return np.array(
            [shares_remaining_frac, time_remaining_frac, np.clip(recent_return, -1, 1), volume_curve_position],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ticker = self._tickers_choice()
        df = self._data[ticker]
        day_idx = int(self.np_random.integers(len(df)))
        day_row = df.iloc[day_idx]

        rng = np.random.default_rng(seed)
        self._path = generate_intraday_path(day_row, self._n_slices, rng)
        self._volume_curve = u_shaped_volume_curve(self._n_slices)
        adv = float(day_row["Volume"])
        self._impact = ImpactModel(adv=adv)

        self._slice_idx = 0
        self._shares_remaining = float(self._total_shares)
        self._exec_prices = []
        self._exec_quantities = []
        self._permanent_offset = 0.0

        return self._build_obs(), {}

    def _tickers_choice(self) -> str:
        return _TICKERS[int(self.np_random.integers(len(_TICKERS)))]

    def step(self, action: np.ndarray):
        frac = float(np.clip(action[0], 0.0, 1.0))
        qty = frac * self._shares_remaining

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
