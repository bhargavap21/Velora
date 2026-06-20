"""
Gymnasium wrapper around the StratRL strategy-mutation environment.

v1 scope cuts (intentional simplifications, not missing features):

  - Action space is discretized to a fixed menu of 11 mutations that only ever
    target the "rsi_14" indicator (present and entry/exit-referenced in all 5
    SEED_STRATEGIES) plus stop_loss/take_profit. The full LLM-facing action
    space (environment.py / episode_core.py) supports arbitrary indicator
    names, add/remove indicator, and ticker swaps — those aren't expressible
    as a fixed-shape Discrete action, so this baseline only exercises the
    subset of mutations that apply uniformly across all seeds.
  - Observation space flattens the strategy dict into a small numeric vector
    (rsi_14 period/entry threshold/exit threshold, stop_loss, take_profit) plus
    the 6 backtest metrics, market regime, and ticker — fixed shape, required
    by classical RL libraries. The free-form text observation used by the LLM
    agent (episode_core.build_turn_observation) is not used here.

This exists to validate that the environment produces learnable signal for a
classical (non-LLM) policy, per the StratRL pitch: "works for both LLM and
classical RL policies." It is not meant to outperform the LLM agent.
"""

from __future__ import annotations

import copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from backtester import run_backtest
from episode_core import MAX_TURNS, SEED_STRATEGIES
from mutations import MutationError, apply_mutation
from regime import classify_regime
from reward import compute_reward

_TICKERS = ["TSLA", "NVDA", "AAPL", "SPY"]
_REGIMES = ["trending", "mean_reverting", "choppy", "unknown"]

_RSI_PERIOD_STEP = 2
_RSI_THRESHOLD_STEP = 5.0
_STOP_LOSS_STEP = 0.01
_TAKE_PROFIT_STEP = 0.02

_ACTIONS = [
    ("rsi_period", +_RSI_PERIOD_STEP),
    ("rsi_period", -_RSI_PERIOD_STEP),
    ("rsi_entry_threshold", +_RSI_THRESHOLD_STEP),
    ("rsi_entry_threshold", -_RSI_THRESHOLD_STEP),
    ("rsi_exit_threshold", +_RSI_THRESHOLD_STEP),
    ("rsi_exit_threshold", -_RSI_THRESHOLD_STEP),
    ("stop_loss", +_STOP_LOSS_STEP),
    ("stop_loss", -_STOP_LOSS_STEP),
    ("take_profit", +_TAKE_PROFIT_STEP),
    ("take_profit", -_TAKE_PROFIT_STEP),
    ("noop", 0.0),
]


def _find_rsi(strategy: dict) -> dict:
    """RSI's indicator name encodes its period (e.g. "rsi_14" -> "rsi_16" after a
    period change), so it must be looked up by type, not by a fixed name."""
    return next(i for i in strategy["indicators"] if i["type"] == "RSI")


def _rsi_entry_value(strategy: dict) -> float:
    rsi_name = _find_rsi(strategy)["name"]
    for c in strategy["entry_conditions"]:
        if c["indicator"] == rsi_name:
            return float(c["value"])
    raise ValueError("strategy has no RSI entry condition")


def _rsi_exit_value(strategy: dict) -> float:
    rsi_name = _find_rsi(strategy)["name"]
    for c in strategy["exit_conditions"]:
        if c["indicator"] == rsi_name:
            return float(c["value"])
    raise ValueError("strategy has no RSI exit condition")


def _action_to_mutation(action_idx: int, strategy: dict) -> dict | None:
    """Returns a mutation dict for apply_mutation, or None for the no-op action."""
    field, delta = _ACTIONS[action_idx]

    if field == "noop":
        return None
    if field == "rsi_period":
        rsi = _find_rsi(strategy)
        return {
            "type": "change_indicator_period",
            "indicator_name": rsi["name"],
            "new_period": rsi["period"] + delta,
            "reasoning": "gym baseline action",
        }
    if field == "rsi_entry_threshold":
        return {
            "type": "change_condition_threshold",
            "indicator_name": _find_rsi(strategy)["name"],
            "condition_type": "entry",
            "new_value": _rsi_entry_value(strategy) + delta,
            "reasoning": "gym baseline action",
        }
    if field == "rsi_exit_threshold":
        return {
            "type": "change_condition_threshold",
            "indicator_name": _find_rsi(strategy)["name"],
            "condition_type": "exit",
            "new_value": _rsi_exit_value(strategy) + delta,
            "reasoning": "gym baseline action",
        }
    if field == "stop_loss":
        return {
            "type": "change_stop_loss",
            "new_value": round(strategy["stop_loss"] + delta, 4),
            "reasoning": "gym baseline action",
        }
    if field == "take_profit":
        return {
            "type": "change_take_profit",
            "new_value": round(strategy["take_profit"] + delta, 4),
            "reasoning": "gym baseline action",
        }
    raise ValueError(f"unknown action field '{field}'")


class StratRLEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seeds: list[dict] | None = None):
        super().__init__()
        self._seeds = seeds if seeds is not None else SEED_STRATEGIES
        self.action_space = spaces.Discrete(len(_ACTIONS))
        self.observation_space = spaces.Dict({
            "metrics": spaces.Box(
                low=np.array([-10.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([10.0, 10.0, 1.0, 1.0, 500.0, 20.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "strategy": spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            ),
            "regime": spaces.Discrete(len(_REGIMES)),
            "ticker": spaces.Discrete(len(_TICKERS)),
        })

        self._strategy: dict = {}
        self._metrics: dict = {}
        self._turn = 0

    def _build_obs(self) -> dict:
        m = self._metrics
        s = self._strategy
        rsi = _find_rsi(s)
        regime = classify_regime(s["ticker"])
        return {
            "metrics": np.array([
                m["sharpe"], m["total_return"], m["max_drawdown"],
                m["win_rate"], m["num_trades"], m["profit_factor"],
            ], dtype=np.float32),
            "strategy": np.array([
                rsi["period"] / 50.0,
                _rsi_entry_value(s) / 100.0,
                _rsi_exit_value(s) / 100.0,
                s["stop_loss"] / 0.50,
                s["take_profit"] / 5.0,
            ], dtype=np.float32),
            "regime": _REGIMES.index(regime) if regime in _REGIMES else _REGIMES.index("unknown"),
            "ticker": _TICKERS.index(s["ticker"]),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        seed_idx = int(self.np_random.integers(len(self._seeds)))
        self._strategy = copy.deepcopy(self._seeds[seed_idx])
        self._metrics = run_backtest(self._strategy)
        self._turn = 0
        return self._build_obs(), {}

    def step(self, action: int):
        mutation = _action_to_mutation(int(action), self._strategy)
        self._turn += 1

        if mutation is None:
            reward = 0.0
        else:
            try:
                new_strategy = apply_mutation(self._strategy, mutation)
                new_metrics = run_backtest(new_strategy)
                reward = compute_reward(self._metrics, new_metrics)
                self._strategy = new_strategy
                self._metrics = new_metrics
            except MutationError:
                reward = 0.0

        terminated = self._turn >= MAX_TURNS
        truncated = False
        return self._build_obs(), reward, terminated, truncated, {}


if __name__ == "__main__":
    from gymnasium.utils.env_checker import check_env

    env = StratRLEnv()
    check_env(env, skip_render_check=True)
    print("check_env passed.")

    obs, info = env.reset()
    print("Initial obs:", obs)
    total_reward = 0.0
    for _ in range(MAX_TURNS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"Random-policy episode total reward: {total_reward:.4f}")
