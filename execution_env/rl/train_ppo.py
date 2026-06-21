"""
Train a baseline PPO agent against ExecutionEnv, then run the project's actual go/no-go
checkpoint: confirm the trained policy beats a naive TWAP baseline on a chronologically
held-out set of days neither policy trained on. If this doesn't pass, the simulator/
reward design needs revisiting before building the LLM agent or live demo on top of it.

Mirrors ../../train_baseline.py's structure.

The core pieces (build_envs / run_training / run_holdout_eval / save_checkpoint_and_curve)
are factored out so both this local CLI (`python -m execution_env.rl.train_ppo`, the
curated DEFAULT_TICKERS, modest timesteps) and the Modal job (modal_train.py, TRAIN_TICKERS,
more timesteps, more CPU) call the same training/eval logic instead of duplicating it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES, naive_twap_action

# Train on a vectorized stack of envs across CPU cores -- the per-episode minute-bar
# processing dominates wall-clock, so parallel rollout collection is a big speedup.
_N_ENVS = 8
_TOTAL_TIMESTEPS = 700_000
_N_EVAL_EPISODES = 80
_TRAIN_DAY_RANGE = (0.0, 0.8)
_HOLDOUT_DAY_RANGE = (0.8, 1.0)
# Institutional order sizes (as a fraction of ADV) where participation-driven impact
# dominates and intraday scheduling actually matters. Kept within a range where the
# slippage reward stays mostly inside the +/-300 bps clip, so PPO gets clean gradients.
_ORDER_ADV_PCT_RANGE = (0.02, 0.10)
_RESULTS_DIR = Path(__file__).parent.parent / "results"
MODEL_PATH = _RESULTS_DIR / "ppo_execution.zip"
CURVE_PATH = _RESULTS_DIR / "execution_ppo_curve.png"

# Expanded training universe spanning liquidity/market-cap tiers, for the Modal job (see
# modal_train.py) -- the local CLI still defaults to the curated DEFAULT_TICKERS (4 names)
# so `python -m execution_env.rl.train_ppo` stays fast and doesn't need this data primed.
# Sanity-checked against live Alpaca/yfinance availability during priming
# (prime_data_cache.py); any symbol that fails to resolve there is skipped, not fatal.
TRAIN_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
    # Large-cap diversified
    "JPM", "V", "MA", "UNH", "HD", "PG", "KO", "XOM", "CVX", "DIS", "NFLX", "BAC", "WMT", "JNJ",
    # Mid-cap
    "RBLX", "PLTR", "SNAP", "PINS", "ROKU", "DOCU", "CHWY", "ETSY", "COIN", "U", "DDOG", "NET", "CRWD", "ZS",
    # Small / lower-liquidity
    "SOFI", "UPST", "AFRM", "OPEN", "BBAI", "IONQ", "LCID", "RIVN", "FUBO", "BB",
    # ETFs
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE",
]


class EpisodeRewardLogger(BaseCallback):
    """Logs per-episode total reward from env 0 of the vectorized stack, giving a
    representative training curve without double-counting parallel envs."""

    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self._current_total = 0.0

    def _on_step(self) -> bool:
        reward = self.locals["rewards"][0]
        self._current_total += reward
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._current_total)
            self._current_total = 0.0
        return True


def twap_action(slice_idx: int, n_slices: int) -> np.ndarray:
    """The VWAP-matching baseline: participation == 1.0 every slice trades exactly the
    volume-curve-implied fair share of remaining inventory, reproducing the VWAP
    benchmark the agent is scored against."""
    return np.array([1.0], dtype=np.float32)


def run_episode(env: ExecutionEnv, seed: int, policy) -> float:
    """policy(slice_idx, obs) -> action. Drives one episode, returns total reward."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    i = 0
    while True:
        action = policy(i, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        i += 1
        if terminated or truncated:
            break
    return total_reward


def evaluate(env: ExecutionEnv, seeds: list[int], policy) -> np.ndarray:
    return np.array([run_episode(env, seed, policy) for seed in seeds])


def build_envs(
    tickers: list[str] | None,
    n_envs: int,
    day_range: tuple[float, float] = _TRAIN_DAY_RANGE,
    order_adv_pct_range: tuple[float, float] = _ORDER_ADV_PCT_RANGE,
) -> SubprocVecEnv:
    """A vectorized stack of `n_envs` training envs sampling from `tickers` (None ->
    ExecutionEnv's own default, the curated DEFAULT_TICKERS)."""

    def _make():
        return ExecutionEnv(tickers=tickers, day_range=day_range, order_adv_pct_range=order_adv_pct_range)

    return SubprocVecEnv([_make for _ in range(n_envs)])


def run_training(model: PPO, total_timesteps: int) -> list[float]:
    """Runs model.learn() and returns the per-episode reward curve from env 0."""
    callback = EpisodeRewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    return callback.episode_rewards


def save_checkpoint_and_curve(
    model: PPO,
    rewards: list[float],
    model_path: Path = MODEL_PATH,
    curve_path: Path = CURVE_PATH,
) -> None:
    """Saves the trained checkpoint and a training-curve plot to the given paths."""
    model_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.4, color="steelblue", label="Episode reward")
    if len(rewards) >= 10:
        rolling = np.convolve(rewards, np.ones(10) / 10, mode="valid")
        ax.plot(range(10, len(rewards) + 1), rolling, color="crimson", linewidth=2, label="10-episode rolling avg")
    ax.set_title("Velora: SB3 PPO — Execution Reward (slippage bps vs. VWAP) Over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    model.save(model_path)


def run_holdout_eval(
    model: PPO,
    tickers: list[str] | None,
    eval_seeds: list[int],
    day_range: tuple[float, float] = _HOLDOUT_DAY_RANGE,
    order_adv_pct_range: tuple[float, float] = _ORDER_ADV_PCT_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The go/no-go checkpoint: PPO vs. the VWAP-match and naive-TWAP baselines on
    chronologically held-out days. Returns (ppo_rewards, vwap_rewards, naive_rewards)."""
    holdout_env = ExecutionEnv(tickers=tickers, day_range=day_range, order_adv_pct_range=order_adv_pct_range)
    # naive_twap_action reads env state, so evaluate it against its own env instance.
    naive_env = ExecutionEnv(tickers=tickers, day_range=day_range, order_adv_pct_range=order_adv_pct_range)

    def ppo_policy(i, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    def vwap_match_policy(i, obs):
        return twap_action(i, _N_SLICES)

    def naive_policy(i, obs):
        return naive_twap_action(naive_env)

    ppo_rewards = evaluate(holdout_env, eval_seeds, ppo_policy)
    vwap_rewards = evaluate(holdout_env, eval_seeds, vwap_match_policy)
    naive_rewards = evaluate(naive_env, eval_seeds, naive_policy)
    return ppo_rewards, vwap_rewards, naive_rewards


def print_holdout_report(
    ppo_rewards: np.ndarray,
    vwap_rewards: np.ndarray,
    naive_rewards: np.ndarray,
    day_range: tuple[float, float],
    order_adv_pct_range: tuple[float, float],
) -> bool:
    """Prints the go/no-go report; returns whether PPO beat naive TWAP (the bar this
    project has always graded itself against before building anything on top of it)."""
    print(f"\n{'=' * 60}")
    print(f"HELD-OUT CHECKPOINT (days {day_range}, {len(ppo_rewards)} episodes,")
    print(f"order size {order_adv_pct_range[0] * 100:.0f}-{order_adv_pct_range[1] * 100:.0f}% of ADV)")
    print(f"{'=' * 60}")
    print(f"PPO         mean reward: {ppo_rewards.mean():.4f}  (std {ppo_rewards.std():.4f})")
    print(f"VWAP-match  mean reward: {vwap_rewards.mean():.4f}  (std {vwap_rewards.std():.4f})")
    print(f"naive TWAP  mean reward: {naive_rewards.mean():.4f}  (std {naive_rewards.std():.4f})")

    vs_naive = ppo_rewards.mean() - naive_rewards.mean()
    passed = ppo_rewards.mean() > naive_rewards.mean()
    print(f"\nPPO advantage over naive TWAP: {vs_naive:+.4f} reward units "
          f"(~{vs_naive * 100:+.1f} bps/episode)")
    if passed:
        print("PASSED -- PPO beats naive TWAP on held-out days.")
    else:
        print("FAILED -- PPO does NOT beat naive TWAP on held-out days. Revisit the "
              "reward/observation design before building the demo on top of it.")
    return passed


def main() -> None:
    train_env = build_envs(tickers=None, n_envs=_N_ENVS)
    model = PPO("MlpPolicy", train_env, verbose=1)

    print(f"Training PPO for {_TOTAL_TIMESTEPS} timesteps across {_N_ENVS} envs on days {_TRAIN_DAY_RANGE}...")
    rewards = run_training(model, _TOTAL_TIMESTEPS)
    print(f"Completed {len(rewards)} training episodes (env 0).")
    print(f"First 10 avg: {np.mean(rewards[:10]):.4f}   Last 10 avg: {np.mean(rewards[-10:]):.4f}")

    save_checkpoint_and_curve(model, rewards)
    print(f"Saved PPO checkpoint to {MODEL_PATH}")
    print(f"Saved training curve to {CURVE_PATH}")

    # --- The actual checkpoint: PPO vs. the baselines on days neither saw in training ---
    eval_seeds = list(range(10_000, 10_000 + _N_EVAL_EPISODES))
    ppo_rewards, vwap_rewards, naive_rewards = run_holdout_eval(model, tickers=None, eval_seeds=eval_seeds)
    print_holdout_report(ppo_rewards, vwap_rewards, naive_rewards, _HOLDOUT_DAY_RANGE, _ORDER_ADV_PCT_RANGE)


if __name__ == "__main__":
    main()
