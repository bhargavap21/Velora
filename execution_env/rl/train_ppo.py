"""
Train a baseline PPO agent against ExecutionEnv, then run the project's actual go/no-go
checkpoint: confirm the trained policy beats a naive TWAP baseline on a chronologically
held-out set of days neither policy trained on. If this doesn't pass, the simulator/
reward design needs revisiting before building the LLM agent or live demo on top of it.

Mirrors ../../train_baseline.py's structure.
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


def _make_train_env():
    return ExecutionEnv(day_range=_TRAIN_DAY_RANGE, order_adv_pct_range=_ORDER_ADV_PCT_RANGE)


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


def main() -> None:
    train_env = SubprocVecEnv([_make_train_env for _ in range(_N_ENVS)])
    holdout_env = ExecutionEnv(day_range=_HOLDOUT_DAY_RANGE, order_adv_pct_range=_ORDER_ADV_PCT_RANGE)

    model = PPO("MlpPolicy", train_env, verbose=1)

    callback = EpisodeRewardLogger()
    print(f"Training PPO for {_TOTAL_TIMESTEPS} timesteps across {_N_ENVS} envs on days {_TRAIN_DAY_RANGE}...")
    model.learn(total_timesteps=_TOTAL_TIMESTEPS, callback=callback)

    rewards = callback.episode_rewards
    print(f"Completed {len(rewards)} training episodes (env 0).")
    print(f"First 10 avg: {np.mean(rewards[:10]):.4f}   Last 10 avg: {np.mean(rewards[-10:]):.4f}")

    _RESULTS_DIR.mkdir(exist_ok=True)
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

    out_path = _RESULTS_DIR / "execution_ppo_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved training curve to {out_path}")

    model.save(MODEL_PATH)
    print(f"Saved PPO checkpoint to {MODEL_PATH}")

    # --- The actual checkpoint: PPO vs. the baselines on days neither saw in training ---
    eval_seeds = list(range(10_000, 10_000 + _N_EVAL_EPISODES))

    def ppo_policy(i, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    def vwap_match_policy(i, obs):
        return twap_action(i, _N_SLICES)

    # naive_twap_action reads env state, so evaluate it against the same env instance.
    naive_env = ExecutionEnv(day_range=_HOLDOUT_DAY_RANGE, order_adv_pct_range=_ORDER_ADV_PCT_RANGE)

    def naive_policy(i, obs):
        return naive_twap_action(naive_env)

    ppo_rewards = evaluate(holdout_env, eval_seeds, ppo_policy)
    vwap_rewards = evaluate(holdout_env, eval_seeds, vwap_match_policy)
    naive_rewards = evaluate(naive_env, eval_seeds, naive_policy)

    print(f"\n{'=' * 60}")
    print(f"HELD-OUT CHECKPOINT (days {_HOLDOUT_DAY_RANGE}, {_N_EVAL_EPISODES} episodes,")
    print(f"order size {_ORDER_ADV_PCT_RANGE[0]*100:.0f}-{_ORDER_ADV_PCT_RANGE[1]*100:.0f}% of ADV)")
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


if __name__ == "__main__":
    main()
