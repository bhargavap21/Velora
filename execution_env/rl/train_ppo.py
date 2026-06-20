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

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES

_N_EPISODES = 40_000
_N_EVAL_EPISODES = 50
_TRAIN_DAY_RANGE = (0.0, 0.8)
_HOLDOUT_DAY_RANGE = (0.8, 1.0)
_RESULTS_DIR = Path(__file__).parent.parent / "results"
MODEL_PATH = _RESULTS_DIR / "ppo_execution.zip"


class EpisodeRewardLogger(BaseCallback):
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


def main() -> None:
    train_env = ExecutionEnv(day_range=_TRAIN_DAY_RANGE)
    holdout_env = ExecutionEnv(day_range=_HOLDOUT_DAY_RANGE)

    model = PPO("MlpPolicy", train_env, verbose=0)

    callback = EpisodeRewardLogger()
    total_timesteps = _N_EPISODES * _N_SLICES
    print(f"Training PPO for {_N_EPISODES} episodes ({total_timesteps} timesteps) on days {_TRAIN_DAY_RANGE}...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    rewards = callback.episode_rewards
    print(f"Completed {len(rewards)} training episodes.")
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

    # --- The actual checkpoint: PPO vs. naive TWAP on days neither saw during training ---
    eval_seeds = list(range(10_000, 10_000 + _N_EVAL_EPISODES))

    def ppo_policy(i, obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    def twap_policy(i, obs):
        return twap_action(i, _N_SLICES)

    ppo_rewards = evaluate(holdout_env, eval_seeds, ppo_policy)
    twap_rewards = evaluate(holdout_env, eval_seeds, twap_policy)

    print(f"\n{'=' * 60}")
    print(f"HELD-OUT CHECKPOINT (days {_HOLDOUT_DAY_RANGE}, {_N_EVAL_EPISODES} episodes)")
    print(f"{'=' * 60}")
    print(f"PPO  mean reward: {ppo_rewards.mean():.4f}  (std {ppo_rewards.std():.4f})")
    print(f"TWAP mean reward: {twap_rewards.mean():.4f}  (std {twap_rewards.std():.4f})")

    passed = ppo_rewards.mean() > twap_rewards.mean()
    if passed:
        print("PASSED -- PPO beats naive TWAP on held-out days.")
    else:
        print("FAILED -- PPO does NOT beat naive TWAP on held-out days. Do not proceed to "
              "the LLM agent or live demo until this passes; revisit the reward/observation design.")


if __name__ == "__main__":
    main()
