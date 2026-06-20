"""
Train a baseline PPO agent against ExecutionEnv.

Mirrors ../../train_baseline.py's structure. Validation checkpoint: confirm the trained
policy beats a naive TWAP (equal-slice) baseline on held-out days before moving on to the
LLM agent or live demo work.

TODO(train_ppo): add the held-out-day evaluation split and the naive-TWAP baseline
comparison -- right now this only trains and plots reward, it doesn't yet prove the
policy is better than doing nothing clever.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from execution_env.rl.execution_gym_env import ExecutionEnv, _N_SLICES

_N_EPISODES = 200
_RESULTS_DIR = Path(__file__).parent.parent / "results"


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


def main() -> None:
    env = ExecutionEnv()
    model = PPO("MlpPolicy", env, verbose=0)

    callback = EpisodeRewardLogger()
    total_timesteps = _N_EPISODES * _N_SLICES
    print(f"Training PPO for {_N_EPISODES} episodes ({total_timesteps} timesteps)...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    rewards = callback.episode_rewards
    print(f"Completed {len(rewards)} episodes.")
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
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
