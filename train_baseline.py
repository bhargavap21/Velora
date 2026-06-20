"""
Train a baseline PPO agent against StratRLEnv (gym_env.py) for ~100 episodes.

This is not meant to outperform the LLM agent — it's the artifact that proves
the environment produces learnable signal for a classical RL policy, not just
an LLM zero-shot prompt. See gym_env.py's module docstring for the v1 scope
cuts (discretized action space, flattened observation).

Usage:
    python train_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from episode_core import MAX_TURNS
from gym_env import StratRLEnv

_N_EPISODES = 100
_RESULTS_DIR = Path(__file__).parent / "results"


class EpisodeRewardLogger(BaseCallback):
    """Collects total reward per completed episode for plotting afterward."""

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
    env = StratRLEnv()
    model = PPO("MultiInputPolicy", env, verbose=0)

    callback = EpisodeRewardLogger()
    total_timesteps = _N_EPISODES * MAX_TURNS
    print(f"Training PPO for {_N_EPISODES} episodes ({total_timesteps} timesteps)...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    rewards = callback.episode_rewards
    print(f"Completed {len(rewards)} episodes.")
    print(f"First 10 avg: {np.mean(rewards[:10]):.4f}   Last 10 avg: {np.mean(rewards[-10:]):.4f}")

    _RESULTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.4, color="steelblue", label="Episode total reward")
    if len(rewards) >= 10:
        window = 10
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(rewards) + 1), rolling, color="crimson", linewidth=2, label=f"{window}-episode rolling avg")
    ax.set_title("StratRL: SB3 PPO Baseline — Episode Reward Over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total episode reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = _RESULTS_DIR / "sb3_ppo_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
