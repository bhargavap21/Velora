"""
Plot average reward per turn, averaged across all seeds, from the most recent
Modal run. This is the headline artifact for the demo: it shows whether the
environment produces a learnable signal (reward should trend upward over the
8 turns) or whether the observation/action space needs enrichment.

Usage:
    python analysis/plot_reward_curve.py [path/to/modal_run_*.json]

If no path is given, uses the most recently modified results/modal_run_*.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

_RESULTS_DIR = Path(__file__).parent.parent / "results"


def _latest_run_path() -> Path:
    candidates = sorted(_RESULTS_DIR.glob("modal_run_*.json"))
    if not candidates:
        raise FileNotFoundError(f"No modal_run_*.json files found in {_RESULTS_DIR}")
    return candidates[-1]


def load_turn_rewards(path: Path) -> dict[str, dict[int, float]]:
    """Returns {seed_name: {turn: reward}}."""
    with open(path) as f:
        data = json.load(f)

    by_seed: dict[str, dict[int, float]] = defaultdict(dict)
    for event in data["events"]:
        if event["type"] == "turn":
            by_seed[event["seed_name"]][event["turn"]] = event["reward"]
    return by_seed


def average_reward_per_turn(by_seed: dict[str, dict[int, float]]) -> tuple[list[int], list[float]]:
    max_turn = max(t for turns in by_seed.values() for t in turns)
    avg_per_turn = []
    turn_indices = []
    for turn in range(1, max_turn + 1):
        rewards_this_turn = [turns[turn] for turns in by_seed.values() if turn in turns]
        if rewards_this_turn:
            avg_per_turn.append(sum(rewards_this_turn) / len(rewards_this_turn))
            turn_indices.append(turn)
    return turn_indices, avg_per_turn


def load_turn_sharpes(path: Path) -> dict[str, dict[int, float]]:
    """Returns {seed_name: {turn: sharpe_after}}."""
    with open(path) as f:
        data = json.load(f)

    by_seed: dict[str, dict[int, float]] = defaultdict(dict)
    for event in data["events"]:
        if event["type"] == "turn":
            by_seed[event["seed_name"]][event["turn"]] = event["sharpe_after"]
    return by_seed


def running_best_delta(by_seed: dict[str, dict[int, float]]) -> tuple[list[int], list[float]]:
    """Running-max Sharpe per turn minus each seed's turn-1 Sharpe, then averaged across
    seeds. Uses absolute delta (not a ratio) so a near-zero baseline doesn't distort the
    average. Captures explore/exploit behavior (regressions during exploration, but a
    monotonically improving 'best so far' envelope) rather than noisy per-step reward.
    """
    max_turn = max(t for turns in by_seed.values() for t in turns)
    delta_curves = []
    for turns in by_seed.values():
        ordered = [turns[t] for t in sorted(turns)]
        baseline = ordered[0]
        running_max = []
        best = float("-inf")
        for v in ordered:
            best = max(best, v)
            running_max.append(best - baseline)
        delta_curves.append(running_max)

    turn_indices = list(range(1, max_turn + 1))
    avg_curve = []
    for i in turn_indices:
        vals = [c[i - 1] for c in delta_curves if i - 1 < len(c)]
        avg_curve.append(sum(vals) / len(vals))
    return turn_indices, avg_curve


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _latest_run_path()
    print(f"Loading {path}")

    by_seed = load_turn_rewards(path)
    print(f"Seeds: {list(by_seed.keys())}")

    turns, avg_rewards = average_reward_per_turn(by_seed)
    for t, r in zip(turns, avg_rewards):
        print(f"  Turn {t}: avg reward = {r:.4f}  (n={sum(1 for s in by_seed.values() if t in s)})")

    # Simple trend check: compare mean of first half vs second half of turns
    midpoint = len(avg_rewards) // 2
    first_half_avg = sum(avg_rewards[:midpoint]) / midpoint if midpoint else 0
    second_half_avg = sum(avg_rewards[midpoint:]) / (len(avg_rewards) - midpoint)
    trend = "UPWARD" if second_half_avg > first_half_avg else "FLAT/DOWNWARD"
    print(f"\nFirst-half avg: {first_half_avg:.4f}  Second-half avg: {second_half_avg:.4f}  Trend: {trend}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(turns, avg_rewards, marker="o", color="steelblue", linewidth=2, label="Avg reward across seeds")

    for seed_name, turn_rewards in by_seed.items():
        seed_turns = sorted(turn_rewards)
        seed_values = [turn_rewards[t] for t in seed_turns]
        ax.plot(seed_turns, seed_values, alpha=0.25, linewidth=1, linestyle="--")

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(f"StratRL: Avg Reward per Turn Across {len(by_seed)} Seeds — {trend}")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Reward")
    ax.set_xticks(turns)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = _RESULTS_DIR / "reward_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")

    sharpe_by_seed = load_turn_sharpes(path)
    best_turns, best_curve = running_best_delta(sharpe_by_seed)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(best_turns, best_curve, marker="o", color="seagreen", linewidth=2,
              label="Avg running-best Sharpe improvement vs turn 1")
    ax2.axhline(0.0, color="gray", linewidth=0.8, linestyle=":", label="Turn-1 baseline")
    ax2.set_title("StratRL: Running-Best Sharpe Improvement Across Seeds (explore/exploit signal)")
    ax2.set_xlabel("Turn")
    ax2.set_ylabel("Best Sharpe so far − Turn-1 Sharpe")
    ax2.set_xticks(best_turns)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    out_path2 = _RESULTS_DIR / "sharpe_envelope.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path2}")
    envelope_trend = "UPWARD" if best_curve[-1] > best_curve[0] else "FLAT/DOWNWARD"
    print(f"Running-best Sharpe improvement envelope: {best_curve[0]:+.2f} -> {best_curve[-1]:+.2f}  ({envelope_trend})")


if __name__ == "__main__":
    main()
