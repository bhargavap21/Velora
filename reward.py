import math


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_reward(before: dict, after: dict) -> float:
    """
    Compute a reward in [0.0, 1.0] based on how much the mutation improved the strategy.
    """
    before_sharpe = before.get("sharpe") or 0.0
    after_sharpe = after.get("sharpe") or 0.0

    sharpe_delta = after_sharpe - before_sharpe
    sharpe_score = _sigmoid(sharpe_delta * 3)

    drawdown_delta = before["max_drawdown"] - after["max_drawdown"]
    drawdown_bonus = _clip(drawdown_delta * 0.5, -0.1, 0.1)

    reward = _clip(sharpe_score + drawdown_bonus, 0.0, 1.0)

    if after["num_trades"] < 5:
        reward *= 0.5

    return round(reward, 4)


def describe_delta(before: dict, after: dict) -> str:
    """
    Return a human-readable one-line summary of what changed.
    e.g. "Sharpe: 0.31 → 0.74 (+0.43) | trades: 12 → 8 | drawdown: 18.2% → 14.1%"
    Used by the demo to print turn-by-turn progress.
    """
    b_sharpe = before.get("sharpe") or 0.0
    a_sharpe = after.get("sharpe") or 0.0
    sharpe_d = a_sharpe - b_sharpe
    sign = "+" if sharpe_d >= 0 else ""
    return (
        f"Sharpe: {b_sharpe:.2f} → {a_sharpe:.2f} ({sign}{sharpe_d:.2f}) | "
        f"trades: {before['num_trades']} → {after['num_trades']} | "
        f"drawdown: {before['max_drawdown'] * 100:.1f}% → {after['max_drawdown'] * 100:.1f}% | "
        f"return: {before['total_return'] * 100:.1f}% → {after['total_return'] * 100:.1f}%"
    )


if __name__ == "__main__":
    cases = [
        (
            "Strong improvement (Sharpe 0.1→0.9, trades 3→12, drawdown improves)",
            {"sharpe": 0.1, "total_return": 0.05, "max_drawdown": 0.28, "win_rate": 0.4, "num_trades": 3, "profit_factor": 1.1},
            {"sharpe": 0.9, "total_return": 0.31, "max_drawdown": 0.18, "win_rate": 0.55, "num_trades": 12, "profit_factor": 1.8},
            "0.85–0.95",
        ),
        (
            "Slight regression (Sharpe 0.6→0.45, still enough trades)",
            {"sharpe": 0.6, "total_return": 0.22, "max_drawdown": 0.15, "win_rate": 0.52, "num_trades": 18, "profit_factor": 1.5},
            {"sharpe": 0.45, "total_return": 0.18, "max_drawdown": 0.17, "win_rate": 0.48, "num_trades": 15, "profit_factor": 1.3},
            "0.35–0.45",
        ),
        (
            "Degenerate strategy (Sharpe 1.4 but only 2 trades)",
            {"sharpe": 0.2, "total_return": 0.08, "max_drawdown": 0.20, "win_rate": 0.45, "num_trades": 10, "profit_factor": 1.2},
            {"sharpe": 1.4, "total_return": 0.45, "max_drawdown": 0.10, "win_rate": 0.80, "num_trades": 2, "profit_factor": 3.0},
            "0.45–0.50",
        ),
        (
            "No change",
            {"sharpe": 0.5, "total_return": 0.18, "max_drawdown": 0.20, "win_rate": 0.50, "num_trades": 14, "profit_factor": 1.4},
            {"sharpe": 0.5, "total_return": 0.18, "max_drawdown": 0.20, "win_rate": 0.50, "num_trades": 14, "profit_factor": 1.4},
            "~0.50",
        ),
    ]

    for label, before, after, expected in cases:
        reward = compute_reward(before, after)
        delta = describe_delta(before, after)
        print(f"{label}")
        print(f"  reward:   {reward}  (expected {expected})")
        print(f"  delta:    {delta}")
        print()
