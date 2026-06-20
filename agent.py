from __future__ import annotations

import copy
from pathlib import Path

import anthropic
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dotenv import load_dotenv

from backtester import get_equity_curve, run_backtest
from episode_core import (
    MAX_CONSECUTIVE_INVALID as _MAX_CONSECUTIVE_INVALID,
    MAX_TURNS as _MAX_TURNS,
    SEED_STRATEGIES,
    build_initial_observation as _build_initial_observation,
    process_turn as _process_turn,
)

load_dotenv()

_MODEL = "claude-sonnet-4-5"
_SYSTEM = (
    "You are a quantitative trading strategy optimizer. "
    "Your sole goal is to maximize the Sharpe ratio of a given strategy over 8 turns. "
    "Each turn output exactly ONE mutation as raw JSON with no prose before or after it. "
    "Explain your reasoning inside the JSON \"reasoning\" field."
)
_SEP = "━" * 40


def _print_turn(turn: int, result: dict) -> None:
    entry = result["history"][-1]
    before = entry["before_metrics"]
    after = entry["after_metrics"]

    dd_label = "improved" if after["max_drawdown"] < before["max_drawdown"] else "worsened"

    print(f"\n{_SEP}")
    print(f"TURN {turn} / {_MAX_TURNS}")
    print(_SEP)
    print(f"Mutation:   {entry['mutation']}")
    print(f"Reasoning:  {entry['reasoning']}")
    print(f"Reward:     {result['reward']}")
    print()
    print(f"Sharpe:     {before['sharpe']:.2f} → {after['sharpe']:.2f}  ({after['sharpe'] - before['sharpe']:+.2f})")
    print(f"Return:     {before['total_return'] * 100:.1f}% → {after['total_return'] * 100:.1f}% ({(after['total_return'] - before['total_return']) * 100:+.1f}%)")
    print(f"Drawdown:   {before['max_drawdown'] * 100:.1f}% → {after['max_drawdown'] * 100:.1f}% ({dd_label})")
    print(f"Trades:     {before['num_trades']} → {after['num_trades']}")


def _render_equity_chart(
    initial_strategy: dict,
    final_strategy: dict,
    initial_sharpe: float,
    final_sharpe: float,
) -> None:
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    initial_curve = get_equity_curve(initial_strategy)
    final_curve = get_equity_curve(final_strategy)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        initial_curve.index, initial_curve.values,
        label=f"Initial (Sharpe: {initial_sharpe:.2f})", color="crimson", alpha=0.7,
    )
    ax.plot(
        final_curve.index, final_curve.values,
        label=f"Final (Sharpe: {final_sharpe:.2f})", color="seagreen", alpha=0.9,
    )
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_title(f"StratRL: {initial_sharpe:.2f} → {final_sharpe:.2f} Sharpe")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = results_dir / "equity_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nEquity curve saved to {out_path}")
    plt.show()


def _render_equity_chart_plotly(
    initial_strategy: dict,
    final_strategy: dict,
    initial_sharpe: float,
    final_sharpe: float,
) -> None:
    """Interactive equivalent of _render_equity_chart, for live demo projection.
    Saves both an interactive HTML (hover/zoom) and a static PNG (for slides)."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    initial_curve = get_equity_curve(initial_strategy)
    final_curve = get_equity_curve(final_strategy)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=initial_curve.index, y=initial_curve.values,
        name=f"Initial (Sharpe: {initial_sharpe:.2f})",
        line=dict(color="crimson"), opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=final_curve.index, y=final_curve.values,
        name=f"Final (Sharpe: {final_sharpe:.2f})",
        line=dict(color="seagreen"),
    ))
    fig.add_hline(y=1.0, line=dict(color="gray", width=0.8, dash="dash"))
    fig.update_layout(
        title=f"StratRL: {initial_sharpe:.2f} → {final_sharpe:.2f} Sharpe",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (normalized)",
        template="plotly_white",
    )

    html_path = results_dir / "equity_curve.html"
    fig.write_html(html_path)
    print(f"Interactive equity curve saved to {html_path}")

    try:
        png_path = results_dir / "equity_curve_plotly.png"
        fig.write_image(png_path)
        print(f"Static equity curve (plotly) saved to {png_path}")
    except Exception as e:
        print(f"Skipped static PNG export (kaleido not available?): {e}")


def run_episode(seed: dict | None = None) -> dict:
    client = anthropic.Anthropic()

    initial_strategy = copy.deepcopy(seed if seed is not None else SEED_STRATEGIES[0])
    initial_metrics = run_backtest(initial_strategy)

    print(f"\nStarting strategy: {initial_strategy['ticker']}")
    print(
        f"Sharpe: {initial_metrics['sharpe']} | "
        f"Trades: {initial_metrics['num_trades']} | "
        f"Return: {initial_metrics['total_return'] * 100:.1f}%"
    )

    messages: list[dict] = []
    current_strategy = copy.deepcopy(initial_strategy)
    current_metrics = copy.deepcopy(initial_metrics)
    history: list = []
    turn = 0
    consecutive_invalid = 0
    all_rewards: list[float] = []

    obs = _build_initial_observation(initial_strategy, initial_metrics)
    messages.append({"role": "user", "content": obs})

    while turn < _MAX_TURNS and consecutive_invalid < _MAX_CONSECUTIVE_INVALID:
        response = client.messages.create(
            model=_MODEL,
            max_tokens=512,
            system=_SYSTEM,
            messages=messages,
        )
        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        turn += 1
        result = _process_turn(
            current_strategy, current_metrics, initial_metrics,
            history, assistant_text, consecutive_invalid, turn,
        )

        if result["valid"]:
            _print_turn(turn, result)
            current_strategy = result["strategy"]
            current_metrics = result["metrics"]
            history = result["history"]
            all_rewards.append(result["reward"])
            consecutive_invalid = 0
        else:
            consecutive_invalid = result["consecutive_invalid"]
            print(f"\n[Turn {turn}] Invalid mutation ({consecutive_invalid}/{_MAX_CONSECUTIVE_INVALID}):")
            print(f"  {result['observation'].splitlines()[1] if len(result['observation'].splitlines()) > 1 else ''}")

        if result["done"]:
            break

        messages.append({"role": "user", "content": result["observation"]})

    avg_reward = round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else 0.0
    reason = "max turns reached" if turn >= _MAX_TURNS else "too many invalid mutations"

    print(f"\n{_SEP}")
    print(f"Episode complete — {reason}")
    print(f"Sharpe: {initial_metrics['sharpe']} → {current_metrics['sharpe']}")
    print(f"Return: {initial_metrics['total_return'] * 100:.1f}% → {current_metrics['total_return'] * 100:.1f}%")
    print(f"Avg reward: {avg_reward}")
    print(_SEP)

    return {
        "initial_strategy": initial_strategy,
        "final_strategy": current_strategy,
        "initial_metrics": initial_metrics,
        "final_metrics": current_metrics,
        "history": history,
        "total_reward": avg_reward,
        "turns": turn,
    }


if __name__ == "__main__":
    result = run_episode()
    print("\nFinal equity curve:")
    _render_equity_chart(
        result["initial_strategy"], result["final_strategy"],
        result["initial_metrics"]["sharpe"], result["final_metrics"]["sharpe"],
    )
    _render_equity_chart_plotly(
        result["initial_strategy"], result["final_strategy"],
        result["initial_metrics"]["sharpe"], result["final_metrics"]["sharpe"],
    )
