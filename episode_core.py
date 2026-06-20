"""
Pure episode logic for StratRL — no dependency on `hud` or `modal`.

This module exists so that `modal_runner.py` (and its remote Modal container,
which never installs hud-python) can run episodes without importing `hud`.
`environment.py` wraps this module with the HUD `@env.scenario` decorator.
"""

from __future__ import annotations

import copy
import json
import os
import re

import anthropic
from dotenv import load_dotenv

from backtester import run_backtest
from mutations import MutationError, apply_mutation
from regime import classify_regime
from reward import compute_reward, describe_delta

load_dotenv()

MAX_TURNS = 8
MAX_CONSECUTIVE_INVALID = 3

MODEL = "claude-sonnet-4-5"
SYSTEM_PROMPT = (
    "You are a quantitative trading strategy optimizer. "
    "Your sole goal is to maximize the Sharpe ratio of a given strategy over 8 turns. "
    "Each turn output exactly ONE mutation as raw JSON with no prose before or after it. "
    "Explain your reasoning inside the JSON \"reasoning\" field."
)


def fetch_news_context(ticker: str) -> str:
    """Fetch and summarize recent news for a ticker via Exa + Claude Haiku.

    Returns a one-sentence summary, or a fallback string if unavailable.
    Never raises — safe to call during an unattended episode.
    """
    try:
        api_key = os.environ.get("EXA_API_KEY", "")
        if not api_key:
            return "No news context available."

        from datetime import datetime, timedelta

        from exa_py import Exa

        exa = Exa(api_key=api_key)
        cutoff = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

        results = exa.search_and_contents(
            f"{ticker} stock news",
            num_results=5,
            include_domains=[
                "finance.yahoo.com", "benzinga.com",
                "marketwatch.com", "seekingalpha.com",
            ],
            start_published_date=cutoff,
            highlights={"num_sentences": 2, "highlights_per_url": 1},
        )

        highlights = []
        for r in results.results:
            if r.highlights:
                highlights.extend(r.highlights)

        if not highlights:
            return "No news context available."

        combined = " ".join(highlights)
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=100,
            system=(
                "Summarize this news in one sentence from the perspective of a quantitative trader. "
                "Be specific about direction and magnitude."
            ),
            messages=[{"role": "user", "content": combined}],
        )
        return response.content[0].text.strip()

    except Exception:
        return "No news context available."


SEED_STRATEGIES = [
    {   # Seed 0 — worst: RSI overbought entry (buys when already extended)
        "ticker": "TSLA",
        "indicators": [{"type": "RSI", "period": 14, "name": "rsi_14"}],
        "entry_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 70}],
        "exit_conditions": [{"indicator": "rsi_14", "operator": "<", "value": 50}],
        "stop_loss": 0.10,
        "take_profit": 0.10,
    },
    {   # Seed 1 — bad: equal stop-loss and take-profit; transaction costs erase any edge
        "ticker": "NVDA",
        "indicators": [
            {"type": "SMA", "period": 20, "name": "sma_20"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "sma_20", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 55},
        ],
        "exit_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 55}],
        "stop_loss": 0.08,
        "take_profit": 0.08,
    },
    {   # Seed 2 — mediocre: shallow RSI threshold lets in too many marginal trades
        "ticker": "AAPL",
        "indicators": [
            {"type": "SMA", "period": 10, "name": "sma_10"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "sma_10", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 55},
        ],
        "exit_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 65}],
        "stop_loss": 0.05,
        "take_profit": 0.12,
    },
    {   # Seed 3 — decent: trend-following on SPY, exit triggers too early
        "ticker": "SPY",
        "indicators": [
            {"type": "EMA", "period": 20, "name": "ema_20"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "ema_20", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 50},
        ],
        "exit_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 65}],
        "stop_loss": 0.05,
        "take_profit": 0.15,
    },
    {   # Seed 4 — best: selective entry with RSI confirmation and asymmetric risk/reward
        "ticker": "TSLA",
        "indicators": [
            {"type": "SMA", "period": 20, "name": "sma_20"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "sma_20", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 45},
        ],
        "exit_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 65}],
        "stop_loss": 0.07,
        "take_profit": 0.20,
    },
]


def _extract_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "")
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return str(content)


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON object found in agent response", text, 0)


def _fmt_metrics(m: dict) -> str:
    return (
        f"  Sharpe:        {m['sharpe']}\n"
        f"  Total return:  {m['total_return'] * 100:.1f}%\n"
        f"  Max drawdown:  {m['max_drawdown'] * 100:.1f}%\n"
        f"  Win rate:      {m['win_rate'] * 100:.1f}%\n"
        f"  Trades:        {m['num_trades']}\n"
        f"  Profit factor: {m['profit_factor']}"
    )


def build_initial_observation(strategy: dict, metrics: dict, news_context: str = "") -> str:
    news_line = f"\nNews context: {news_context}" if news_context else ""
    regime = classify_regime(strategy["ticker"])
    regime_line = f"\nMarket regime: {regime}" if regime != "unknown" else ""
    return f"""You are a quantitative trading strategy optimizer. Your goal is to improve the Sharpe ratio of the following strategy over {MAX_TURNS} turns. Each turn, propose exactly ONE mutation as JSON.

Mutation types available:
- change_indicator_period: {{"type": "change_indicator_period", "indicator_name": "...", "new_period": int, "reasoning": "..."}}
- change_condition_threshold: {{"type": "change_condition_threshold", "indicator_name": "...", "condition_type": "entry"|"exit", "new_value": float, "reasoning": "..."}}
- add_indicator: {{"type": "add_indicator", "indicator": {{"type": "SMA"|"EMA"|"RSI"|"MACD"|"BB", "period": int, "name": "..."}}, "condition": {{"indicator": "...", "operator": ">"|"<"|">="|"<=", "value": float|"close"}}, "condition_type": "entry"|"exit", "reasoning": "..."}}
- remove_indicator: {{"type": "remove_indicator", "indicator_name": "...", "reasoning": "..."}}
- change_stop_loss: {{"type": "change_stop_loss", "new_value": float, "reasoning": "..."}}
- change_take_profit: {{"type": "change_take_profit", "new_value": float, "reasoning": "..."}}
- change_ticker: {{"type": "change_ticker", "new_ticker": "TSLA"|"NVDA"|"AAPL"|"SPY", "reasoning": "..."}}

Bounds (mutations outside these ranges are rejected):
- Indicator periods: SMA/EMA/BB 5-200, RSI/MACD 2-50
- RSI thresholds (condition value when indicator is RSI): 0-100
- SMA/EMA/BB/MACD thresholds: unbounded (compared against price or another indicator's scale)
- stop_loss: 0.001-0.50, take_profit: 0.01-5.0, and stop_loss must be < take_profit

Rules:
- One mutation per turn. Multiple mutations in one response will be rejected.
- Always include "reasoning" explaining why you're making this change.
- Output only the JSON — no prose before or after it.

Starting strategy:
{json.dumps(strategy, indent=2)}

Starting metrics:
{_fmt_metrics(metrics)}{regime_line}{news_line}

Propose your first mutation."""


def build_turn_observation(
    turn: int,
    strategy: dict,
    metrics: dict,
    before_metrics: dict,
    after_metrics: dict,
    reward: float,
    mutation_type: str,
    reasoning: str,
    history: list,
    news_context: str = "",
) -> str:
    delta = describe_delta(before_metrics, after_metrics)
    history_lines = "\n".join(
        f"  Turn {h['turn']}: {h['mutation']} → reward {h['reward']}"
        for h in history
    )
    news_line = f"\nNews context: {news_context}" if news_context else ""
    regime = classify_regime(strategy["ticker"])
    regime_line = f"\nMarket regime: {regime}" if regime != "unknown" else ""
    return f"""Turn {turn}/{MAX_TURNS} complete.

Mutation applied: {mutation_type}
Reasoning: {reasoning}
{delta}
Reward this turn: {reward}

Current strategy:
{json.dumps(strategy, indent=2)}

Current metrics:
{_fmt_metrics(metrics)}{regime_line}{news_line}

History so far:
{history_lines}

Propose your next mutation as JSON."""


def build_error_observation(
    consecutive_invalid: int,
    error_msg: str,
) -> str:
    return f"""Invalid mutation (attempt {consecutive_invalid}/{MAX_CONSECUTIVE_INVALID}):
{error_msg}

Current strategy is unchanged. Try again with a valid mutation JSON."""


def build_final_observation(
    reason: str,
    initial_metrics: dict,
    final_metrics: dict,
    history: list,
) -> str:
    history_lines = "\n".join(
        f"  Turn {h['turn']}: {h['mutation']} | reward {h['reward']} "
        f"| Sharpe {h['before_metrics']['sharpe']} → {h['after_metrics']['sharpe']}"
        for h in history
    ) or "  (no valid mutations completed)"
    return f"""Episode complete ({reason}).

Starting strategy metrics:  {initial_metrics}
Final strategy metrics:     {final_metrics}
Total improvement — Sharpe: {initial_metrics['sharpe']} → {final_metrics['sharpe']}

Full history:
{history_lines}"""


def process_turn(
    current_strategy: dict,
    current_metrics: dict,
    initial_metrics: dict,
    history: list,
    mutation_json_str: str,
    consecutive_invalid: int,
    turn: int,
    news_context: str = "",
) -> dict:
    """Apply one mutation, run backtest, compute reward. Returns updated state dict.

    Keys: valid, observation, reward, strategy, metrics, history, consecutive_invalid, done.
    """
    try:
        mutation = _extract_json(mutation_json_str)
        new_strategy = apply_mutation(current_strategy, mutation)
        new_metrics = run_backtest(new_strategy)
        reward = compute_reward(current_metrics, new_metrics)

        new_history = history + [
            {
                "turn": turn,
                "mutation": mutation.get("type", "unknown"),
                "reasoning": mutation.get("reasoning", ""),
                "before_metrics": current_metrics,
                "after_metrics": new_metrics,
                "reward": reward,
            }
        ]
        all_rewards = [h["reward"] for h in new_history]
        avg_reward = round(sum(all_rewards) / len(all_rewards), 4)

        is_final = turn >= MAX_TURNS
        if is_final:
            obs = build_final_observation("max turns reached", initial_metrics, new_metrics, new_history)
            return {
                "valid": True, "done": True, "observation": obs,
                "reward": avg_reward, "strategy": new_strategy,
                "metrics": new_metrics, "history": new_history,
                "consecutive_invalid": 0,
            }

        obs = build_turn_observation(
            turn, new_strategy, new_metrics, current_metrics, new_metrics,
            reward, mutation.get("type", "unknown"), mutation.get("reasoning", ""), new_history,
            news_context=news_context,
        )
        return {
            "valid": True, "done": False, "observation": obs,
            "reward": reward, "strategy": new_strategy,
            "metrics": new_metrics, "history": new_history,
            "consecutive_invalid": 0,
        }

    except (json.JSONDecodeError, ValueError, MutationError) as e:
        new_consecutive = consecutive_invalid + 1
        obs = build_error_observation(new_consecutive, str(e))
        return {
            "valid": False, "done": new_consecutive >= MAX_CONSECUTIVE_INVALID,
            "observation": obs, "reward": 0.0,
            "strategy": current_strategy, "metrics": current_metrics,
            "history": history, "consecutive_invalid": new_consecutive,
        }


def _seed_name(strategy: dict) -> str:
    ticker = strategy["ticker"].lower()
    indicators = "_".join(ind["type"].lower() for ind in strategy["indicators"])
    return f"{ticker}_{indicators}" if indicators else ticker


def run_episode_events(seed_strategy: dict):
    """Runs one full episode against the real Anthropic API, yielding one event
    dict per completed turn, then a final episode_complete event.

    Shared by modal_runner.py (collects into a list, since Modal's `.map()` can't
    be called on a generator function) and server/main.py's live SSE stream
    (consumes incrementally for true per-turn streaming, no Modal involved).
    """
    seed_name = _seed_name(seed_strategy)
    initial_strategy = copy.deepcopy(seed_strategy)
    initial_metrics = run_backtest(initial_strategy)
    news_context = fetch_news_context(initial_strategy["ticker"])

    current_strategy = copy.deepcopy(initial_strategy)
    current_metrics = copy.deepcopy(initial_metrics)
    history: list = []
    turn = 0
    consecutive_invalid = 0
    all_rewards: list[float] = []
    messages: list[dict] = []

    obs = build_initial_observation(initial_strategy, initial_metrics, news_context)
    messages.append({"role": "user", "content": obs})

    client = anthropic.Anthropic()

    while turn < MAX_TURNS and consecutive_invalid < MAX_CONSECUTIVE_INVALID:
        response = client.messages.create(
            model=MODEL, max_tokens=512, system=SYSTEM_PROMPT, messages=messages,
        )
        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})

        turn += 1
        result = process_turn(
            current_strategy, current_metrics, initial_metrics,
            history, assistant_text, consecutive_invalid, turn,
            news_context=news_context,
        )

        if result["valid"]:
            entry = result["history"][-1]
            yield {
                "type": "turn",
                "seed_name": seed_name,
                "turn": turn,
                "mutation_type": entry["mutation"],
                "reasoning": entry["reasoning"],
                "reward": result["reward"],
                "sharpe_before": entry["before_metrics"]["sharpe"],
                "sharpe_after": entry["after_metrics"]["sharpe"],
            }
            current_strategy = result["strategy"]
            current_metrics = result["metrics"]
            history = result["history"]
            all_rewards.append(result["reward"])
            consecutive_invalid = 0
        else:
            consecutive_invalid = result["consecutive_invalid"]

        if result["done"]:
            break

        messages.append({"role": "user", "content": result["observation"]})

    avg_reward = round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else 0.0

    yield {
        "type": "episode_complete",
        "seed_name": seed_name,
        "initial_sharpe": initial_metrics["sharpe"],
        "final_sharpe": current_metrics["sharpe"],
        "total_reward": avg_reward,
        "turns_taken": turn,
        "ticker": initial_strategy["ticker"],
    }


if __name__ == "__main__":
    print("Smoke test: single turn end-to-end (no HUD)\n")

    strategy = copy.deepcopy(SEED_STRATEGIES[0])
    metrics = run_backtest(strategy)

    print(f"Seed strategy: {strategy['ticker']} | Sharpe: {metrics['sharpe']} | Trades: {metrics['num_trades']}\n")

    mutation = json.dumps({
        "type": "change_stop_loss",
        "new_value": 0.05,
        "reasoning": "Tightening stop loss from 10% to 5% to reduce drawdown on volatile TSLA positions.",
    })

    result = process_turn(strategy, metrics, metrics, [], mutation, 0, 1)

    print("=== Turn 1 Observation ===")
    print(result["observation"])
    print()
    print(f"reward: {result['reward']}")
    print(f"valid:  {result['valid']}")
    print(f"done:   {result['done']}")
