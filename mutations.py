import copy
import json

_VALID_TICKERS = {"TSLA", "NVDA", "AAPL", "SPY"}
_VALID_INDICATOR_TYPES = {"SMA", "EMA", "RSI", "MACD", "BB"}
_VALID_OPERATORS = {">", "<", ">=", "<="}

# Per-indicator-type period bounds (inclusive). MACD's `period` field is currently
# unused by the backtester (its fast/slow/signal spans are hardcoded to 12/26/9 in
# backtester.py) — bounds are still enforced here so a mutation can't set a
# nonsensical value that would be confusing if MACD periods become configurable later.
_PERIOD_BOUNDS: dict[str, tuple[int, int]] = {
    "SMA": (5, 200),
    "EMA": (5, 200),
    "RSI": (2, 50),
    "MACD": (2, 50),
    "BB": (5, 200),
}

# RSI is a 0-100 oscillator, so its condition thresholds are bounded. Other indicator
# types (SMA/EMA/BB/MACD) are compared against price or another indicator's scale and
# are left unbounded.
_THRESHOLD_BOUNDS: dict[str, tuple[float, float]] = {
    "RSI": (0.0, 100.0),
}


def _period_bounds_for(ind_type: str) -> tuple[int, int]:
    return _PERIOD_BOUNDS.get(ind_type, (2, 200))


def _check_period_bounds(ind_type: str, period: int, ind_name: str) -> None:
    lo, hi = _period_bounds_for(ind_type)
    if period is None or not (lo <= period <= hi):
        raise MutationError(
            f"indicator '{ind_name}' ({ind_type}) has period {period} — "
            f"{ind_type} period must be between {lo} and {hi}"
        )


def _check_threshold_bounds(ind_type: str, value, ind_name: str) -> None:
    if value == "close" or ind_type not in _THRESHOLD_BOUNDS:
        return
    lo, hi = _THRESHOLD_BOUNDS[ind_type]
    if not (lo <= float(value) <= hi):
        raise MutationError(
            f"threshold {value} for indicator '{ind_name}' ({ind_type}) is out of range — "
            f"{ind_type} thresholds must be between {lo} and {hi}"
        )


class MutationError(Exception):
    pass


def _indicator_names(strategy: dict) -> list[str]:
    return [ind["name"] for ind in strategy.get("indicators", [])]


def _find_indicator(strategy: dict, name: str) -> dict:
    for ind in strategy.get("indicators", []):
        if ind["name"] == name:
            return ind
    available = ", ".join(_indicator_names(strategy)) or "none"
    raise MutationError(
        f"indicator '{name}' not found in strategy — available indicators: {available}"
    )


def validate_strategy(strategy: dict) -> None:
    """
    Validate a strategy dict for internal consistency.
    Raises MutationError with a descriptive message if invalid.
    Called automatically by apply_mutation after every mutation.
    """
    ticker = strategy.get("ticker")
    if ticker not in _VALID_TICKERS:
        raise MutationError(
            f"ticker '{ticker}' is not supported — available tickers: {sorted(_VALID_TICKERS)}"
        )

    indicators = strategy.get("indicators", [])
    ind_types_by_name: dict[str, str] = {}
    seen_names: set[str] = set()
    for ind in indicators:
        t = ind.get("type")
        if t not in _VALID_INDICATOR_TYPES:
            raise MutationError(
                f"indicator type '{t}' is not supported — valid types: {sorted(_VALID_INDICATOR_TYPES)}"
            )
        name = ind.get("name")
        _check_period_bounds(t, ind.get("period"), name)
        if name in seen_names:
            raise MutationError(f"duplicate indicator name '{name}' in indicators list")
        seen_names.add(name)
        ind_types_by_name[name] = t

    stop_loss = strategy.get("stop_loss")
    take_profit = strategy.get("take_profit")

    if not (0.001 <= stop_loss <= 0.50):
        raise MutationError(
            f"stop_loss {stop_loss} is out of range — must be between 0.001 and 0.50"
        )
    if not (0.01 <= take_profit <= 5.0):
        raise MutationError(
            f"take_profit {take_profit} is out of range — must be between 0.01 and 5.0"
        )
    if stop_loss >= take_profit:
        raise MutationError(
            f"stop_loss ({stop_loss}) must be less than take_profit ({take_profit})"
        )

    all_conditions = strategy.get("entry_conditions", []) + strategy.get("exit_conditions", [])
    for cond in all_conditions:
        ind_name = cond.get("indicator")
        if ind_name not in seen_names:
            raise MutationError(
                f"condition references indicator '{ind_name}' which is not defined — "
                f"defined indicators: {', '.join(seen_names) or 'none'}"
            )
        _check_threshold_bounds(ind_types_by_name[ind_name], cond.get("value"), ind_name)

    if not strategy.get("entry_conditions"):
        raise MutationError(
            "strategy has no entry conditions — the agent cannot enter any trades"
        )


def _apply_change_indicator_period(strategy: dict, mutation: dict) -> dict:
    ind_name = mutation.get("indicator_name")
    new_period = mutation.get("new_period")

    _find_indicator(strategy, ind_name)

    if new_period is None or new_period < 2:
        raise MutationError(
            f"new_period {new_period} is invalid — period must be >= 2"
        )

    ind = _find_indicator(strategy, ind_name)
    ind_type = ind["type"].lower()
    new_name = f"{ind_type}_{new_period}"

    # Ensure the new name doesn't collide with a different existing indicator
    existing_names = _indicator_names(strategy)
    if new_name in existing_names and new_name != ind_name:
        raise MutationError(
            f"renaming '{ind_name}' to '{new_name}' would collide with an existing indicator"
        )

    ind["period"] = new_period
    ind["name"] = new_name

    for cond in strategy.get("entry_conditions", []) + strategy.get("exit_conditions", []):
        if cond["indicator"] == ind_name:
            cond["indicator"] = new_name

    return strategy


def _apply_change_condition_threshold(strategy: dict, mutation: dict) -> dict:
    ind_name = mutation.get("indicator_name")
    condition_type = mutation.get("condition_type")
    new_value = mutation.get("new_value")

    _find_indicator(strategy, ind_name)

    if condition_type not in ("entry", "exit"):
        raise MutationError(
            f"condition_type '{condition_type}' is invalid — must be 'entry' or 'exit'"
        )

    cond_list = strategy.get(f"{condition_type}_conditions", [])
    matching = [c for c in cond_list if c["indicator"] == ind_name]
    if not matching:
        raise MutationError(
            f"no {condition_type} condition references indicator '{ind_name}' — "
            f"cannot change threshold"
        )

    for cond in matching:
        cond["value"] = new_value

    return strategy


def _apply_add_indicator(strategy: dict, mutation: dict) -> dict:
    new_ind = mutation.get("indicator", {})
    condition = mutation.get("condition", {})
    condition_type = mutation.get("condition_type")

    ind_type = new_ind.get("type")
    ind_name = new_ind.get("name")

    if ind_type not in _VALID_INDICATOR_TYPES:
        raise MutationError(
            f"indicator type '{ind_type}' is not supported — valid types: {sorted(_VALID_INDICATOR_TYPES)}"
        )

    existing_names = _indicator_names(strategy)
    if ind_name in existing_names:
        raise MutationError(
            f"indicator name '{ind_name}' already exists in strategy — choose a different name"
        )

    if condition_type not in ("entry", "exit"):
        raise MutationError(
            f"condition_type '{condition_type}' is invalid — must be 'entry' or 'exit'"
        )

    strategy["indicators"].append(new_ind)
    strategy[f"{condition_type}_conditions"].append(condition)

    return strategy


def _apply_remove_indicator(strategy: dict, mutation: dict) -> dict:
    ind_name = mutation.get("indicator_name")

    _find_indicator(strategy, ind_name)

    entry_after_removal = [
        c for c in strategy.get("entry_conditions", []) if c["indicator"] != ind_name
    ]
    if not entry_after_removal:
        raise MutationError(
            f"removing '{ind_name}' would leave the strategy with no entry conditions — "
            f"add a replacement entry condition before removing this indicator"
        )

    strategy["indicators"] = [i for i in strategy["indicators"] if i["name"] != ind_name]
    strategy["entry_conditions"] = entry_after_removal
    strategy["exit_conditions"] = [
        c for c in strategy.get("exit_conditions", []) if c["indicator"] != ind_name
    ]

    return strategy


def _apply_change_stop_loss(strategy: dict, mutation: dict) -> dict:
    new_value = mutation.get("new_value")

    if not (0.001 <= new_value <= 0.50):
        raise MutationError(
            f"new stop_loss {new_value} is out of range — must be between 0.001 and 0.50"
        )

    strategy["stop_loss"] = new_value
    return strategy


def _apply_change_take_profit(strategy: dict, mutation: dict) -> dict:
    new_value = mutation.get("new_value")

    if not (0.01 <= new_value <= 5.0):
        raise MutationError(
            f"new take_profit {new_value} is out of range — must be between 0.01 and 5.0"
        )

    strategy["take_profit"] = new_value
    return strategy


def _apply_change_ticker(strategy: dict, mutation: dict) -> dict:
    new_ticker = mutation.get("new_ticker")
    current = strategy.get("ticker")

    if new_ticker not in _VALID_TICKERS:
        raise MutationError(
            f"ticker '{new_ticker}' is not supported — available tickers: {sorted(_VALID_TICKERS)}"
        )
    if new_ticker == current:
        raise MutationError(
            f"new_ticker '{new_ticker}' is the same as the current ticker — no change made"
        )

    strategy["ticker"] = new_ticker
    return strategy


_MUTATION_HANDLERS = {
    "change_indicator_period": _apply_change_indicator_period,
    "change_condition_threshold": _apply_change_condition_threshold,
    "add_indicator": _apply_add_indicator,
    "remove_indicator": _apply_remove_indicator,
    "change_stop_loss": _apply_change_stop_loss,
    "change_take_profit": _apply_change_take_profit,
    "change_ticker": _apply_change_ticker,
}


def apply_mutation(strategy: dict, mutation: dict) -> dict:
    """
    Apply a mutation to a strategy. Returns a new strategy dict (never mutates the input).
    Raises MutationError with a descriptive message if the mutation is invalid or
    produces an illegal strategy.
    """
    mutation_type = mutation.get("type")
    handler = _MUTATION_HANDLERS.get(mutation_type)
    if handler is None:
        valid = ", ".join(sorted(_MUTATION_HANDLERS))
        raise MutationError(
            f"unknown mutation type '{mutation_type}' — valid types: {valid}"
        )

    result = copy.deepcopy(strategy)
    result = handler(result, mutation)
    validate_strategy(result)
    return result


if __name__ == "__main__":
    base = {
        "ticker": "TSLA",
        "indicators": [
            {"type": "SMA", "period": 20, "name": "sma_20"},
            {"type": "RSI", "period": 14, "name": "rsi_14"},
        ],
        "entry_conditions": [
            {"indicator": "sma_20", "operator": ">", "value": "close"},
            {"indicator": "rsi_14", "operator": "<", "value": 40},
        ],
        "exit_conditions": [
            {"indicator": "rsi_14", "operator": ">", "value": 70},
        ],
        "stop_loss": 0.05,
        "take_profit": 0.15,
    }

    def check(label: str, fn):
        try:
            fn()
            print(f"  PASS  {label}")
        except Exception as e:
            print(f"  FAIL  {label}: {e}")

    def expect_error(label: str, fn):
        try:
            fn()
            print(f"  FAIL  {label}: expected MutationError but none was raised")
        except MutationError as e:
            print(f"  PASS  {label}: MutationError — {e}")
        except Exception as e:
            print(f"  FAIL  {label}: wrong exception type {type(e).__name__}: {e}")

    print("Running mutation tests...\n")

    def test_change_period():
        result = apply_mutation(base, {
            "type": "change_indicator_period",
            "indicator_name": "sma_20",
            "new_period": 50,
            "reasoning": "testing",
        })
        names = [i["name"] for i in result["indicators"]]
        cond_refs = [c["indicator"] for c in result["entry_conditions"]]
        assert "sma_50" in names, f"sma_50 not in indicators: {names}"
        assert "sma_20" not in names, f"sma_20 still in indicators: {names}"
        assert "sma_50" in cond_refs, f"sma_50 not in entry_conditions: {cond_refs}"
        assert "sma_20" not in cond_refs, f"sma_20 still in entry_conditions: {cond_refs}"
        assert base["indicators"][0]["name"] == "sma_20", "input strategy was mutated"

    def test_change_threshold():
        result = apply_mutation(base, {
            "type": "change_condition_threshold",
            "indicator_name": "rsi_14",
            "condition_type": "entry",
            "new_value": 30,
            "reasoning": "testing",
        })
        rsi_entry = next(c for c in result["entry_conditions"] if c["indicator"] == "rsi_14")
        assert rsi_entry["value"] == 30, f"expected 30, got {rsi_entry['value']}"
        original = next(c for c in base["entry_conditions"] if c["indicator"] == "rsi_14")
        assert original["value"] == 40, "input strategy was mutated"

    def test_add_indicator():
        result = apply_mutation(base, {
            "type": "add_indicator",
            "indicator": {"type": "EMA", "period": 50, "name": "ema_50"},
            "condition": {"indicator": "ema_50", "operator": ">", "value": "close"},
            "condition_type": "entry",
            "reasoning": "testing",
        })
        names = [i["name"] for i in result["indicators"]]
        cond_refs = [c["indicator"] for c in result["entry_conditions"]]
        assert "ema_50" in names, f"ema_50 not in indicators: {names}"
        assert "ema_50" in cond_refs, f"ema_50 not in entry_conditions: {cond_refs}"

    def test_remove_indicator():
        result = apply_mutation(base, {
            "type": "remove_indicator",
            "indicator_name": "sma_20",
            "reasoning": "testing",
        })
        names = [i["name"] for i in result["indicators"]]
        all_cond_refs = (
            [c["indicator"] for c in result["entry_conditions"]] +
            [c["indicator"] for c in result["exit_conditions"]]
        )
        assert "sma_20" not in names, f"sma_20 still in indicators: {names}"
        assert "sma_20" not in all_cond_refs, f"sma_20 still referenced in conditions"

    def test_change_ticker():
        result = apply_mutation(base, {
            "type": "change_ticker",
            "new_ticker": "NVDA",
            "reasoning": "testing",
        })
        assert result["ticker"] == "NVDA", f"expected NVDA, got {result['ticker']}"
        assert base["ticker"] == "TSLA", "input strategy was mutated"

    def test_invalid_period_nonexistent():
        apply_mutation(base, {
            "type": "change_indicator_period",
            "indicator_name": "bollinger_20",
            "new_period": 50,
            "reasoning": "testing",
        })

    def test_remove_last_entry_condition():
        # sma_20 is the only indicator with an entry condition referencing it
        # removing it would leave rsi_14 as the only entry condition — that's fine
        # so we need a strategy where removing the indicator kills ALL entry conditions
        single_entry_strategy = {
            "ticker": "TSLA",
            "indicators": [
                {"type": "SMA", "period": 20, "name": "sma_20"},
                {"type": "RSI", "period": 14, "name": "rsi_14"},
            ],
            "entry_conditions": [
                {"indicator": "sma_20", "operator": ">", "value": "close"},
            ],
            "exit_conditions": [
                {"indicator": "rsi_14", "operator": ">", "value": 70},
            ],
            "stop_loss": 0.05,
            "take_profit": 0.15,
        }
        apply_mutation(single_entry_strategy, {
            "type": "remove_indicator",
            "indicator_name": "sma_20",
            "reasoning": "testing",
        })

    def test_invalid_stop_loss():
        apply_mutation(base, {
            "type": "change_stop_loss",
            "new_value": 0.99,
            "reasoning": "testing",
        })

    def test_sma_period_above_bound():
        apply_mutation(base, {
            "type": "change_indicator_period",
            "indicator_name": "sma_20",
            "new_period": 500,
            "reasoning": "testing",
        })

    def test_sma_period_below_bound():
        apply_mutation(base, {
            "type": "change_indicator_period",
            "indicator_name": "sma_20",
            "new_period": 3,
            "reasoning": "testing",
        })

    def test_rsi_period_within_sma_range_but_out_of_rsi_range():
        # 100 is within SMA's (5, 200) but outside RSI's (2, 50)
        apply_mutation(base, {
            "type": "change_indicator_period",
            "indicator_name": "rsi_14",
            "new_period": 100,
            "reasoning": "testing",
        })

    def test_rsi_threshold_above_100():
        apply_mutation(base, {
            "type": "change_condition_threshold",
            "indicator_name": "rsi_14",
            "condition_type": "entry",
            "new_value": 150,
            "reasoning": "testing",
        })

    def test_rsi_threshold_below_0():
        apply_mutation(base, {
            "type": "change_condition_threshold",
            "indicator_name": "rsi_14",
            "condition_type": "exit",
            "new_value": -10,
            "reasoning": "testing",
        })

    def test_rsi_threshold_in_bounds():
        result = apply_mutation(base, {
            "type": "change_condition_threshold",
            "indicator_name": "rsi_14",
            "condition_type": "entry",
            "new_value": 25,
            "reasoning": "testing",
        })
        rsi_entry = next(c for c in result["entry_conditions"] if c["indicator"] == "rsi_14")
        assert rsi_entry["value"] == 25, f"expected 25, got {rsi_entry['value']}"

    def test_add_indicator_with_out_of_bounds_rsi_threshold():
        apply_mutation(base, {
            "type": "add_indicator",
            "indicator": {"type": "RSI", "period": 14, "name": "rsi_14_b"},
            "condition": {"indicator": "rsi_14_b", "operator": ">", "value": 200},
            "condition_type": "entry",
            "reasoning": "testing",
        })

    def test_sma_threshold_against_close_is_unbounded():
        # SMA/EMA conditions compare against "close" or price-scale values — no bound applies
        result = apply_mutation(base, {
            "type": "change_condition_threshold",
            "indicator_name": "sma_20",
            "condition_type": "entry",
            "new_value": 9999,
            "reasoning": "testing",
        })
        sma_entry = next(c for c in result["entry_conditions"] if c["indicator"] == "sma_20")
        assert sma_entry["value"] == 9999, f"expected 9999, got {sma_entry['value']}"

    check("change_indicator_period: sma_20 → sma_50, all refs updated", test_change_period)
    check("change_condition_threshold: rsi_14 entry value → 30", test_change_threshold)
    check("add_indicator: EMA 50 + entry condition", test_add_indicator)
    check("remove_indicator: sma_20 and its conditions removed", test_remove_indicator)
    check("change_ticker: TSLA → NVDA", test_change_ticker)
    check("change_condition_threshold: rsi_14 entry value → 25 (in bounds)", test_rsi_threshold_in_bounds)
    check("change_condition_threshold: sma_20 vs 9999 (unbounded, price-scale)", test_sma_threshold_against_close_is_unbounded)
    expect_error("invalid mutation: non-existent indicator 'bollinger_20'", test_invalid_period_nonexistent)
    expect_error("invalid mutation: remove only-entry-condition indicator", test_remove_last_entry_condition)
    expect_error("invalid mutation: stop_loss 0.99 exceeds max 0.50", test_invalid_stop_loss)
    expect_error("invalid mutation: SMA period 500 exceeds max 200", test_sma_period_above_bound)
    expect_error("invalid mutation: SMA period 3 below min 5", test_sma_period_below_bound)
    expect_error("invalid mutation: RSI period 100 exceeds RSI max 50", test_rsi_period_within_sma_range_but_out_of_rsi_range)
    expect_error("invalid mutation: RSI threshold 150 exceeds max 100", test_rsi_threshold_above_100)
    expect_error("invalid mutation: RSI threshold -10 below min 0", test_rsi_threshold_below_0)
    expect_error("invalid mutation: add_indicator RSI with threshold 200", test_add_indicator_with_out_of_bounds_rsi_threshold)
