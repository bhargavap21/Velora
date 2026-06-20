import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

_TICKERS = ["TSLA", "NVDA", "AAPL", "SPY"]
_START = "2022-01-01"
_END = "2024-12-31"
_TC = 0.001  # one-way transaction cost

_DATA_DIR = Path(__file__).parent / "data"


def _load_data() -> dict[str, pd.DataFrame]:
    _DATA_DIR.mkdir(exist_ok=True)
    result = {}
    missing = []
    for ticker in _TICKERS:
        path = _DATA_DIR / f"{ticker}.parquet"
        if path.exists():
            result[ticker] = pd.read_parquet(path)
        else:
            missing.append(ticker)

    if missing:
        raw = yf.download(
            missing,
            start=_START,
            end=_END,
            auto_adjust=True,
            progress=False,
        )
        for ticker in missing:
            if len(missing) == 1:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                df = raw.xs(ticker, axis=1, level=1)[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = pd.to_datetime(df.index)
            df.dropna(inplace=True)
            df.to_parquet(_DATA_DIR / f"{ticker}.parquet")
            result[ticker] = df

    return result


_DATA: dict[str, pd.DataFrame] = _load_data()


def get_available_tickers() -> list[str]:
    return list(_TICKERS)


def _compute_indicators(df: pd.DataFrame, indicators: list[dict]) -> pd.DataFrame:
    out = df.copy()
    close = df["Close"]

    for ind in indicators:
        t = ind["type"]
        name = ind["name"]
        period = ind.get("period", 14)

        if t == "SMA":
            out[name] = close.rolling(window=period).mean()

        elif t == "EMA":
            out[name] = close.ewm(span=period, adjust=False).mean()

        elif t == "RSI":
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            # Wilder smoothing (alpha=1/period) matches the industry-standard RSI definition
            avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            out[name] = 100 - (100 / (1 + rs))

        elif t == "MACD":
            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            out[name] = macd_line
            out[f"{name}_signal"] = macd_line.ewm(span=9, adjust=False).mean()

        elif t == "BB":
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            out[f"{name}_upper"] = sma + 2 * std
            out[f"{name}_mid"] = sma
            out[f"{name}_lower"] = sma - 2 * std

        else:
            raise ValueError(f"Unknown indicator type: '{t}'")

    return out


def _warmup_period(indicators: list[dict]) -> int:
    longest = 0
    for ind in indicators:
        t = ind["type"]
        period = ind.get("period", 14)
        if t == "MACD":
            longest = max(longest, 26 + 9)
        else:
            longest = max(longest, period)
    return longest


def _resolve_value(val, row: pd.Series) -> float:
    if val == "close":
        return float(row["Close"])
    return float(val)


def _eval_condition(cond: dict, row: pd.Series) -> bool:
    indicator_val = float(row[cond["indicator"]])
    compare_val = _resolve_value(cond["value"], row)
    op = cond["operator"]
    if op == ">":
        return indicator_val > compare_val
    elif op == "<":
        return indicator_val < compare_val
    elif op == ">=":
        return indicator_val >= compare_val
    elif op == "<=":
        return indicator_val <= compare_val
    elif op == "==":
        return indicator_val == compare_val
    else:
        raise ValueError(f"Unknown operator: '{op}'")


def _simulate(strategy: dict) -> tuple[dict, list[float], pd.DatetimeIndex]:
    """Core simulation engine. Returns (metrics, equity_curve, dates).

    equity_curve has n+1 elements (index 0 is pre-episode equity=1.0).
    dates has n elements aligned with equity_curve[1:].
    """
    ticker = strategy.get("ticker")
    if ticker not in _DATA:
        raise ValueError(f"Ticker '{ticker}' not in cached data. Available: {_TICKERS}")

    indicators = strategy.get("indicators", [])
    entry_conditions = strategy.get("entry_conditions", [])
    exit_conditions = strategy.get("exit_conditions", [])
    stop_loss = strategy.get("stop_loss", 0.05)
    take_profit = strategy.get("take_profit", 0.15)

    ind_names: set[str] = set()
    for ind in indicators:
        t = ind["type"]
        name = ind["name"]
        if t == "MACD":
            ind_names.add(name)
            ind_names.add(f"{name}_signal")
        elif t == "BB":
            ind_names.add(f"{name}_upper")
            ind_names.add(f"{name}_mid")
            ind_names.add(f"{name}_lower")
        else:
            ind_names.add(name)

    for cond in entry_conditions + exit_conditions:
        if cond["indicator"] not in ind_names:
            raise ValueError(
                f"Condition references unknown indicator name '{cond['indicator']}'. "
                f"Defined indicators: {ind_names}"
            )

    df_ind = _compute_indicators(_DATA[ticker], indicators)
    warmup = _warmup_period(indicators)
    dates = pd.DatetimeIndex(_DATA[ticker].index[warmup:])
    df = df_ind.iloc[warmup:].reset_index(drop=True)

    n = len(df)
    _empty_metrics = {
        "sharpe": 0.0, "total_return": 0.0, "max_drawdown": 0.0,
        "win_rate": 0.0, "num_trades": 0, "profit_factor": 0.0,
    }
    if n < 2:
        return _empty_metrics, [1.0], dates[:0]

    equity = 1.0
    equity_curve: list[float] = [equity]
    daily_returns: list[float] = []
    trades: list[float] = []

    in_position = False
    entry_price = 0.0
    pending_entry = False
    pending_exit = False
    prev_close: float | None = None  # last close while in a position (for daily return ref)

    for i in range(n):
        row = df.iloc[i]
        close = float(row["Close"])
        open_price = float(row["Open"])

        just_entered = False

        if pending_entry:
            entry_price = open_price * (1 + _TC)
            in_position = True
            pending_entry = False
            just_entered = True

        if pending_exit:
            exit_price = open_price * (1 - _TC)
            pnl = (exit_price - entry_price) / entry_price
            trades.append(pnl)
            # Return for exit day is measured from last in-position close to today's exit open.
            # prev_close is guaranteed non-None here: pending_exit only set while in_position,
            # which requires at least one prior in-position day to have set prev_close.
            daily_ret = (exit_price / prev_close) - 1  # type: ignore[operator]
            in_position = False
            pending_exit = False
            prev_close = None

        elif in_position:
            if just_entered:
                # No prior close reference; compute return from entry open to today's close
                daily_ret = (close / entry_price) - 1
            else:
                daily_ret = (close / prev_close) - 1  # type: ignore[operator]
            prev_close = close

        else:
            daily_ret = 0.0

        equity *= 1.0 + daily_ret
        daily_returns.append(daily_ret)
        equity_curve.append(equity)

        if i >= n - 1:
            break

        if not in_position and not pending_entry:
            if entry_conditions:
                if all(_eval_condition(c, row) for c in entry_conditions):
                    pending_entry = True
            else:
                pending_entry = False

        elif in_position:
            force_exit = i == n - 2
            sl_hit = close <= entry_price * (1 - stop_loss)
            tp_hit = close >= entry_price * (1 + take_profit)
            cond_exit = (
                any(_eval_condition(c, row) for c in exit_conditions)
                if exit_conditions
                else False
            )
            if force_exit or sl_hit or tp_hit or cond_exit:
                pending_exit = True

    num_trades = len(trades)
    total_return = equity - 1.0

    if num_trades < 2:
        sharpe = 0.0
    else:
        ret_arr = np.array(daily_returns)
        std = ret_arr.std()
        sharpe = float(ret_arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    curve_arr = np.array(equity_curve)
    peaks = np.maximum.accumulate(curve_arr)
    drawdowns = (peaks - curve_arr) / np.where(peaks > 0, peaks, 1.0)
    max_drawdown = float(drawdowns.max())

    if num_trades == 0:
        win_rate = 0.0
        profit_factor = 0.0
    else:
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t <= 0]
        win_rate = len(wins) / num_trades
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    metrics = {
        "sharpe": round(float(sharpe), 4),
        "total_return": round(float(total_return), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "win_rate": round(float(win_rate), 4),
        "num_trades": int(num_trades),
        "profit_factor": round(float(profit_factor), 4),
    }
    return metrics, equity_curve, dates


def run_backtest(strategy: dict) -> dict:
    """Run strategy against historical data. Returns metrics dict."""
    return _simulate(strategy)[0]


def get_equity_curve(strategy: dict) -> pd.Series:
    """Run strategy and return day-by-day portfolio value as a dated Series."""
    metrics, curve, dates = _simulate(strategy)
    return pd.Series(curve[1:len(dates) + 1], index=dates, name="equity")


if __name__ == "__main__":
    import time

    spy_buyhold = {
        "ticker": "SPY",
        "indicators": [{"type": "SMA", "period": 5, "name": "sma_5"}],
        "entry_conditions": [{"indicator": "sma_5", "operator": ">", "value": 0}],
        "exit_conditions": [],
        "stop_loss": 0.99,
        "take_profit": 10.0,
    }

    t0 = time.perf_counter()
    metrics = run_backtest(spy_buyhold)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print("SPY buy-and-hold sanity check:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  elapsed: {elapsed_ms:.1f}ms")
    print()
    print("Expected: total_return ~0.14 annualized, sharpe ~0.4-0.6, num_trades = 1")
    print()

    bad_strategy = {
        "ticker": "TSLA",
        "indicators": [{"type": "RSI", "period": 14, "name": "rsi_14"}],
        "entry_conditions": [{"indicator": "rsi_14", "operator": ">", "value": 80}],
        "exit_conditions": [{"indicator": "rsi_14", "operator": "<", "value": 50}],
        "stop_loss": 0.05,
        "take_profit": 0.15,
    }

    t0 = time.perf_counter()
    bad_metrics = run_backtest(bad_strategy)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print("TSLA RSI>80 entry (bad strategy) sanity check:")
    for k, v in bad_metrics.items():
        print(f"  {k}: {v}")
    print(f"  elapsed: {elapsed_ms:.1f}ms")
    print()
    print("Expected: very few trades, likely negative or near-zero return")
