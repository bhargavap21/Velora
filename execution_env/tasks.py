"""Task definitions for the Velora optimal-execution HUD environment.

    hud eval execution_env/tasks.py claude --task-ids buy-10k-aapl -y --max-steps 8
    hud eval execution_env/tasks.py claude --task-ids buy-10k-tsla  -y --max-steps 8
    hud eval execution_env/tasks.py claude --task-ids sell-10k-spy  -y --max-steps 8
    hud eval execution_env/tasks.py claude --task-ids execution-random -y --max-steps 8
"""

from execution_env.env import (  # noqa: F401  (re-export env for `hud eval tasks.py`)
    buy_10k_aapl,
    buy_10k_tsla,
    env,
    execution_random,
    sell_10k_spy,
)

_PROMPT = (
    "You are an institutional execution trader. Minimize VWAP slippage on the order "
    "described in the task. Use read_market_context() to inspect the market, then "
    "submit_schedule() with your full {n_slices}-slice participation plan before the "
    "session ends. Trade more when volume is high and less when price moves against you."
)

_buy_aapl = buy_10k_aapl(prompt=_PROMPT.format(n_slices=26))
_buy_aapl.slug = "buy-10k-aapl"

_buy_tsla = buy_10k_tsla(prompt=_PROMPT.format(n_slices=26))
_buy_tsla.slug = "buy-10k-tsla"

_sell_spy = sell_10k_spy(prompt=_PROMPT.format(n_slices=26))
_sell_spy.slug = "sell-10k-spy"

# Randomized institutional-size scenario (ticker/side/order-size/seed sampled fresh each
# run) -- the trainable task the 3 fixed 10k-share tasks above can't provide. See
# execution_env/env.py::execution_random and issue #15, "Phase 1: make the env trainable."
_execution_random = execution_random(prompt=_PROMPT.format(n_slices=26))
_execution_random.slug = "execution-random"

tasks = [_buy_aapl, _buy_tsla, _sell_spy, _execution_random]
