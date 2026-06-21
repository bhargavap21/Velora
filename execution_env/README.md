# Velora — Optimal Execution Environment

RL environment for optimal trade execution: an agent must fill a large order (e.g. buy
10,000 shares) over a fixed time window without moving the price against itself. Scored
on slippage vs. VWAP, the same benchmark real execution desks are graded on.

This directory is the active hackathon project.

## Layout

```
execution_env/
  env.py                    HUD v6 environment definition (the actual submission entrypoint)
  tasks.py                  Task definitions (prompt + grading config + slug)
  server.py                 FastAPI API: /api/episode, /api/episode/stream (SSE), /api/config
  sandbox_config.py         Sandbox defaults, constraints, regime/date resolution
  Dockerfile.hud            Container image for hosted HUD runs
  data_cache/               Committed Alpaca SIP daily/minute parquet (offline replay, no keys)
  results/                  Trained PPO checkpoint (ppo_execution.zip)
  simulator/
    market_sim.py           Intraday price path + volume curve + impact model (real + synthetic)
    benchmark.py             VWAP/TWAP calculation + slippage reward
  rl/
    execution_gym_env.py     gymnasium.Env wrapper around the simulator
    train_ppo.py             SB3 PPO baseline training
  agents/
    llm_agent.py             Claude-driven execution-schedule agent
    fireworks_agent.py       Open-source-model agent via Fireworks (gpt-oss comparison arm)
  tests/                     Offline tests (no keys, no network): simulator, gym env,
                             agents, sandbox config, and the server/SSE API
```

## Status / TODO map

- [x] **Simulator core** (`simulator/market_sim.py`) — intraday price path + U-shaped
  volume curve + Almgren-Chriss impact model, backed by real Alpaca SIP daily/minute bars
  (yfinance + synthetic Brownian-bridge fallbacks). Covered by `tests/test_market_sim.py`.
- [x] **Benchmark + reward** (`simulator/benchmark.py`) — VWAP/TWAP calc, slippage reward,
  unfilled-inventory penalty.
- [x] **Gym env** (`rl/execution_gym_env.py`) — simulator + benchmark wired into a
  `gymnasium.Env`; passes `check_env`. Covered by `tests/test_execution_gym_env.py`.
- [x] **PPO training** (`rl/train_ppo.py`) — trained; checkpoint committed at
  `results/ppo_execution.zip` and served by the `ppo` policy.
- [x] **HUD env wrapper** (`env.py`, `tasks.py`, `Dockerfile.hud`) — verified against HUD
  v6.6; `hud serve execution_env/env.py:env` boots (3 tasks, 2 tools) and the
  read-context → submit-schedule → grade pipeline scores end-to-end.
- [x] **LLM agent** (`agents/llm_agent.py`) — Claude proposes a per-episode schedule.
- [x] **Fireworks agent** (`agents/fireworks_agent.py`) — gpt-oss open-source comparison arm.
- [x] **Live demo wiring** (`server.py`, frontend) — real per-slice SSE streaming via
  `GET /api/episode/stream`; Live + Sandbox pages render each slice as the backend
  produces it. Covered by `tests/test_server.py`.

### Remaining / nice-to-have

- [ ] **PPO is fixed to 26 slices** — retrain per-timeframe if the sandbox should support
  PPO at other slice counts.
- [ ] **PPO's edge is order-size dependent** — clear, statistically significant advantage
  (83-95% win-rate) at institutional sizes (~8% of ADV) where participation-driven impact
  dominates; at the small 10k-share size used in the HUD tasks, impact is too small for
  scheduling to matter and PPO is roughly a coin flip vs. VWAP-match (~47% win-rate). Reward
  shaping that makes the small-order regime matter (or training data weighted toward it)
  would close that gap.

## Benchmark results

Reward is normalized so 0.50 = the impact-free VWAP benchmark price.

**HUD agent eval — PPO** (`python -m execution_env.rl.hud_ppo_agent`, 2026-06-20; the
trained PPO policy rolled out through the HUD MCP env). 

| Task          | PPO   | Claude | vs VWAP (PPO) |
|---------------|-------|--------|---------------|
| buy-10k-aapl  | 0.529 | 0.537  | beat          |
| buy-10k-tsla  | 0.725 | 0.728  | beat          |
| sell-10k-spy  | 0.499 | 0.456  | parity        |
| **Mean**      | **0.584** | 0.573 | 2/3 beat VWAP |

PPO job: https://hud.ai/jobs/7aa479945c9f4c5798eb9b5f1caa26da
Claude job: https://hud.ai/jobs/79f7ed9e46b047b1ad709d5c03aee3e7

Note: these tasks are 10k-share orders, where impact is small and the result is dominated
by single-day timing on a fixed seed. The robust, statistically-powered evidence is the
held-out eval below (institutional ADV-sized orders, hundreds of paired days).

**PPO held-out eval** (recalibrated, participation-saturated impact model; orders sized at
8% of ADV; paired against the **VWAP-match** baseline on the identical held-out price path,
the last 20% of each ticker's history, unseen in training):

- Pooled over all 4 tickers x {buy, sell}, n=480: win-rate **83%**, mean **+25 bps**,
  median **+27 bps**, t **16.7**.
- AAPL buy example (one-click reproducible on the Proof page), n=60: win-rate **95%**,
  median **+34.5 bps**, t **10.24**, mean order notional ≈ $1.2B.
- PPO also beats the naive equal-time TWAP floor: pooled win-rate **89%**, +57 bps median.

The landing page reads these numbers from `frontend/src/data/benchmarks.js`.

## Endpoints (`server.py`, port 8010)

- `GET /api/config` — sandbox defaults, constraints, tickers/date ranges.
- `GET /api/policies` — available policies (twap always; ppo if checkpoint present; llm /
  fireworks if the matching API key is set).
- `GET /api/episode` — run one episode and return the full trace as JSON.
- `GET /api/episode/stream` — same episode streamed slice-by-slice over SSE
  (`meta` → `slice`×N → `done`, or a single `error` event on bad input). `tick_ms` paces it.

## Setup

```bash
pip install -r ../requirements.txt   # gymnasium + stable-baselines3 already present
```

Real-data replay works offline: the committed `data_cache/*.parquet` files are read before
any network call, so the demo runs with no API keys. Set `ALPACA_API_KEY` /
`ALPACA_SECRET_KEY` to refresh or extend the cache, `ANTHROPIC_API_KEY` for the Claude
policy, and `FIREWORKS_API_KEY` for the gpt-oss policy.
