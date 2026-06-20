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

- [ ] **Full HUD agent eval** — `hud eval execution_env/tasks.py claude ...` against a live
  model loop (needs HUD cloud creds). Only local serve + scoring is verified so far.
- [ ] **PPO is fixed to 26 slices** — retrain per-timeframe if the sandbox should support
  PPO at other slice counts.

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
