# Velora — Optimal Execution Environment

RL environment for optimal trade execution: an agent must fill a large order (e.g. buy
10,000 shares) over a fixed time window without moving the price against itself. Scored
on slippage vs. VWAP, the same benchmark real execution desks are graded on.

This directory is the active hackathon project. The rest of the repo (`backtester.py`,
`episode_core.py`, `gym_env.py`, etc. at the project root) is the prior StratRL project —
left in place as a working fallback, not part of this submission.

## Layout

```
execution_env/
  env.py                    HUD v6 environment definition (the actual submission entrypoint)
  tasks.py                  Task definitions (prompt + grading config + slug)
  Dockerfile.hud            Container image for hosted HUD runs
  simulator/
    market_sim.py           Synthetic intraday price path + volume curve + impact model
    benchmark.py             VWAP/TWAP calculation + slippage reward
  rl/
    execution_gym_env.py     gymnasium.Env wrapper around the simulator
    train_ppo.py             SB3 PPO baseline training (mirrors ../train_baseline.py)
  agents/
    llm_agent.py             Claude-driven execution-schedule agent
    fireworks_agent.py       Open-source-model agent via Fireworks (comparison leaderboard)
  tests/
    test_market_sim.py       Offline tests for the simulator (no keys, no network)
```

## Status / TODO map (for issue-splitting)

Each TODO below is sized to be one GitHub issue. Marked with the file it lives in.

- [ ] **Simulator core** (`simulator/market_sim.py`) — synthetic price path generator,
  U-shaped volume curve, Almgren-Chriss impact model. Highest priority, everything depends
  on this.
- [ ] **Benchmark + reward** (`simulator/benchmark.py`) — VWAP/TWAP calc, slippage reward,
  unfilled-inventory penalty.
- [ ] **Gym env** (`rl/execution_gym_env.py`) — wire simulator + benchmark into a
  `gymnasium.Env`, validate `check_env`.
- [ ] **PPO training** (`rl/train_ppo.py`) — train, confirm reward curve beats naive TWAP
  baseline. This is the validation checkpoint — do not move on until this passes.
- [ ] **HUD env wrapper** (`env.py`, `tasks.py`, `Dockerfile.hud`) — confirm exact HUD v6
  decorator API against a real cloned template (`hud-evals/*-template`) before relying on
  the skeleton here; it's written from README descriptions, not verified source.
- [ ] **LLM agent** (`agents/llm_agent.py`) — Claude sets a per-episode execution schedule,
  reusing the observation/decision/feedback loop shape from `../episode_core.py`.
- [ ] **Fireworks agent** (`agents/fireworks_agent.py`) — open-source model comparison arm.
- [ ] **Live demo wiring** (`server/`, frontend) — adapt the existing SSE pattern in
  `../server/main.py` to stream per-slice execution decisions.

## Setup

```bash
pip install -r ../requirements.txt   # gymnasium + stable-baselines3 already present
```
