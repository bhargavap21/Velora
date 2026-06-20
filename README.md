# Velora

An RL environment for **optimal trade execution**: an agent must fill a large stock
order (e.g. buy 10,000 shares) over a fixed time window without moving the price
against itself. Scored on slippage vs. **VWAP** — the same benchmark real execution
desks are measured against — not an arbitrary score.

Built for the [HUD x YC Hackathon](https://www.aivalley.io/hackathons/hud-frontier-rsi-rl-environments-hackathon)
(June 20–21, 2026).

## Why this problem

Naive execution costs money: dump a large order at once and you tank the price against
yourself; spread it out and you eat the day's natural drift instead. Optimal execution
(Almgren-Chriss, Nevmyvaka et al.) is a well-studied, high-value problem in quant
finance — the dollar amount saved vs. a naive baseline is a real, verifiable number, not
a synthetic score invented for this hackathon.

The environment supports both a classical RL policy (PPO) and an LLM agent operating
over the same simulator, so we can compare how each approach learns to beat the
benchmark.

## Active project: `execution_env/`

```
execution_env/
  env.py                    HUD environment definition (submission entrypoint)
  tasks.py                  Task definitions
  Dockerfile.hud            Container image for hosted HUD runs
  simulator/
    market_sim.py           Synthetic intraday price path + volume curve + impact model
    benchmark.py            VWAP/TWAP calculation + slippage reward
  rl/
    execution_gym_env.py    gymnasium.Env wrapper around the simulator
    train_ppo.py            SB3 PPO baseline training
  agents/
    llm_agent.py            Claude-driven execution-schedule agent
    fireworks_agent.py      Open-source-model agent via Fireworks (comparison leaderboard)
  tests/
    test_market_sim.py      Offline unit tests (no keys, no network)
```

See [`execution_env/README.md`](execution_env/README.md) for the full TODO map and
per-phase testing benchmarks.

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # ANTHROPIC_API_KEY, EXA_API_KEY, FIREWORKS_API_KEY as needed
```

### Run

```bash
# unit tests
python -m pytest execution_env/tests/ -q

# sanity-check the Gym env (random policy, one episode)
python -m execution_env.rl.execution_gym_env

# train the PPO baseline
python -m execution_env.rl.train_ppo

# naive TWAP scenario through the HUD-facing entrypoint
python -m execution_env.env
```

## Track positioning

Submitting under **Autonomous Business** — "turn real-world demand into verified
business value" maps directly onto this problem: a large order is real demand, and
minimized slippage vs. VWAP is a literal, quantifiable dollar amount saved.

## Team

- [bhargavap21](https://github.com/bhargavap21)
- [draj222](https://github.com/draj222)
