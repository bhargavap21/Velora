"""
Issue #15, Phase 1 sanity checks for the new randomized %-ADV task (env.py::execution_random):

  (C) Reward-shaping check -- does the normalized [0,1] reward spread across this task
      distribution avoid saturating at the floor/ceiling, now that order sizes carry real
      impact (unlike the original 10k-share tasks)?
  (D) Sanity eval -- base model (Claude, zero-shot) vs. VWAP-match vs. the Modal-retrained
      PPO checkpoint, on the same randomized scenarios. Records the gap an RFT'd model
      needs to close (per the issue's own Phase 1 deliverable: "base model vs VWAP-match
      vs PPO on the new task set").

Local replay (ExecutionEnv directly), not live `hud eval` -- faster, no HUD_API_KEY spend,
and gives the same bps/fill detail train_ppo.py's eval already reports.

Usage:
    .venv/bin/python -m execution_env.rl.sanity_eval_random_task
"""

from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from execution_env.agents.llm_agent import propose_schedule, twap as llm_twap
from execution_env.env import _normalize_reward, _sample_random_scenario
from execution_env.rl.execution_gym_env import ExecutionEnv
from execution_env.rl.train_ppo import MODEL_PATH
from execution_env.simulator.benchmark import compute_vwap, execution_vwap, slippage_reward

_N_SCENARIOS = 20


def _score_env(env: ExecutionEnv, side: str) -> dict:
    agent_vwap = execution_vwap(np.array(env._exec_prices), np.array(env._exec_quantities))
    benchmark_vwap = compute_vwap(env._path[1:], env._volume_curve * env._total_shares)
    filled_fraction = 1.0 - env._shares_remaining / env._total_shares
    raw_reward = slippage_reward(agent_vwap, benchmark_vwap, side, filled_fraction)
    sign = 1.0 if side == "buy" else -1.0
    slippage_bps = sign * (benchmark_vwap - agent_vwap) / benchmark_vwap * 10_000 if benchmark_vwap > 0 else 0.0
    return {"raw_reward": raw_reward, "normalized": _normalize_reward(raw_reward), "slippage_bps": slippage_bps}


def _run_schedule(scenario: dict, schedule: list[float]) -> dict:
    env = ExecutionEnv(
        n_slices=26, total_shares=scenario["total_shares"], side=scenario["side"], tickers=[scenario["ticker"]]
    )
    env.reset(seed=scenario["seed"])
    for i in range(env._n_slices):
        multiplier = schedule[i] if i < len(schedule) else 2.0
        _, _, terminated, _, _ = env.step(np.array([multiplier], dtype=np.float32))
        if terminated:
            break
    return _score_env(env, scenario["side"])


def _run_ppo(scenario: dict, model: PPO) -> dict:
    """PPO acts per-slice from its own observation (deterministic policy), not a
    pre-built schedule -- same seed as the other policies' runs so the price path
    (and therefore the comparison) is paired."""
    env = ExecutionEnv(
        n_slices=26, total_shares=scenario["total_shares"], side=scenario["side"], tickers=[scenario["ticker"]]
    )
    obs, _ = env.reset(seed=scenario["seed"])
    for _ in range(env._n_slices):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    return _score_env(env, scenario["side"])


def main() -> None:
    ppo_model = PPO.load(MODEL_PATH)
    if ppo_model.observation_space.shape[0] != ExecutionEnv().observation_space.shape[0]:
        raise RuntimeError(
            f"PPO checkpoint obs dim {ppo_model.observation_space.shape[0]} != "
            f"env obs dim {ExecutionEnv().observation_space.shape[0]} -- retrain or fix the mismatch."
        )

    vwap_match_normalized = []
    claude_normalized = []
    ppo_normalized = []
    claude_advantages_bps = []
    ppo_advantages_bps = []

    print(f"Sampling {_N_SCENARIOS} random (ticker, side, adv_pct, seed) scenarios...\n")
    for i in range(_N_SCENARIOS):
        scenario = _sample_random_scenario()
        n_slices = 26
        # Throwaway probe reset just to read the day's volume_curve/open_price -- the
        # actual scored runs below each get their own fresh env (same seed -> identical
        # price path, a paired comparison), since step() mutates env state.
        probe_info = ExecutionEnv(
            n_slices=n_slices, total_shares=scenario["total_shares"], side=scenario["side"], tickers=[scenario["ticker"]]
        ).reset(seed=scenario["seed"])[1]
        volume_curve = probe_info["volume_curve"]
        open_price = probe_info["open_price"]

        vwap_match = [1.0] * n_slices
        vwap_result = _run_schedule(scenario, vwap_match)
        vwap_match_normalized.append(vwap_result["normalized"])

        try:
            multipliers, parsed, _raw = propose_schedule(
                scenario["ticker"], scenario["total_shares"], n_slices, open_price, volume_curve
            )
        except Exception as exc:  # noqa: BLE001 -- fall back to TWAP on any API/parse error
            multipliers = llm_twap(n_slices, np.asarray(volume_curve, dtype=float))
            parsed = {"primitive": "twap (fallback)", "reasoning": str(exc)}
        claude_result = _run_schedule(scenario, multipliers)
        claude_normalized.append(claude_result["normalized"])

        ppo_result = _run_ppo(scenario, ppo_model)
        ppo_normalized.append(ppo_result["normalized"])

        claude_advantage = claude_result["slippage_bps"] - vwap_result["slippage_bps"]
        ppo_advantage = ppo_result["slippage_bps"] - vwap_result["slippage_bps"]
        claude_advantages_bps.append(claude_advantage)
        ppo_advantages_bps.append(ppo_advantage)

        print(
            f"[{i:2d}] {scenario['side']:4s} {scenario['ticker']:5s} "
            f"{scenario['total_shares']:>10,} sh | "
            f"VWAP-match: {vwap_result['normalized']:.3f} ({vwap_result['slippage_bps']:+.1f} bps) | "
            f"Claude ({parsed.get('primitive', '?')}): {claude_result['normalized']:.3f} "
            f"({claude_result['slippage_bps']:+.1f} bps, adv {claude_advantage:+.1f}) | "
            f"PPO: {ppo_result['normalized']:.3f} ({ppo_result['slippage_bps']:+.1f} bps, adv {ppo_advantage:+.1f})"
        )

    vwap_arr = np.array(vwap_match_normalized)
    claude_arr = np.array(claude_normalized)
    ppo_arr = np.array(ppo_normalized)
    claude_adv_arr = np.array(claude_advantages_bps)
    ppo_adv_arr = np.array(ppo_advantages_bps)

    print(f"\n{'=' * 70}")
    print("(C) Reward-shaping spread (VWAP-match normalized reward, should not saturate):")
    print(f"    mean={vwap_arr.mean():.3f} std={vwap_arr.std():.3f} min={vwap_arr.min():.3f} max={vwap_arr.max():.3f}")
    print(f"    at floor (<=0.01): {(vwap_arr <= 0.01).sum()}/{len(vwap_arr)}   "
          f"at ceil (>=0.99): {(vwap_arr >= 0.99).sum()}/{len(vwap_arr)}")

    print("\n(D) Base model (Claude, zero-shot) vs. VWAP-match vs. PPO, same randomized scenarios:")
    print(f"    Claude mean normalized: {claude_arr.mean():.3f}  |  VWAP-match: {vwap_arr.mean():.3f}  |  PPO: {ppo_arr.mean():.3f}")
    print(f"    Claude win-rate vs VWAP-match: {(claude_adv_arr > 0).mean():.2%}  "
          f"(mean adv {claude_adv_arr.mean():+.2f} bps, median {np.median(claude_adv_arr):+.2f} bps)")
    print(f"    PPO    win-rate vs VWAP-match: {(ppo_adv_arr > 0).mean():.2%}  "
          f"(mean adv {ppo_adv_arr.mean():+.2f} bps, median {np.median(ppo_adv_arr):+.2f} bps)")
    print(
        "    -> the Claude-vs-VWAP-match gap is what an RFT'd model needs to beat (primary success\n"
        "       metric, per issue #15); the PPO-vs-VWAP-match gap is the closed-loop ceiling RFT is\n"
        "       not expected to reach (per the issue's honest-framing note)."
    )


if __name__ == "__main__":
    main()
