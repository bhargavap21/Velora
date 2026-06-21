"""
Issue #15, Phase 1 sanity checks for the new randomized %-ADV task (env.py::execution_random):

  (C) Reward-shaping check -- does the normalized [0,1] reward spread across this task
      distribution avoid saturating at the floor/ceiling, now that order sizes carry real
      impact (unlike the original 10k-share tasks)?
  (D) Sanity eval -- base model (Claude, zero-shot) vs. the VWAP-match baseline, on the
      same randomized scenarios. Records the gap an RFT'd model needs to close.

Local replay (ExecutionEnv directly), not live `hud eval` -- faster, no HUD_API_KEY spend,
and gives the same bps/fill detail train_ppo.py's eval already reports.

Usage:
    .venv/bin/python -m execution_env.rl.sanity_eval_random_task
"""

from __future__ import annotations

import numpy as np

from execution_env.agents.llm_agent import propose_schedule, twap as llm_twap
from execution_env.env import _normalize_reward, _sample_random_scenario
from execution_env.rl.execution_gym_env import ExecutionEnv
from execution_env.rl.train_ppo import twap_action
from execution_env.simulator.benchmark import compute_vwap, execution_vwap, slippage_reward

_N_SCENARIOS = 20


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

    agent_vwap = execution_vwap(np.array(env._exec_prices), np.array(env._exec_quantities))
    benchmark_vwap = compute_vwap(env._path[1:], env._volume_curve * env._total_shares)
    filled_fraction = 1.0 - env._shares_remaining / env._total_shares
    raw_reward = slippage_reward(agent_vwap, benchmark_vwap, scenario["side"], filled_fraction)
    sign = 1.0 if scenario["side"] == "buy" else -1.0
    slippage_bps = sign * (benchmark_vwap - agent_vwap) / benchmark_vwap * 10_000 if benchmark_vwap > 0 else 0.0
    return {"raw_reward": raw_reward, "normalized": _normalize_reward(raw_reward), "slippage_bps": slippage_bps}


def main() -> None:
    vwap_match_normalized = []
    claude_normalized = []
    advantages_bps = []

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

        advantage = claude_result["slippage_bps"] - vwap_result["slippage_bps"]
        advantages_bps.append(advantage)

        print(
            f"[{i:2d}] {scenario['side']:4s} {scenario['ticker']:5s} "
            f"{scenario['total_shares']:>10,} sh | "
            f"VWAP-match: {vwap_result['normalized']:.3f} ({vwap_result['slippage_bps']:+.1f} bps) | "
            f"Claude ({parsed.get('primitive', '?')}): {claude_result['normalized']:.3f} "
            f"({claude_result['slippage_bps']:+.1f} bps) | adv {advantage:+.1f} bps"
        )

    vwap_arr = np.array(vwap_match_normalized)
    claude_arr = np.array(claude_normalized)
    adv_arr = np.array(advantages_bps)

    print(f"\n{'=' * 70}")
    print("(C) Reward-shaping spread (VWAP-match normalized reward, should not saturate):")
    print(f"    mean={vwap_arr.mean():.3f} std={vwap_arr.std():.3f} min={vwap_arr.min():.3f} max={vwap_arr.max():.3f}")
    print(f"    at floor (<=0.01): {(vwap_arr <= 0.01).sum()}/{len(vwap_arr)}   "
          f"at ceil (>=0.99): {(vwap_arr >= 0.99).sum()}/{len(vwap_arr)}")

    print("\n(D) Claude (zero-shot, base model) vs. VWAP-match baseline:")
    print(f"    Claude mean normalized: {claude_arr.mean():.3f}  vs VWAP-match: {vwap_arr.mean():.3f}")
    print(f"    Claude win-rate vs VWAP-match: {(adv_arr > 0).mean():.2%}")
    print(f"    Mean advantage: {adv_arr.mean():+.2f} bps  median: {np.median(adv_arr):+.2f} bps")
    print("    -> this is the gap an RFT'd model needs to beat (per issue #15 Phase 1 deliverable).")


if __name__ == "__main__":
    main()
