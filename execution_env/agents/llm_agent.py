"""
Claude-driven execution-schedule agent.

Unlike the PPO agent (which decides a fraction-of-remaining-inventory per slice), the
LLM agent sets a coarser per-episode execution *schedule* (e.g. "front-load 60% in the
first third," "match the volume curve," "hold back if early slices show adverse price
movement") and that schedule is then played out slice-by-slice against the same
ExecutionEnv. This keeps the LLM call budget to ~1 call per episode instead of one per
slice.

TODO(llm_agent): this is unimplemented. Reuse the observation/decision/feedback text-
building pattern from ../../episode_core.py (build_initial_observation, process_turn,
run_episode_events) as the template -- same shape, different action vocabulary:
  - schedule primitives: front_load(pct), back_load(pct), follow_volume_curve(),
    pause_if_adverse_move(threshold_bps)
  - one Claude call proposes a schedule (JSON) at the start of the episode
  - the schedule is converted into a fixed sequence of per-slice fractions and replayed
    through ExecutionEnv.step() to get the final slippage reward
"""

from __future__ import annotations

raise NotImplementedError("llm_agent.py is a scaffolding stub -- see module docstring TODO")
