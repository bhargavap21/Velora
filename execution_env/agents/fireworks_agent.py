"""
Open-source-model execution-schedule agent via Fireworks AI, for the comparison
leaderboard (naive TWAP vs. PPO vs. Claude vs. this).

TODO(fireworks_agent): unimplemented. Same schedule-proposal shape as ../llm_agent.py,
but calling a Fireworks-hosted open model (e.g. Llama) via their OpenAI-compatible API
instead of the Anthropic SDK. Factor out the shared "schedule JSON -> per-slice fraction
sequence -> replay through ExecutionEnv" logic into a common helper once both agents
exist, rather than duplicating it.
"""

from __future__ import annotations

raise NotImplementedError("fireworks_agent.py is a scaffolding stub -- see module docstring TODO")
