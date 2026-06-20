"""
Claude-driven execution-schedule agent.

Unlike the PPO agent (which decides a fraction-of-remaining-inventory per slice), the
LLM agent sets a coarser per-episode execution *schedule* (e.g. "front-load 60% in the
first third," "match the volume curve," "hold back if early slices show adverse price
movement") and that schedule is then played out slice-by-slice against the same
ExecutionEnv. This keeps the LLM call budget to ~1 call per episode instead of one per
slice.
"""

from __future__ import annotations

import json
import re

import anthropic
import numpy as np

from execution_env.simulator.market_sim import u_shaped_volume_curve

_MODEL = "claude-opus-4-8"

# ---------------------------------------------------------------------------
# Step 1: Schedule primitive helpers
# ---------------------------------------------------------------------------
# ExecutionEnv.step() takes the *fraction of remaining inventory* to execute
# this slice — not an absolute quantity. All primitives below work in absolute
# "fraction of total" space and then call _qty_to_fracs() to convert, so the
# math inside each primitive stays intuitive.


def _qty_to_fracs(target_shares: np.ndarray) -> list[float]:
    """Convert a per-slice absolute-quantity distribution to fractions-of-remaining.

    The last slice is always forced to 1.0 so the order is fully filled even if
    the earlier fractions don't sum to exactly 1.0 due to floating-point drift.
    """
    total = float(target_shares.sum())
    if total <= 0:
        n = len(target_shares)
        return [0.0] * (n - 1) + [1.0]

    fracs: list[float] = []
    remaining = total
    n = len(target_shares)
    for i, qty in enumerate(target_shares):
        if i == n - 1:
            fracs.append(1.0)
        elif remaining <= 1e-9:
            fracs.append(0.0)
        else:
            fracs.append(float(np.clip(qty / remaining, 0.0, 1.0)))
            remaining = max(remaining - qty, 0.0)
    return fracs


def front_load(pct: float, n_slices: int) -> list[float]:
    """Execute pct of total inventory in the first third of slices, rest evenly.

    Example: front_load(0.6, 26) puts 60% in slices 0-8, 40% in slices 9-25.
    """
    pct = float(np.clip(pct, 0.0, 1.0))
    early = max(1, n_slices // 3)
    late = n_slices - early
    weights = np.array([pct / early] * early + [(1.0 - pct) / max(late, 1)] * late)
    return _qty_to_fracs(weights)


def back_load(pct: float, n_slices: int) -> list[float]:
    """Execute pct of total inventory in the final third of slices, rest evenly.

    Example: back_load(0.6, 26) puts 40% in slices 0-17, 60% in slices 18-25.
    """
    pct = float(np.clip(pct, 0.0, 1.0))
    late = max(1, n_slices // 3)
    early = n_slices - late
    weights = np.array([(1.0 - pct) / max(early, 1)] * early + [pct / late] * late)
    return _qty_to_fracs(weights)


def follow_volume_curve(n_slices: int) -> list[float]:
    """Weight each slice proportionally to the U-shaped intraday volume curve.

    Heavier at open and close (when market liquidity is highest), lighter midday.
    This minimises market impact by trading when natural volume absorbs the order.
    """
    weights = u_shaped_volume_curve(n_slices)
    return _qty_to_fracs(weights)


def twap(n_slices: int) -> list[float]:
    """Equal absolute quantity each slice — the naive TWAP baseline."""
    weights = np.ones(n_slices) / n_slices
    return _qty_to_fracs(weights)


# ---------------------------------------------------------------------------
# Step 3: Build the Claude prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an institutional execution trader. Your goal is to buy a large block of "
    "shares at the lowest possible average price — minimising slippage versus VWAP. "
    "Output exactly one JSON object with no prose before or after it."
)


def build_observation(
    ticker: str,
    total_shares: int,
    n_slices: int,
    open_price: float,
    volume_curve: list[float],
) -> str:
    """Build the text prompt sent to Claude at the start of an episode.

    Describes the trading task, the available schedule primitives, and the current
    market context. Claude returns a single JSON choosing one primitive (and optional
    pause modifier) — we expand it to per-slice fractions in schedule_from_response().
    """
    vc = [round(v, 4) for v in volume_curve]
    midday_idx = n_slices // 2
    return f"""You must execute a buy order for {total_shares:,} shares of {ticker} over {n_slices} time slices today.
Each slice represents one equal-length interval of the trading day (e.g. 15-minute windows for a 6.5h session).

Your goal is to minimise VWAP slippage (buying as cheaply as possible vs. the market VWAP).
Large orders in a single slice move the price against you; spreading too evenly misses good windows.

Current market context:
  Ticker:      {ticker}
  Open price:  ${open_price:.2f}
  Order size:  {total_shares:,} shares over {n_slices} slices
  Volume curve (per-slice weight, sums to 1.0):
    Open:    {vc[0]:.4f}  (slice 0)
    Midday:  {vc[midday_idx]:.4f}  (slice {midday_idx})
    Close:   {vc[-1]:.4f}  (slice {n_slices - 1})
  Interpretation: higher weight = more natural market volume = less impact per share traded.

Available schedule primitives:
  "front_load"           — concentrate pct of order in first 1/3 of slices, rest spread evenly
                           (good if you expect price to rise intraday)
  "back_load"            — concentrate pct of order in final 1/3 of slices, rest spread evenly
                           (good if you expect price to fall intraday)
  "follow_volume_curve"  — weight each slice by the natural volume curve above
                           (minimises market impact by trading when liquidity is highest)
  "twap"                 — equal absolute quantity every slice (neutral baseline)

Optional modifier:
  "pause_if_adverse_move" — skip a slice entirely if the price has moved against you by more
                            than threshold_bps basis points since the last executed slice.
                            Use this to avoid chasing a rapidly rising price.

Output ONLY this JSON (no other text):
{{
  "primitive": "front_load" | "back_load" | "follow_volume_curve" | "twap",
  "pct": <float 0-1, required for front_load / back_load, omit otherwise>,
  "pause_if_adverse_move": {{"enabled": true, "threshold_bps": <int>}} | {{"enabled": false}},
  "reasoning": "<one sentence>"
}}"""


# ---------------------------------------------------------------------------
# Step 3b: Parse Claude's JSON response
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Strip markdown fences and extract the first JSON object from text.

    Mirrors the same helper in episode_core.py so both agents behave consistently
    when Claude adds unwanted prose or code fences.
    """
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON object found in agent response", text, 0)


def schedule_from_response(response_text: str, n_slices: int) -> tuple[list[float], dict]:
    """Parse Claude's JSON response into a per-slice fraction list.

    Returns (fracs, parsed_json). Raises ValueError on invalid primitive names or
    missing required fields so the caller can surface a useful error.
    """
    parsed = _extract_json(response_text)
    primitive = parsed.get("primitive", "")
    pct = float(parsed.get("pct", 0.5))

    if primitive == "front_load":
        fracs = front_load(pct, n_slices)
    elif primitive == "back_load":
        fracs = back_load(pct, n_slices)
    elif primitive == "follow_volume_curve":
        fracs = follow_volume_curve(n_slices)
    elif primitive == "twap":
        fracs = twap(n_slices)
    else:
        raise ValueError(
            f"Unknown primitive '{primitive}'. Expected one of: "
            "front_load, back_load, follow_volume_curve, twap."
        )

    return fracs, parsed


# ---------------------------------------------------------------------------
# Step 4: Claude API call — propose a schedule for the episode
# ---------------------------------------------------------------------------

def propose_schedule(
    ticker: str,
    total_shares: int,
    n_slices: int,
    open_price: float,
    volume_curve: list[float],
) -> tuple[list[float], dict, str]:
    """Call Claude once at episode start to choose a schedule primitive.

    Returns (fracs, parsed_json, raw_response_text). The raw text is kept for
    logging/debugging; callers should use fracs for execution.

    The model sees only the market context — no historical price path, since that
    doesn't exist until the episode plays out. Claude picks a named primitive
    (e.g. "front_load") plus optional parameters; we expand it deterministically.
    """
    client = anthropic.Anthropic()
    user_message = build_observation(ticker, total_shares, n_slices, open_price, volume_curve)
    response = client.messages.create(
        model=_MODEL,
        max_tokens=512,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text
    fracs, parsed = schedule_from_response(raw, n_slices)
    return fracs, parsed, raw


# ---------------------------------------------------------------------------
# Step 5: Run one episode end-to-end
# ---------------------------------------------------------------------------

def run_episode(env, *, verbose: bool = False) -> dict:
    """Reset env, call Claude once for a schedule, replay it slice-by-slice.

    Applies `pause_if_adverse_move` at runtime: if the price has risen more than
    `threshold_bps` basis points since the last executed slice, that slice is
    skipped (action = 0.0). The final slice's frac is always 1.0 (from
    _qty_to_fracs), so any paused inventory is executed at close.

    Args:
        env: An ExecutionEnv instance (gymnasium-compatible). Must NOT be pre-reset;
             this function calls reset() itself.
        verbose: If True, prints Claude's chosen primitive and reasoning.

    Returns:
        dict with keys: reward, ticker, open_price, primitive, pct,
        pause_enabled, reasoning, raw_response.
    """
    obs, info = env.reset()
    ticker: str = info["ticker"]
    open_price: float = info["open_price"]
    volume_curve: list[float] = info["volume_curve"]
    n_slices: int = info["n_slices"]
    total_shares: int = info["total_shares"]

    try:
        fracs, parsed, raw_response = propose_schedule(
            ticker, total_shares, n_slices, open_price, volume_curve
        )
    except (json.JSONDecodeError, ValueError, anthropic.APIError) as exc:
        fracs = twap(n_slices)
        parsed = {"primitive": "twap", "reasoning": f"fallback: {exc}"}
        raw_response = ""

    if verbose:
        print(
            f"[{ticker}] primitive={parsed.get('primitive')}  "
            f"pct={parsed.get('pct', '—')}"
        )
        print(f"  reasoning: {parsed.get('reasoning', '')}")

    pause_cfg: dict = parsed.get("pause_if_adverse_move", {"enabled": False})
    pause_enabled: bool = bool(pause_cfg.get("enabled", False))
    pause_threshold_bps: float = float(pause_cfg.get("threshold_bps", 0))

    total_reward = 0.0
    current_price: float = open_price  # tracks the base market price each slice
    last_exec_price: float | None = None  # last slice where frac > 0

    for i in range(n_slices):
        # obs[2] = (path[i] - path[i-1]) / path[i-1]; update only after first step.
        # At i=0, obs is from reset() and obs[2] = 0.0, so price stays at open_price.
        if i > 0:
            current_price *= 1.0 + float(obs[2])

        frac = fracs[i]

        # Pause modifier: skip slice if price has risen adversely since last fill.
        if pause_enabled and last_exec_price is not None and pause_threshold_bps > 0:
            price_move_bps = (current_price - last_exec_price) / last_exec_price * 10_000
            if price_move_bps > pause_threshold_bps:
                frac = 0.0

        obs, reward, terminated, truncated, _ = env.step(
            np.array([frac], dtype=np.float32)
        )
        total_reward += reward

        if frac > 0:
            last_exec_price = current_price

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "ticker": ticker,
        "open_price": open_price,
        "primitive": parsed.get("primitive"),
        "pct": parsed.get("pct"),
        "pause_enabled": pause_enabled,
        "reasoning": parsed.get("reasoning", ""),
        "raw_response": raw_response,
    }
