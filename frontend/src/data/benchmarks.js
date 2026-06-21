// Recorded benchmark results — single source of truth for the landing page.
//
// These are real, reproducible measurements, not estimates:
//   - hudEval: the HUD agent evaluation (`hud eval execution_env/tasks.py claude`),
//     run on 2026-06-20 with claude-sonnet-4-6 through the HUD gateway. Reflects the
//     pre-retrain (7-dim observation) PPO checkpoint -- not yet rerun against the new
//     8-dim/51-ticker checkpoint below.
//   - ppoPooled / ppoHoldout: the server's held-out evaluation over chronologically
//     held-out days (the last 20% of each ticker's history) the PPO model never trained
//     on, paired by seed so each policy sees the identical price path. Measured against
//     the recalibrated, participation-saturated impact model (market_sim.py:_MAX_PARTICIPATION).
//     Re-measured 2026-06-21 against the PPO checkpoint retrained on Modal (8-dim obs incl.
//     volume_surprise, 2.03M timesteps, 51-ticker training universe) -- verified by directly
//     replaying the eval loop locally against execution_env/results/ppo_execution.zip
//     (not just trusting the Modal training log).
//       ppoPooled  = pooled over the original 4 tickers x {buy, sell}, 60 days each (n=480).
//       ppoHoldout = the single-ticker example reproducible one-click on the Proof page
//                    (policy=ppo, baseline=twap/VWAP-match, ticker=AAPL, side=buy,
//                     adv_pct=8, n_episodes=60).
//     Both compare PPO against the *VWAP-matching* baseline — the strong baseline, not the
//     naive one — so the reported edge is conservative.
//
//     CAVEAT: this edge is order-size dependent. It holds at institutional sizes (adv_pct=8,
//     i.e. orders sized as ~8% of average daily volume) where participation-driven impact
//     dominates and intraday scheduling matters. At the small order size used in the HUD
//     tasks above (10,000 shares), PPO is roughly a coin flip against VWAP-match -- impact is
//     too small at that size for scheduling skill to show up. Don't present the pooled/holdout
//     win-rates as if they applied to a 10k-share order; always pair them with the adv_pct
//     they were measured at.
//
//     The new checkpoint also generalizes to ~51 tickers beyond the original 4 (see
//     execution_env/rl/train_ppo.py:TRAIN_TICKERS) -- those win-rates aren't re-measured here,
//     since ppoPooled/ppoHoldout intentionally stay pinned to the original 4-ticker set this
//     page has always reported.
//
// Reward is normalized so 0.50 == the VWAP/TWAP benchmark; > 0.50 beats it.

// Two HUD traces on the same 3 tasks, normalized so 0.50 == the impact-free VWAP price.
// Claude calls the tools directly (read_market_context -> submit_schedule). PPO has no
// native HUD/MCP interface (it's a sequential gym policy, not a tool-calling agent), so
// its run uses execution_env/rl/hud_ppo_agent.py: roll the trained policy out locally
// against a seed-matched copy of the env to get its full schedule, then submit that
// schedule through the real HUD MCP tools against the actual hosted env -- a genuine
// hud.ai job, not a fabricated one, because the grading episode is deterministic on seed.
export const hudEval = {
  runtime: 'HUD MCP env · local rollout',
  date: '2026-06-20',
  baselineScore: 0.5, // normalized reward equivalent to matching VWAP
  tasks: [
    { id: 'buy-10k-aapl', label: 'Buy 10k AAPL' },
    { id: 'buy-10k-tsla', label: 'Buy 10k TSLA' },
    { id: 'sell-10k-spy', label: 'Sell 10k SPY' },
  ],
  agents: [
    {
      id: 'ppo',
      label: 'PPO (RL agent)',
      color: '#acafb9', // pearl
      jobUrl: 'https://hud.ai/jobs/7aa479945c9f4c5798eb9b5f1caa26da',
      meanScore: 0.584,
      tasksBeat: 2,
      scores: { 'buy-10k-aapl': 0.529, 'buy-10k-tsla': 0.725, 'sell-10k-spy': 0.499 },
    },
    {
      id: 'claude',
      label: 'Claude',
      color: '#cc9166', // ember gold
      jobUrl: 'https://hud.ai/jobs/79f7ed9e46b047b1ad709d5c03aee3e7',
      meanScore: 0.573,
      tasksBeat: 2,
      scores: { 'buy-10k-aapl': 0.537, 'buy-10k-tsla': 0.728, 'sell-10k-spy': 0.456 },
    },
  ],
}

// Pooled held-out result across all 4 tickers and both sides (n=480) vs VWAP-match.
// Reproduce per-ticker on the Proof page (base_seed=10000) and pool the cells.
export const ppoPooled = {
  baseline: 'VWAP-match',
  nEpisodes: 480,
  winRate: 0.825,
  meanAdvantageBps: 30.89,
  medianAdvantageBps: 33.36,
  tStat: 18.51,
  medianUsdSavedPerOrder: 5_408_990,
  meanNotionalUsd: 2_270_055_855,
}

// Single-ticker example, reproducible one-click on the Proof page (its default scenario).
export const ppoHoldout = {
  baseline: 'VWAP-match',
  ticker: 'AAPL',
  side: 'buy',
  advPct: 8,
  nEpisodes: 60,
  winRate: 0.95,
  meanAdvantageBps: 36.24,
  medianAdvantageBps: 37.2,
  tStat: 10.4,
  usdSavedPerOrder: 4_245_398,
  meanNotionalUsd: 1_171_370_488,
}

// RFT (issue #15, Phase 4) held-out result -- a forked, trainable Qwen3 8B (Tinker) GRPO-
// trained for 15 groups (group_size=8) via HUD's TrainingClient, best checkpoint (step-000007,
// in-training mean_reward=0.241) promoted as the active head via set_head(). Compared against
// the pristine un-forked base model (Qwen/Qwen3-8B) on n=12 fresh held-out scenarios (distinct
// seed pool from training), paired by seed so both models see the identical market path.
//
// Measured 2026-06-21. Deliberately small/honest: this is NOT a "RFT beats base model" claim --
// the delta is small and positive but the t-stat is far from significant at this n (contrast
// ppoPooled's t=18.5 at n=480). What's real here: the full collect -> train -> promote-checkpoint
// -> held-out-eval pipeline now runs end-to-end (a real blocking bug -- a missing
// `return_token_ids` flag -- was found and fixed to get here). Reaching a defensible "beats
// zero-shot" claim needs more training steps and a larger held-out n; this is the in-progress
// number, not the final one. See GitHub issue #15 and the /pitch deck (Slide 10) for the
// reward curve and full writeup.
export const rftHoldout = {
  baseline: 'zero-shot Qwen3 8B',
  model: 'velora-execution-rft-qwen3-8b (step-000007)',
  nEpisodes: 12,
  baseMeanReward: 0.156,
  rftMeanReward: 0.179,
  meanDelta: 0.023,
  wins: 4,
  ties: 5,
  losses: 3,
  tStat: 0.28,
  isSignificant: false,
  trainingGroups: 15,
  trainingStepsApplied: 14,
}

// Headline stat band shown on the landing page.
export const headlineStats = [
  { label: 'HUD tasks beating VWAP', value: '2 / 3', sub: 'PPO + Claude agents' },
  { label: 'PPO win-rate, institutional size', value: '83%', sub: 'vs VWAP-match · 8% ADV · 480 paired days' },
  { label: 'Tickers, real SIP data', value: '4', sub: 'AAPL · NVDA · TSLA · SPY' },
  { label: 'Execution policies', value: '5', sub: 'TWAP · VWAP · PPO · LLMs' },
]
