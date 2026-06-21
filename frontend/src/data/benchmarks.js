// Recorded benchmark results — single source of truth for the landing page.
//
// These are real, reproducible measurements, not estimates:
//   - hudEval: the HUD agent evaluation (`hud eval execution_env/tasks.py claude`),
//     run on 2026-06-20 with claude-sonnet-4-6 through the HUD gateway.
//   - ppoHoldout: the server's /api/eval/stream over chronologically held-out days
//     the PPO model never trained on, paired against the VWAP-matching baseline.
//     Refreshed 2026-06-20 against the recalibrated per-slice impact model with
//     params: policy=ppo, baseline=twap (VWAP-match), ticker=AAPL, side=buy,
//     adv_pct=8, n_episodes=60.
//
// Reward is normalized so 0.50 == the VWAP/TWAP benchmark; > 0.50 beats it.

export const hudEval = {
  model: 'claude-sonnet-4-6',
  runtime: 'HUD gateway (local rollout)',
  date: '2026-06-20',
  jobUrl: 'https://hud.ai/jobs/3e476dd34f514ec19d642fee652c5498',
  baselineScore: 0.5, // normalized reward equivalent to matching VWAP
  meanScore: 0.573,
  stdScore: 0.114,
  tasksBeatBenchmark: 2,
  tasksTotal: 3,
  tasks: [
    { id: 'buy-10k-aapl', label: 'Buy 10k AAPL', score: 0.536, beat: true },
    { id: 'buy-10k-tsla', label: 'Buy 10k TSLA', score: 0.727, beat: true },
    { id: 'sell-10k-spy', label: 'Sell 10k SPY', score: 0.455, beat: false },
  ],
}

export const ppoHoldout = {
  baseline: 'VWAP-match',
  ticker: 'AAPL',
  advPct: 8,
  nEpisodes: 60,
  winRate: 0.633,
  meanAdvantageBps: 5.28,
  medianAdvantageBps: 5.42,
  tStat: 3.76,
  usdSavedPerOrder: 618_972,
  meanNotionalUsd: 1_171_370_488,
}

// Headline stat band shown on the landing page.
export const headlineStats = [
  { label: 'HUD tasks beating VWAP', value: '2 / 3', sub: 'Claude execution agent' },
  { label: 'PPO win-rate, unseen days', value: '63%', sub: 'vs VWAP-match baseline' },
  { label: 'Tickers, real SIP data', value: '4', sub: 'AAPL · NVDA · TSLA · SPY' },
  { label: 'Execution policies', value: '5', sub: 'TWAP · VWAP · PPO · LLMs' },
]
