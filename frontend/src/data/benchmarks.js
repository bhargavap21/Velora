// Recorded benchmark results — single source of truth for the landing page.
//
// These are real, reproducible measurements, not estimates:
//   - hudEval: the HUD agent evaluation (`hud eval execution_env/tasks.py claude`),
//     run on 2026-06-20 with claude-sonnet-4-6 through the HUD gateway.
//   - ppoHoldout: the server's /api/eval/stream over chronologically held-out days
//     the PPO model never trained on, paired against the VWAP-matching baseline.
//
// Reward is normalized so 0.50 == the VWAP/TWAP benchmark; > 0.50 beats it.

export const hudEval = {
  model: 'claude-sonnet-4-6',
  runtime: 'HUD gateway (local rollout)',
  date: '2026-06-20',
  jobUrl: 'https://hud.ai/jobs/79f7ed9e46b047b1ad709d5c03aee3e7',
  baselineScore: 0.5, // normalized reward equivalent to matching VWAP
  meanScore: 0.573,
  stdScore: 0.114,
  tasksBeatBenchmark: 2,
  tasksTotal: 3,
  tasks: [
    { id: 'buy-10k-aapl', label: 'Buy 10k AAPL', score: 0.537, beat: true },
    { id: 'buy-10k-tsla', label: 'Buy 10k TSLA', score: 0.728, beat: true },
    { id: 'sell-10k-spy', label: 'Sell 10k SPY', score: 0.456, beat: false },
  ],
}

export const ppoHoldout = {
  baseline: 'VWAP-match',
  ticker: 'AAPL',
  nEpisodes: 60,
  winRate: 0.55,
  medianAdvantageBps: 0.48,
  meanNotionalUsd: 2_616_418,
}

// Headline stat band shown on the landing page.
export const headlineStats = [
  { label: 'HUD tasks beating VWAP', value: '2 / 3', sub: 'Claude execution agent' },
  { label: 'PPO win-rate, unseen days', value: '55%', sub: 'vs VWAP-match baseline' },
  { label: 'Tickers, real SIP data', value: '4', sub: 'AAPL · NVDA · TSLA · SPY' },
  { label: 'Execution policies', value: '5', sub: 'TWAP · VWAP · PPO · LLMs' },
]
