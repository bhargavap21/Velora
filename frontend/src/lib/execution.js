// Shared execution-math + presentation helpers used across the demo pages.
// Mirrors the server's metric definitions (execution_env/server.py:_episode_metrics
// and simulator/benchmark.py) so the UI numbers match the backend exactly.

// Per-policy display metadata: label + chart color. Tuned for the obsidian/ember-gold
// design system — gold reserved for the LLM star, pearl for the RL agent, warm red for
// the naive baseline, neutral grays for the reference policies.
export const POLICY_META = {
  ppo: { label: 'PPO (RL agent)', short: 'PPO', color: '#acafb9' },        // pearl
  naive_twap: { label: 'TWAP (equal-time)', short: 'Naive TWAP', color: '#c0553a' }, // warm red
  twap: { label: 'VWAP-match', short: 'VWAP-match', color: '#5e616e' },    // fog
  llm: { label: 'Claude', short: 'Claude', color: '#cc9166' },             // ember gold
  fireworks: { label: 'GPT-OSS', short: 'GPT-OSS', color: '#ae9357' },     // molten gold
}

export const BENCHMARK_COLOR = '#777a88' // ash — neutral VWAP reference line
export const MARKET_COLOR = '#464853'    // steel — understated market price

export function policyMeta(id) {
  return POLICY_META[id] ?? { label: String(id).toUpperCase(), short: String(id).toUpperCase(), color: '#9194a1' }
}

export function formatUsd(n, opts = {}) {
  if (n == null || Number.isNaN(n)) return '—'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: opts.maximumFractionDigits ?? 0,
    ...opts,
  }).format(n)
}

export function formatBps(n, { signed = true, digits = 1 } = {}) {
  if (n == null || Number.isNaN(n)) return '—'
  const sign = signed && n >= 0 ? '+' : ''
  return `${sign}${n.toFixed(digits)} bps`
}

// Cumulative benchmark VWAP through slice `upTo`, matching the server benchmark:
// compute_vwap(path[1:], volume_curve * total_shares). The benchmark price for slice i
// (1-indexed fills) is path[i], weighted by volume_curve[i-1] * total_shares.
export function benchmarkVwapUpTo(path, volumeCurve, totalShares, upTo) {
  let notional = 0
  let qty = 0
  for (let i = 1; i <= upTo; i++) {
    const w = (volumeCurve[i - 1] ?? 0) * totalShares
    notional += path[i] * w
    qty += w
  }
  return qty > 0 ? notional / qty : null
}

// Cumulative agent VWAP through slice `upTo` from an exec trace.
export function agentVwapUpTo(execPrices, execQuantities, upTo) {
  let notional = 0
  let qty = 0
  for (let i = 0; i < upTo; i++) {
    const p = execPrices[i]
    const q = execQuantities[i]
    if (p == null || q == null) continue
    notional += p * q
    qty += q
  }
  return qty > 0 ? notional / qty : null
}

// Slippage vs benchmark in bps, sign-adjusted so positive = better than benchmark for
// both buy and sell. Returns null until at least one slice has executed.
export function slippageBpsUpTo(scenario, execPrices, execQuantities, upTo) {
  const bench = benchmarkVwapUpTo(scenario.path, scenario.volume_curve, scenario.total_shares, upTo)
  const agent = agentVwapUpTo(execPrices, execQuantities, upTo)
  if (bench == null || agent == null || bench <= 0) return null
  const sign = scenario.side === 'sell' ? -1 : 1
  return (sign * (bench - agent) / bench) * 10_000
}

// Dollar impact of a bps figure on a given notional. advantageBps can be negative.
export function usdFromBps(bps, notional) {
  if (bps == null || notional == null) return null
  return (bps / 10_000) * notional
}

// Build recharts rows for a multi-policy comparison through `upTo`. Each row carries the
// market price, the benchmark VWAP, and one cumulative agent-VWAP series per policy keyed
// by `vwap_<policyId>`.
export function buildComparisonRows(scenario, policies, upTo) {
  const rows = []
  for (let i = 0; i <= upTo; i++) {
    const row = {
      i,
      price: scenario.path[i],
      benchmarkVwap: benchmarkVwapUpTo(scenario.path, scenario.volume_curve, scenario.total_shares, i),
    }
    for (const p of policies) {
      row[`vwap_${p.policy}`] = agentVwapUpTo(p.exec_prices, p.exec_quantities, i)
    }
    rows.push(row)
  }
  return rows
}

// Histogram bins for an array of per-episode advantage values (bps).
export function histogram(values, binCount = 21) {
  if (!values.length) return []
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const width = span / binCount
  const bins = Array.from({ length: binCount }, (_, k) => ({
    x: min + (k + 0.5) * width,
    x0: min + k * width,
    count: 0,
  }))
  for (const v of values) {
    let idx = Math.floor((v - min) / width)
    if (idx >= binCount) idx = binCount - 1
    if (idx < 0) idx = 0
    bins[idx].count += 1
  }
  return bins
}
