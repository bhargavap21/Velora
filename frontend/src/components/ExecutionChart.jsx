import {
  ComposedChart, Line, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

/**
 * Builds chart-ready rows for slices [0, upToSlice] (inclusive) from a raw episode
 * trace (see execution_env/server.py's GET /api/episode for the exact shape this expects).
 */
export function buildChartData(episode, upToSlice) {
  const { path, volume_curve, exec_prices, exec_quantities, total_shares } = episode
  const rows = []
  let cumAgentNotional = 0
  let cumAgentQty = 0
  let cumBenchNotional = 0
  let cumBenchQty = 0

  for (let i = 0; i <= upToSlice; i++) {
    let agentVwap = null
    let execQty = null
    if (i > 0) {
      const execPrice = exec_prices[i - 1]
      const qty = exec_quantities[i - 1]
      cumAgentNotional += execPrice * qty
      cumAgentQty += qty
      agentVwap = cumAgentQty > 0 ? cumAgentNotional / cumAgentQty : null
      execQty = qty

      const benchQty = volume_curve[i - 1] * total_shares
      cumBenchNotional += path[i] * benchQty
      cumBenchQty += benchQty
    }
    const benchmarkVwap = cumBenchQty > 0 ? cumBenchNotional / cumBenchQty : null

    rows.push({
      i,
      price: path[i],
      agentVwap,
      benchmarkVwap,
      execQty,
    })
  }
  return rows
}

export function currentSlippageBps(episode, upToSlice) {
  const rows = buildChartData(episode, upToSlice)
  const last = rows[rows.length - 1]
  if (!last || last.agentVwap == null || last.benchmarkVwap == null) return null
  const sign = episode.side === 'sell' ? -1 : 1
  return sign * ((last.benchmarkVwap - last.agentVwap) / last.benchmarkVwap) * 10_000
}

export default function ExecutionChart({ episode, currentSlice }) {
  const data = buildChartData(episode, currentSlice)

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis dataKey="i" stroke="var(--muted-foreground)" tick={{ fontSize: 11 }} label={{ value: 'Slice', position: 'insideBottom', offset: -2, fontSize: 11, fill: 'var(--muted-foreground)' }} />
        <YAxis stroke="var(--muted-foreground)" tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8 }}
          labelStyle={{ color: 'var(--muted-foreground)' }}
          formatter={(value, name) => [typeof value === 'number' ? value.toFixed(3) : value, name]}
        />
        <Legend />
        <Line type="monotone" dataKey="price" name="Market price" stroke="#64748b" dot={false} strokeWidth={1.5} isAnimationActive={false} />
        <Line type="monotone" dataKey="benchmarkVwap" name="VWAP benchmark" stroke="#f59e0b" strokeDasharray="4 3" dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line type="monotone" dataKey="agentVwap" name="Agent avg. exec price" stroke="#10b981" dot={false} strokeWidth={2} isAnimationActive={false} />
        <Scatter dataKey="execQty" name="Execution size" fill="#3b82f6" />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
