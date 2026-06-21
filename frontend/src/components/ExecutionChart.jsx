import {
  ComposedChart, Line, Area, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
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

    rows.push({ i, price: path[i], agentVwap, benchmarkVwap, execQty })
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

const TOOLTIP_STYLE = {
  backgroundColor: '#121317',
  border: '1px solid #2e3038',
  borderRadius: 4,
  fontSize: 12,
  color: '#acafb9',
  boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
}

export default function ExecutionChart({ episode, currentSlice }) {
  const data = buildChartData(episode, currentSlice)

  return (
    <ResponsiveContainer width="100%" height={320}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 12, left: 0 }}>
        <defs>
          {/* Molten gold area fill — agent execution price */}
          <linearGradient id="agentFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#cc9166" stopOpacity={0.2} />
            <stop offset="100%" stopColor="#cc9166" stopOpacity={0}   />
          </linearGradient>
          {/* Subtle fill under market price */}
          <linearGradient id="priceFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#464853" stopOpacity={0.12} />
            <stop offset="100%" stopColor="#464853" stopOpacity={0}    />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="3 3" stroke="#2e3038" vertical={false} />

        <XAxis
          dataKey="i"
          stroke="#2e3038"
          tick={{ fontSize: 11, fill: '#777a88', fontFamily: 'Inter, sans-serif' }}
          axisLine={{ stroke: '#2e3038' }}
          tickLine={false}
          label={{ value: 'Slice', position: 'insideBottom', offset: -4, fontSize: 11, fill: '#5e616e' }}
        />
        <YAxis
          stroke="#2e3038"
          tick={{ fontSize: 11, fill: '#777a88', fontFamily: 'Inter, sans-serif' }}
          axisLine={false}
          tickLine={false}
          domain={['auto', 'auto']}
          width={56}
        />

        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          labelStyle={{ color: '#5e616e', marginBottom: 4 }}
          itemStyle={{ color: '#acafb9' }}
          cursor={{ stroke: '#2e3038', strokeWidth: 1 }}
          formatter={(value, name) => [
            typeof value === 'number' ? value.toFixed(3) : value,
            name,
          ]}
        />

        <Legend
          wrapperStyle={{
            fontSize: 12,
            color: '#9194a1',
            fontFamily: 'Inter, sans-serif',
            paddingTop: 12,
          }}
        />

        {/* Market price — understated background line */}
        <Area
          type="monotone"
          dataKey="price"
          name="Market price"
          stroke="#464853"
          strokeWidth={1}
          fill="url(#priceFill)"
          dot={false}
          isAnimationActive={false}
        />

        {/* VWAP benchmark — ember gold dashed */}
        <Line
          type="monotone"
          dataKey="benchmarkVwap"
          name="VWAP benchmark"
          stroke="#cc9166"
          strokeDasharray="4 3"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />

        {/* Agent avg. execution price — pearl with molten gold fill */}
        <Area
          type="monotone"
          dataKey="agentVwap"
          name="Agent avg. price"
          stroke="#acafb9"
          strokeWidth={2}
          fill="url(#agentFill)"
          dot={false}
          isAnimationActive={false}
        />

        {/* Execution sizes — gold accent dots */}
        <Scatter
          dataKey="execQty"
          name="Exec size"
          fill="#cc9166"
          opacity={0.55}
        />
      </ComposedChart>
    </ResponsiveContainer>
  )
}
