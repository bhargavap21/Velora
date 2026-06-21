import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { buildComparisonRows, policyMeta, BENCHMARK_COLOR, MARKET_COLOR } from '@/lib/execution'

// Renders the shared price path + benchmark VWAP, with one cumulative agent-VWAP line per
// policy, revealed up to `currentSlice`. Used by the Showdown and Sandbox overlay views.
export default function MultiExecutionChart({ scenario, policies, currentSlice, height = 340 }) {
  const data = buildComparisonRows(scenario, policies, currentSlice)

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="i"
          stroke="var(--muted-foreground)"
          tick={{ fontSize: 11 }}
          label={{ value: 'Slice', position: 'insideBottom', offset: -2, fontSize: 11, fill: 'var(--muted-foreground)' }}
        />
        <YAxis
          stroke="var(--muted-foreground)"
          tick={{ fontSize: 11 }}
          domain={['auto', 'auto']}
          tickFormatter={(v) => v.toFixed(2)}
          width={56}
        />
        <Tooltip
          contentStyle={{ backgroundColor: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: 'var(--muted-foreground)' }}
          formatter={(value, name) => [typeof value === 'number' ? value.toFixed(3) : value, name]}
          labelFormatter={(l) => `Slice ${l}`}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          type="monotone" dataKey="price" name="Market price"
          stroke={MARKET_COLOR} dot={false} strokeWidth={1.25} isAnimationActive={false}
        />
        <Line
          type="monotone" dataKey="benchmarkVwap" name="VWAP benchmark"
          stroke={BENCHMARK_COLOR} strokeDasharray="4 3" dot={false} strokeWidth={2} isAnimationActive={false}
        />
        {policies.map((p) => {
          const meta = policyMeta(p.policy)
          return (
            <Line
              key={p.policy}
              type="monotone"
              dataKey={`vwap_${p.policy}`}
              name={`${meta.short} avg fill`}
              stroke={meta.color}
              dot={false}
              strokeWidth={2.25}
              isAnimationActive={false}
            />
          )
        })}
      </ComposedChart>
    </ResponsiveContainer>
  )
}
