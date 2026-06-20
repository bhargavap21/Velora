import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import Navbar from '../components/Navbar'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import {
  Table, TableHeader, TableBody, TableRow, TableHead, TableCell,
} from '@/components/ui/table'
import {
  Accordion, AccordionItem, AccordionTrigger, AccordionContent,
} from '@/components/ui/accordion'
import { getResults } from '../api'

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

const SEED_LABELS = {
  tsla_rsi:     'TSLA RSI Momentum',
  spy_ema_rsi:  'SPY EMA + RSI',
  nvda_sma_rsi: 'NVDA SMA + RSI',
  aapl_sma_rsi: 'AAPL SMA + RSI',
  tsla_sma_rsi: 'TSLA SMA + RSI',
}

function DeltaCell({ value }) {
  const positive = value >= 0
  return (
    <span className={positive ? 'text-green-500' : 'text-red-500'}>
      {positive ? '+' : ''}{value.toFixed(2)}
    </span>
  )
}

export default function Results() {
  const { runId } = useParams()
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    async function poll() {
      try {
        const data = await getResults(runId)
        if (cancelled) return
        if (data.status === 202) {
          setTimeout(poll, 2000)
        } else {
          setResults(data.results)
        }
      } catch (e) {
        if (!cancelled) setError(e.message)
      }
    }
    poll()
    return () => { cancelled = true }
  }, [runId])

  if (error) return (
    <div className="flex min-h-screen items-center justify-center bg-background text-foreground">
      <p className="text-destructive">{error}</p>
    </div>
  )

  if (!results) return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="flex items-center justify-center py-32 text-muted-foreground">Loading results…</div>
    </div>
  )

  const ranked = [...results].sort((a, b) => b.total_reward - a.total_reward)

  const equityCurveData = ranked.some(r => r.equity_curve?.length > 0)
    ? (() => {
        const len = Math.max(...ranked.map(r => r.equity_curve?.length || 0))
        return Array.from({ length: len }, (_, i) => {
          const row = { i }
          ranked.forEach(r => {
            if (r.equity_curve?.[i] != null) row[r.seed_name] = r.equity_curve[i]
          })
          return row
        })
      })()
    : null

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="mx-auto max-w-5xl space-y-10 px-4 py-8">
        <h2 className="text-xl font-bold">Results</h2>

        {/* Ranked table */}
        <Table>
          <TableHeader>
            <TableRow>
              {['Rank','Strategy','Ticker','Init Sharpe','Final Sharpe','Delta','Reward','Turns'].map(h => (
                <TableHead key={h}>{h}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {ranked.map((r, i) => (
              <TableRow key={r.seed_name}>
                <TableCell className="text-muted-foreground">{i + 1}</TableCell>
                <TableCell className="font-medium">{SEED_LABELS[r.seed_name] || r.seed_name}</TableCell>
                <TableCell>
                  <Badge variant="secondary">{r.ticker}</Badge>
                </TableCell>
                <TableCell>{r.initial_sharpe?.toFixed(2)}</TableCell>
                <TableCell>{r.final_sharpe?.toFixed(2)}</TableCell>
                <TableCell>
                  <DeltaCell value={(r.final_sharpe || 0) - (r.initial_sharpe || 0)} />
                </TableCell>
                <TableCell className="text-primary">{r.total_reward?.toFixed(2)}</TableCell>
                <TableCell>{r.turns_taken}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        {/* Equity curves */}
        {equityCurveData && (
          <div>
            <h3 className="mb-4 text-base font-semibold">Portfolio Growth by Seed</h3>
            <Card>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={equityCurveData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="i" stroke="var(--muted-foreground)" tick={{ fontSize: 11 }} />
                    <YAxis stroke="var(--muted-foreground)" tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8 }}
                      labelStyle={{ color: 'var(--muted-foreground)' }}
                    />
                    <Legend />
                    {ranked.map((r, i) => (
                      <Line
                        key={r.seed_name}
                        type="monotone"
                        dataKey={r.seed_name}
                        name={SEED_LABELS[r.seed_name] || r.seed_name}
                        stroke={COLORS[i % COLORS.length]}
                        dot={false}
                        strokeWidth={2}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Mutation history accordion */}
        <div>
          <h3 className="mb-4 text-base font-semibold">Mutation History</h3>
          <Card>
            <CardContent>
              <Accordion type="multiple">
                {ranked.map(r => (
                  <AccordionItem key={r.seed_name} value={r.seed_name}>
                    <AccordionTrigger>
                      {SEED_LABELS[r.seed_name] || r.seed_name}
                    </AccordionTrigger>
                    <AccordionContent>
                      <div className="divide-y divide-border">
                        {(r.turns || []).map((t, i) => (
                          <div key={i} className="py-3 text-xs">
                            <div className="flex items-center gap-3 text-foreground/80">
                              <span className="w-12 text-muted-foreground">Turn {t.turn}</span>
                              <span className="font-medium">{t.mutation_type}</span>
                              <span className="ml-auto text-primary">reward: {t.reward.toFixed(2)}</span>
                            </div>
                            {t.reasoning && (
                              <p className="mt-1 pl-12 text-muted-foreground">{t.reasoning}</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
