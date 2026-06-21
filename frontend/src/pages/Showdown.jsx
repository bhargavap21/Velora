import { useCallback, useEffect, useRef, useState } from 'react'
import Navbar from '../components/Navbar'
import MultiExecutionChart from '../components/MultiExecutionChart'
import AnimatedNumber from '../components/AnimatedNumber'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  policyMeta, formatUsd, formatBps, slippageBpsUpTo, usdFromBps,
} from '@/lib/execution'

const inputClass =
  'h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

const TICK_MS = 180

function filledThrough(quantities, upTo, total) {
  let q = 0
  for (let i = 0; i < upTo; i++) q += quantities[i] ?? 0
  return total > 0 ? q / total : 0
}

export default function Showdown() {
  const [cfg, setCfg] = useState(null)
  const [policies, setPolicies] = useState([])
  const [form, setForm] = useState(null)
  const [result, setResult] = useState(null)
  const [cursor, setCursor] = useState(0)
  const [running, setRunning] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const timerRef = useRef(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/config').then(r => r.json()),
      fetch('/api/policies').then(r => r.json()),
    ]).then(([config, pol]) => {
      setCfg(config)
      setPolicies(pol.available ?? ['naive_twap', 'twap'])
      setForm({
        ticker: 'TSLA',
        side: config.defaults.side,
        adv_pct: config.defaults.adv_pct ?? 8,
        // A representative *random* held-out day (not a cherry-picked stress regime):
        // seed 7 resolves to 2026-03-23, where PPO's edge over VWAP-match (~+31 bps) sits
        // right at the median TSLA result. Change the regime or clear the seed to explore
        // freely -- including the ~20% of days where the baseline wins.
        regime: 'random',
        seed: 7,
        // Default baseline is VWAP-match -- the industry-standard benchmark. Switch to
        // "Naive TWAP" to see the larger (but much easier) edge over the equal-time floor.
        policyA: (pol.available ?? []).includes('ppo') ? 'ppo' : 'twap',
        policyB: (pol.available ?? []).includes('twap') ? 'twap' : 'naive_twap',
      })
    }).catch(() => setError('Failed to load configuration'))
  }, [])

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [])

  useEffect(() => stopTimer, [stopTimer])

  const run = useCallback(() => {
    if (!form) return
    stopTimer()
    setLoading(true)
    setError(null)
    setResult(null)
    setCursor(0)
    setRunning(false)

    const params = new URLSearchParams()
    params.set('policies', `${form.policyA},${form.policyB}`)
    params.set('ticker', form.ticker)
    params.set('side', form.side)
    params.set('adv_pct', String(form.adv_pct))
    if (form.regime && form.regime !== 'random') params.set('regime', form.regime)
    if (form.seed !== '' && form.seed != null) params.set('seed', String(form.seed))

    fetch(`/api/compare?${params.toString()}`)
      .then(async r => {
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || 'Request failed')
        return r.json()
      })
      .then(data => {
        setResult(data)
        setLoading(false)
        setRunning(true)
        setCursor(0)
        let c = 0
        timerRef.current = setInterval(() => {
          c += 1
          setCursor(c)
          if (c >= data.n_slices) {
            stopTimer()
            setRunning(false)
          }
        }, TICK_MS)
      })
      .catch(err => {
        setError(err.message || 'Request failed')
        setLoading(false)
      })
  }, [form, stopTimer])

  if (!cfg || !form) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Navbar />
        <div className="mx-auto max-w-6xl px-4 py-16 text-center text-sm text-muted-foreground">Loading…</div>
      </div>
    )
  }

  const tickerInfo = cfg.ticker_info[form.ticker]
  const scenario = result
  const notional = scenario?.notional_usd

  const live = (scenario?.policies ?? []).map(p => {
    const slip = scenario ? slippageBpsUpTo(scenario, p.exec_prices, p.exec_quantities, cursor) : null
    return {
      ...p,
      meta: policyMeta(p.policy),
      liveSlip: slip,
      liveUsd: usdFromBps(slip, notional),
      fill: scenario ? filledThrough(p.exec_quantities, cursor, scenario.total_shares) : 0,
    }
  })

  // Advantage of A over B (positive = A better). Uses live cumulative slippage.
  const a = live[0]
  const b = live[1]
  const advBps = a && b && a.liveSlip != null && b.liveSlip != null ? a.liveSlip - b.liveSlip : null
  const advUsd = usdFromBps(advBps, notional)
  const done = scenario && cursor >= scenario.n_slices
  const leader = advBps == null ? null : advBps >= 0 ? a : b

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6">
          <h2 className="text-[28px] font-light tracking-[-0.02em] text-foreground">
            Head-to-head{' '}
            <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 500, color: '#cc9166' }}>showdown.</span>
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Two execution policies fill the <span className="font-medium text-foreground">same order</span> on the
            {' '}<span className="font-medium text-foreground">same price path</span> — the only variable is the policy.
            Watch them race, slice by slice, and see who lags VWAP by less.
          </p>
        </div>

        {/* Scenario bar */}
        <Card className="mb-6">
          <CardContent className="grid grid-cols-2 gap-4 pt-6 md:grid-cols-8">
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Ticker</label>
              <input className={inputClass} list="ticker-suggestions" value={form.ticker}
                placeholder="Any symbol"
                onChange={e => setForm({ ...form, ticker: e.target.value.toUpperCase() })} />
              <datalist id="ticker-suggestions">
                {cfg.tickers.map(t => <option key={t} value={t} />)}
              </datalist>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Side</label>
              <div className="flex gap-1">
                {['buy', 'sell'].map(s => (
                  <Button key={s} size="sm" variant={form.side === s ? 'default' : 'outline'}
                    className="flex-1 capitalize" onClick={() => setForm({ ...form, side: s })}>{s}</Button>
                ))}
              </div>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Order size (% ADV)</label>
              <input type="number" className={inputClass} min={cfg.constraints.adv_pct.min}
                max={cfg.constraints.adv_pct.max} step={cfg.constraints.adv_pct.step}
                value={form.adv_pct} onChange={e => setForm({ ...form, adv_pct: Number(e.target.value) })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Market day</label>
              <select className={inputClass} value={form.regime} onChange={e => setForm({ ...form, regime: e.target.value })}>
                {cfg.regime_samples.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Seed</label>
              <input type="number" className={inputClass} placeholder="random"
                value={form.seed} onChange={e => setForm({ ...form, seed: e.target.value === '' ? '' : Number(e.target.value) })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Agent</label>
              <select className={inputClass} value={form.policyA} onChange={e => setForm({ ...form, policyA: e.target.value })}>
                {policies.map(p => <option key={p} value={p}>{policyMeta(p).short}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Baseline</label>
              <select className={inputClass} value={form.policyB} onChange={e => setForm({ ...form, policyB: e.target.value })}>
                {policies.map(p => <option key={p} value={p}>{policyMeta(p).short}</option>)}
              </select>
            </div>
            <div className="flex items-end">
              <Button className="w-full" onClick={run} disabled={loading || running}>
                {loading ? 'Setting up…' : running ? 'Racing…' : 'Run showdown'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {error && (
          <Card className="mb-4 border-[#c0553a]/50"><CardContent className="pt-6 text-sm text-[#c0553a]">{error}</CardContent></Card>
        )}

        {scenario && (
          <>
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{scenario.ticker}</Badge>
              <Badge variant="secondary">{scenario.date}</Badge>
              <Badge variant="secondary" className="capitalize">{scenario.side} {scenario.total_shares.toLocaleString()} sh</Badge>
              <Badge variant="secondary">{scenario.order_adv_pct}% of ADV</Badge>
              <Badge variant="secondary">{formatUsd(scenario.notional_usd)}</Badge>
              <Badge variant={scenario.data_source === 'real_minute' ? 'default' : 'outline'}>
                {scenario.data_source === 'real_minute' ? 'Real minute bars' : 'Synthetic path'}
              </Badge>
              {running ? (
                <span className="flex items-center gap-1.5 text-xs font-medium text-[#c0553a]">
                  <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#c0553a]" /> LIVE · slice {cursor}/{scenario.n_slices}
                </span>
              ) : done ? <span className="text-xs font-medium text-[#cc9166]">✓ Complete</span> : null}
            </div>

            {/* Scoreboard */}
            <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
              {live.map((p, idx) => {
                const isLeader = leader && leader.policy === p.policy && advBps != null && Math.abs(advBps) > 0.01
                return (
                  <Card key={p.policy} className="relative overflow-hidden"
                    style={{ boxShadow: isLeader ? `inset 0 0 0 1.5px ${p.meta.color}` : undefined }}>
                    <div className="absolute left-0 top-0 h-full w-1" style={{ backgroundColor: p.meta.color }} />
                    <CardContent className="pt-6 pl-6">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: p.meta.color }} />
                          <span className="font-semibold">{p.meta.label}</span>
                          <span className="text-xs text-muted-foreground">{idx === 0 ? 'agent' : 'baseline'}</span>
                        </div>
                        {isLeader && <Badge style={{ backgroundColor: p.meta.color, color: 'white' }}>Leading</Badge>}
                      </div>
                      <div className="mt-4 grid grid-cols-3 gap-3">
                        <div>
                          <p className="text-xs text-muted-foreground">Slippage vs VWAP</p>
                          <p className="mt-0.5 text-xl font-bold tabular-nums" style={{ color: p.meta.color }}>
                            {p.liveSlip == null ? '—'
                              : <AnimatedNumber value={p.liveSlip} format={(n) => formatBps(n)} />}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">vs VWAP ($)</p>
                          <p className="mt-0.5 text-xl font-bold tabular-nums">
                            {p.liveUsd == null ? '—'
                              : <AnimatedNumber value={p.liveUsd} format={(n) => formatUsd(n)} />}
                          </p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Filled</p>
                          <p className="mt-0.5 text-xl font-bold tabular-nums">{(p.fill * 100).toFixed(0)}%</p>
                        </div>
                      </div>
                      <Progress className="mt-3" value={p.fill * 100} />
                    </CardContent>
                  </Card>
                )
              })}
            </div>

            {/* Chart */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Average execution price vs VWAP benchmark</CardTitle>
                <CardDescription>
                  Open ${scenario.open_price.toFixed(2)} → Close ${scenario.close_price.toFixed(2)} · ADV {Math.round(scenario.adv).toLocaleString()}
                  {' '}· lower is better for a buy, higher for a sell
                </CardDescription>
              </CardHeader>
              <CardContent>
                <MultiExecutionChart scenario={scenario} policies={scenario.policies} currentSlice={cursor} />
              </CardContent>
            </Card>

            {/* Verdict */}
            {advBps != null && (
              <Card className="mt-4" style={{ boxShadow: done ? `inset 0 0 0 1.5px ${leader?.meta.color}` : undefined }}>
                <CardContent className="pt-6">
                  <div className="flex flex-col items-center gap-1 text-center">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground">
                      {done ? 'Final verdict' : 'Current standing'}
                    </p>
                    <p className="text-lg">
                      <span className="font-bold" style={{ color: a.meta.color }}>{a.meta.short}</span>
                      {' '}{advBps >= 0 ? 'beats' : 'trails'}{' '}
                      <span className="font-bold" style={{ color: b.meta.color }}>{b.meta.short}</span> by
                    </p>
                    <p className="text-4xl font-extrabold tabular-nums" style={{ color: leader?.meta.color }}>
                      <AnimatedNumber value={Math.abs(advBps)} format={(n) => `${n.toFixed(1)} bps`} />
                    </p>
                    <p className="text-sm text-muted-foreground">
                      ≈ <span className="font-semibold text-foreground">{formatUsd(Math.abs(advUsd))}</span> on this
                      {' '}{formatUsd(scenario.notional_usd)} order
                    </p>
                    {done && (
                      <p className="mt-2 max-w-xl text-xs text-muted-foreground">
                        Same ticker, same day, same seed → identical price path. The entire difference is attributable
                        to <span className="font-medium">how</span> each policy scheduled the order across the day.
                        For statistical proof across many unseen days, see the Proof page.
                      </p>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}

            {scenario.policies.some(p => p.llm_reasoning) && (
              <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
                {scenario.policies.filter(p => p.llm_reasoning).map(p => (
                  <Card key={p.policy}>
                    <CardContent className="pt-4">
                      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {policyMeta(p.policy).label} reasoning
                      </p>
                      <p className="text-sm leading-relaxed">{p.llm_reasoning}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </>
        )}

        {!scenario && !loading && (
          <Card>
            <CardContent className="flex h-72 flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
              <p>Pick a scenario and hit <span className="font-medium text-foreground">Run showdown</span></p>
              <p className="text-xs">Clear the seed for a genuinely random day, or shuffle regimes — the Proof page has the win-rate across all of them</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
