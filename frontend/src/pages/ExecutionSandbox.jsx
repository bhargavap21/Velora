import { useCallback, useEffect, useRef, useState } from 'react'
import Navbar from '../components/Navbar'
import MultiExecutionChart from '../components/MultiExecutionChart'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from '@/components/ui/table'
import {
  policyMeta, formatUsd, formatBps, slippageBpsUpTo, usdFromBps,
} from '@/lib/execution'

const inputClass =
  'h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

const TICK_MS = 150

export default function ExecutionSandbox() {
  const [cfg, setCfg] = useState(null)
  const [available, setAvailable] = useState(['naive_twap', 'twap'])
  const [form, setForm] = useState(null)
  const [selected, setSelected] = useState([])
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
      const avail = pol.available ?? ['naive_twap', 'twap']
      setCfg(config)
      setAvailable(avail)
      setSelected(avail.includes('ppo') ? ['ppo', 'naive_twap'] : ['twap', 'naive_twap'])
      setForm({
        ticker: config.defaults.ticker,
        side: config.defaults.side,
        orderMode: 'adv_pct',
        adv_pct: config.defaults.adv_pct ?? 8,
        total_shares: config.defaults.total_shares,
        capital_usd: config.defaults.capital_usd,
        n_slices: config.defaults.n_slices,
        timeframeId: config.timeframe_presets.find(p => p.n_slices === config.defaults.n_slices)?.id ?? '15min',
        dateMode: 'regime',
        date: config.ticker_info[config.defaults.ticker]?.date_range.end ?? '',
        regime: 'high_vol',
        seed: '',
      })
    }).catch(() => setError('Failed to load sandbox configuration'))
  }, [])

  const stopTimer = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null }
  }, [])
  useEffect(() => stopTimer, [stopTimer])

  const update = (patch) => setForm(prev => {
    const next = { ...prev, ...patch }
    if (patch.ticker && cfg?.ticker_info?.[patch.ticker]) next.date = cfg.ticker_info[patch.ticker].date_range.end
    return next
  })

  const togglePolicy = (p) => setSelected(prev =>
    prev.includes(p) ? prev.filter(x => x !== p) : [...prev, p])

  const run = useCallback(() => {
    if (!form || !selected.length) return
    stopTimer()
    setLoading(true); setError(null); setResult(null); setCursor(0); setRunning(false)

    const params = new URLSearchParams()
    params.set('policies', selected.join(','))
    params.set('ticker', form.ticker)
    params.set('side', form.side)
    params.set('n_slices', String(form.n_slices))
    if (form.orderMode === 'adv_pct') params.set('adv_pct', String(form.adv_pct))
    else if (form.orderMode === 'capital') params.set('capital_usd', String(form.capital_usd))
    else params.set('total_shares', String(form.total_shares))
    if (form.dateMode === 'specific' && form.date) params.set('date', form.date)
    else if (form.dateMode === 'regime') params.set('regime', form.regime)
    if (form.seed !== '' && form.seed != null) params.set('seed', String(form.seed))

    fetch(`/api/compare?${params.toString()}`)
      .then(async r => { if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || 'Request failed'); return r.json() })
      .then(data => {
        setResult(data); setLoading(false); setRunning(true); setCursor(0)
        let c = 0
        timerRef.current = setInterval(() => {
          c += 1; setCursor(c)
          if (c >= data.n_slices) { stopTimer(); setRunning(false) }
        }, TICK_MS)
      })
      .catch(err => { setError(err.message); setLoading(false) })
  }, [form, selected, stopTimer])

  if (!cfg || !form) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Navbar />
        <div className="mx-auto max-w-6xl px-4 py-16 text-center text-sm text-muted-foreground">Loading sandbox…</div>
      </div>
    )
  }

  const tickerInfo = cfg.ticker_info[form.ticker]
  const ppoBlocked = form.n_slices !== 26
  const scenario = result

  const rows = (scenario?.policies ?? []).map(p => {
    const slip = scenario ? slippageBpsUpTo(scenario, p.exec_prices, p.exec_quantities, cursor) : null
    return { ...p, meta: policyMeta(p.policy), liveSlip: slip, liveUsd: usdFromBps(slip, scenario?.notional_usd) }
  })
  const done = scenario && cursor >= scenario.n_slices
  // Best (highest slippage = least cost) among shown rows, for highlighting.
  const best = rows.reduce((acc, r) => (r.liveSlip != null && (acc == null || r.liveSlip > acc.liveSlip) ? r : acc), null)

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6">
          <h2 className="text-[28px] font-light tracking-[-0.02em] text-foreground">
            Execution{' '}
            <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 500, color: '#cc9166' }}>sandbox.</span>
          </h2>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Configure a scenario, pick any policies to overlay, and run them on the same intraday path.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
          <Card className="h-fit lg:sticky lg:top-6">
            <CardHeader>
              <CardTitle className="text-base">Scenario</CardTitle>
              <CardDescription>Pin a market day or sample a regime</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Ticker</label>
                <select className={inputClass} value={form.ticker} onChange={e => update({ ticker: e.target.value })}>
                  {cfg.tickers.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Side</label>
                <div className="flex gap-2">
                  {['buy', 'sell'].map(s => (
                    <Button key={s} size="sm" variant={form.side === s ? 'default' : 'outline'}
                      className="flex-1 capitalize" onClick={() => update({ side: s })}>{s}</Button>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Order size</label>
                <div className="mb-2 flex gap-1">
                  {[{ id: 'adv_pct', label: '% ADV' }, { id: 'capital', label: '$' }, { id: 'shares', label: 'Shares' }].map(m => (
                    <Button key={m.id} size="sm" variant={form.orderMode === m.id ? 'default' : 'outline'}
                      className="flex-1 text-xs" onClick={() => update({ orderMode: m.id })}>{m.label}</Button>
                  ))}
                </div>
                {form.orderMode === 'adv_pct' && (
                  <>
                    <input type="number" className={inputClass} min={cfg.constraints.adv_pct.min}
                      max={cfg.constraints.adv_pct.max} step={cfg.constraints.adv_pct.step}
                      value={form.adv_pct} onChange={e => update({ adv_pct: Number(e.target.value) })} />
                    <p className="mt-1 text-xs text-muted-foreground">
                      ≈ {tickerInfo?.median_adv ? Math.floor(tickerInfo.median_adv * form.adv_pct / 100 / 1000) * 1000 : 0}
                      {' '}sh · ADV {tickerInfo?.median_adv?.toLocaleString()}
                    </p>
                  </>
                )}
                {form.orderMode === 'capital' && (
                  <input type="number" className={inputClass} min={cfg.constraints.capital_usd.min}
                    max={cfg.constraints.capital_usd.max} step={cfg.constraints.capital_usd.step}
                    value={form.capital_usd} onChange={e => update({ capital_usd: Number(e.target.value) })} />
                )}
                {form.orderMode === 'shares' && (
                  <input type="number" className={inputClass} min={cfg.constraints.total_shares.min}
                    max={cfg.constraints.total_shares.max} step={cfg.constraints.total_shares.step}
                    value={form.total_shares} onChange={e => update({ total_shares: Number(e.target.value) })} />
                )}
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Timeframe</label>
                <select className={inputClass} value={form.timeframeId}
                  onChange={e => { const p = cfg.timeframe_presets.find(x => x.id === e.target.value); update({ timeframeId: e.target.value, n_slices: p.n_slices }) }}>
                  {cfg.timeframe_presets.map(p => (
                    <option key={p.id} value={p.id}>{p.label} ({p.n_slices} slices)</option>
                  ))}
                </select>
              </div>

              <Separator />

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Replay day</label>
                <select className={`${inputClass} mb-2`} value={form.dateMode} onChange={e => update({ dateMode: e.target.value })}>
                  <option value="random">Random day</option>
                  <option value="regime">Regime sample</option>
                  <option value="specific">Specific date</option>
                </select>
                {form.dateMode === 'regime' && (
                  <select className={inputClass} value={form.regime} onChange={e => update({ regime: e.target.value })}>
                    {cfg.regime_samples.map(r => <option key={r.id} value={r.id}>{r.label}</option>)}
                  </select>
                )}
                {form.dateMode === 'specific' && tickerInfo && (
                  <input type="date" className={inputClass} min={tickerInfo.date_range.start} max={tickerInfo.date_range.end}
                    value={form.date || tickerInfo.date_range.end} onChange={e => update({ date: e.target.value })} />
                )}
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Seed (optional)</label>
                <input type="number" className={inputClass} placeholder="Reproducible day + path"
                  value={form.seed} onChange={e => update({ seed: e.target.value })} />
              </div>

              <Separator />

              <div>
                <label className="mb-2 block text-xs font-medium text-muted-foreground">Policies to overlay</label>
                <div className="space-y-1.5">
                  {available.map(p => {
                    const meta = policyMeta(p)
                    const disabled = p === 'ppo' && ppoBlocked
                    return (
                      <button key={p} type="button" disabled={disabled} onClick={() => togglePolicy(p)}
                        className={`flex w-full items-center gap-2 rounded-md border px-2.5 py-1.5 text-left text-sm transition-colors disabled:opacity-40 ${selected.includes(p) ? 'border-foreground/30 bg-muted' : 'border-input'}`}>
                        <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: meta.color, opacity: selected.includes(p) ? 1 : 0.3 }} />
                        <span className="flex-1">{meta.label}</span>
                        {disabled && <span className="text-xs text-muted-foreground">26 slices</span>}
                        {selected.includes(p) && !disabled && <span className="text-xs" style={{ color: meta.color }}>✓</span>}
                      </button>
                    )
                  })}
                </div>
              </div>

              <Button className="w-full" onClick={run} disabled={loading || running || !selected.length}>
                {loading ? 'Setting up…' : running ? 'Running…' : scenario ? 'Re-run' : 'Run sandbox'}
              </Button>
            </CardContent>
          </Card>

          <div>
            {error && <Card className="mb-4 border-[#c0553a]/50"><CardContent className="pt-6 text-sm text-[#c0553a]">{error}</CardContent></Card>}

            {scenario && (
              <div className="mb-4 flex flex-wrap items-center gap-2">
                <Badge variant="secondary">{scenario.ticker}</Badge>
                <Badge variant="secondary">{scenario.date}</Badge>
                <Badge variant="secondary" className="capitalize">{scenario.side} {scenario.total_shares.toLocaleString()} sh</Badge>
                {scenario.order_adv_pct != null && <Badge variant="secondary">{scenario.order_adv_pct}% ADV</Badge>}
                <Badge variant="secondary">{formatUsd(scenario.notional_usd)}</Badge>
                <Badge variant={scenario.data_source === 'real_minute' ? 'default' : 'outline'}>
                  {scenario.data_source === 'real_minute' ? 'Real minute bars' : 'Synthetic path'}
                </Badge>
                {running ? <span className="flex items-center gap-1.5 text-xs font-medium text-[#c0553a]"><span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#c0553a]" />slice {cursor}/{scenario.n_slices}</span>
                  : done ? <span className="text-xs font-medium text-[#cc9166]">✓ Complete</span> : null}
              </div>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Price path &amp; execution</CardTitle>
                {scenario && <CardDescription>Open ${scenario.open_price.toFixed(2)} → Close ${scenario.close_price.toFixed(2)} · ADV {Math.round(scenario.adv).toLocaleString()}</CardDescription>}
              </CardHeader>
              <CardContent>
                {scenario ? <MultiExecutionChart scenario={scenario} policies={scenario.policies} currentSlice={cursor} />
                  : <div className="flex h-80 flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
                      <p>Configure a scenario and hit Run sandbox</p>
                      <p className="text-xs">Overlay PPO, VWAP-match and naive TWAP on the same day</p>
                    </div>}
              </CardContent>
            </Card>

            {scenario && (
              <Card className="mt-4">
                <CardContent className="pt-6">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Policy</TableHead>
                        <TableHead className="text-right">Slippage vs VWAP</TableHead>
                        <TableHead className="text-right">vs VWAP ($)</TableHead>
                        <TableHead className="text-right">Avg fill price</TableHead>
                        <TableHead className="text-right">Filled</TableHead>
                        <TableHead className="text-right">Reward</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {rows.map(r => (
                        <TableRow key={r.policy}>
                          <TableCell>
                            <span className="flex items-center gap-2">
                              <span className="inline-block h-3 w-3 rounded-full" style={{ backgroundColor: r.meta.color }} />
                              {r.meta.label}
                              {best && best.policy === r.policy && done && <Badge className="ml-1" style={{ backgroundColor: r.meta.color, color: 'white' }}>Best</Badge>}
                            </span>
                          </TableCell>
                          <TableCell className="text-right font-medium tabular-nums" style={{ color: r.meta.color }}>{formatBps(r.liveSlip)}</TableCell>
                          <TableCell className="text-right tabular-nums">{r.liveUsd == null ? '—' : formatUsd(r.liveUsd)}</TableCell>
                          <TableCell className="text-right tabular-nums">{done ? `$${r.agent_vwap?.toFixed(2)}` : '—'}</TableCell>
                          <TableCell className="text-right tabular-nums">{done ? `${(r.filled_fraction * 100).toFixed(0)}%` : '—'}</TableCell>
                          <TableCell className="text-right tabular-nums">{done ? r.final_reward?.toFixed(3) : '—'}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
