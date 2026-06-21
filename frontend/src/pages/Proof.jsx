import { useCallback, useEffect, useRef, useState } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import Navbar from '../components/Navbar'
import AnimatedNumber from '../components/AnimatedNumber'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { policyMeta, formatUsd, formatBps, histogram } from '@/lib/execution'
import { apiUrl } from '@/lib/api'
import { rftHoldout } from '@/data/benchmarks'

const inputClass =
  'h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

const WIN_COLOR = '#cc9166'  // ember gold — days the agent beat the baseline
const LOSS_COLOR = '#c0553a' // warm red — days it lost

export default function Proof() {
  const [cfg, setCfg] = useState(null)
  const [policies, setPolicies] = useState([])
  const [form, setForm] = useState(null)
  const [meta, setMeta] = useState(null)
  const [advantages, setAdvantages] = useState([])
  const [summary, setSummary] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const esRef = useRef(null)

  useEffect(() => {
    Promise.all([
      fetch(apiUrl('/api/config')).then(r => r.json()),
      fetch(apiUrl('/api/policies')).then(r => r.json()),
    ]).then(([config, pol]) => {
      setCfg(config)
      const avail = (pol.available ?? []).filter(p => ['naive_twap', 'twap', 'ppo'].includes(p))
      setPolicies(avail)
      // Defaults reproduce the landing-page card exactly (PPO vs the strong VWAP-match
      // baseline, AAPL buy, 8% ADV, 60 held-out days). Switch the baseline to the naive
      // equal-time TWAP to see the larger (but easier) edge.
      setForm({
        policy: avail.includes('ppo') ? 'ppo' : 'twap',
        baseline: avail.includes('twap') ? 'twap' : 'naive_twap',
        ticker: 'AAPL',
        side: 'buy',
        adv_pct: config.defaults.adv_pct ?? 8,
        n_episodes: 60,
      })
    }).catch(() => setError('Failed to load configuration'))
  }, [])

  const close = useCallback(() => {
    if (esRef.current) { esRef.current.close(); esRef.current = null }
  }, [])
  useEffect(() => close, [close])

  const run = useCallback(() => {
    if (!form) return
    close()
    setError(null)
    setMeta(null)
    setSummary(null)
    setAdvantages([])
    setRunning(true)

    const params = new URLSearchParams({
      policy: form.policy, baseline: form.baseline, ticker: form.ticker,
      side: form.side, adv_pct: String(form.adv_pct), n_episodes: String(form.n_episodes),
    })
    const es = new EventSource(apiUrl(`/api/eval/stream?${params.toString()}`))
    esRef.current = es

    es.addEventListener('meta', e => setMeta(JSON.parse(e.data)))
    es.addEventListener('episode', e => {
      const d = JSON.parse(e.data)
      setAdvantages(prev => [...prev, d.advantage_bps])
    })
    es.addEventListener('summary', e => {
      setSummary(JSON.parse(e.data))
      setRunning(false)
      close()
    })
    es.addEventListener('error', e => {
      if (e.data) { try { setError(JSON.parse(e.data).detail) } catch { setError('Stream error') } }
      else if (es.readyState === EventSource.CLOSED && !summary) setError('Connection lost')
      setRunning(false)
      close()
    })
  }, [form, close, summary])

  if (!cfg || !form) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Navbar />
        <div className="mx-auto max-w-6xl px-4 py-16 text-center text-sm text-muted-foreground">Loading…</div>
      </div>
    )
  }

  const progress = meta ? (advantages.length / meta.n_episodes) * 100 : 0
  const liveWins = advantages.filter(a => a > 0).length
  const liveWinRate = advantages.length ? liveWins / advantages.length : 0
  const liveMean = advantages.length ? advantages.reduce((s, a) => s + a, 0) / advantages.length : 0
  const bins = histogram(advantages, 23)
  const pMeta = policyMeta(form.policy)
  const bMeta = policyMeta(form.baseline)
  const significant = summary && Math.abs(summary.t_stat) >= 2

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6">
          <h2 className="text-[28px] font-light tracking-[-0.02em] text-foreground">
            Proof:{' '}
            <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 500, color: '#cc9166' }}>held-out evaluation.</span>
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            One showdown can get lucky. This runs the agent against the baseline across many
            {' '}<span className="font-medium text-foreground">held-out trading days the model never saw in training</span>,
            on the same paired price paths, and reports the distribution of outcomes — the actual evidence.
          </p>
        </div>

        <Card className="mb-6">
          <CardContent className="grid grid-cols-2 gap-4 pt-6 md:grid-cols-7">
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Agent</label>
              <select className={inputClass} value={form.policy} onChange={e => setForm({ ...form, policy: e.target.value })}>
                {policies.map(p => <option key={p} value={p}>{policyMeta(p).short}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Baseline</label>
              <select className={inputClass} value={form.baseline} onChange={e => setForm({ ...form, baseline: e.target.value })}>
                {policies.filter(p => p !== 'ppo').map(p => <option key={p} value={p}>{policyMeta(p).short}</option>)}
              </select>
            </div>
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
              <select className={inputClass} value={form.side} onChange={e => setForm({ ...form, side: e.target.value })}>
                <option value="buy">Buy</option><option value="sell">Sell</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Size (% ADV)</label>
              <input type="number" className={inputClass} min={cfg.constraints.adv_pct.min}
                max={cfg.constraints.adv_pct.max} step={cfg.constraints.adv_pct.step}
                value={form.adv_pct} onChange={e => setForm({ ...form, adv_pct: Number(e.target.value) })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground"># Days</label>
              <input type="number" className={inputClass} min={5} max={300} step={5}
                value={form.n_episodes} onChange={e => setForm({ ...form, n_episodes: Number(e.target.value) })} />
            </div>
            <div className="flex items-end">
              <Button className="w-full" onClick={run} disabled={running}>
                {running ? 'Evaluating…' : 'Run evaluation'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {error && <Card className="mb-4 border-[#c0553a]/50"><CardContent className="pt-6 text-sm text-[#c0553a]">{error}</CardContent></Card>}

        {meta && (
          <>
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <Badge variant="secondary">{meta.ticker}</Badge>
              <Badge variant="secondary" className="capitalize">{meta.side}</Badge>
              {meta.adv_pct && <Badge variant="secondary">{meta.adv_pct}% ADV</Badge>}
              <Badge variant="outline">{meta.split}</Badge>
              {running && (
                <span className="flex items-center gap-1.5 text-xs font-medium text-[#c0553a]">
                  <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#c0553a]" />
                  {advantages.length}/{meta.n_episodes} days
                </span>
              )}
            </div>

            {running && <Progress className="mb-4" value={progress} />}

            {/* Headline stat cards */}
            <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-4">
              <Card>
                <CardContent className="pt-6">
                  <p className="text-xs text-muted-foreground">Win rate vs baseline</p>
                  <p className="mt-1 text-3xl font-extrabold tabular-nums" style={{ color: WIN_COLOR }}>
                    <AnimatedNumber value={(summary?.win_rate ?? liveWinRate) * 100} format={n => `${n.toFixed(0)}%`} />
                  </p>
                  <p className="text-xs text-muted-foreground">of {summary?.n_episodes ?? advantages.length} held-out days</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-xs text-muted-foreground">Mean advantage</p>
                  <p className="mt-1 text-3xl font-extrabold tabular-nums" style={{ color: WIN_COLOR }}>
                    <AnimatedNumber value={summary?.mean_advantage_bps ?? liveMean} format={n => formatBps(n)} />
                  </p>
                  <p className="text-xs text-muted-foreground">per order vs {bMeta.short}</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-xs text-muted-foreground">Avg $ saved / order</p>
                  <p className="mt-1 text-3xl font-extrabold tabular-nums">
                    {summary ? <AnimatedNumber value={summary.usd_saved_per_order} format={n => formatUsd(n)} /> : '—'}
                  </p>
                  <p className="text-xs text-muted-foreground">on ~{summary ? formatUsd(summary.mean_notional_usd) : '—'} orders</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-xs text-muted-foreground">Significance (t-stat)</p>
                  <p className="mt-1 text-3xl font-extrabold tabular-nums">
                    {summary ? <AnimatedNumber value={summary.t_stat} format={n => n.toFixed(2)} /> : '—'}
                  </p>
                  <p className="text-xs">
                    {summary
                      ? (significant ? <span style={{ color: WIN_COLOR }}>significant (|t| ≥ 2)</span>
                        : <span className="text-muted-foreground">not yet significant</span>)
                      : <span className="text-muted-foreground">running…</span>}
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Distribution */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Per-day advantage distribution</CardTitle>
                <CardDescription>
                  {pMeta.short} minus {bMeta.short} slippage, in bps, on each held-out day.
                  Bars right of zero (green) are days the agent won.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={bins} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="x" stroke="var(--muted-foreground)" tick={{ fontSize: 11 }}
                      tickFormatter={v => v.toFixed(0)}
                      label={{ value: 'Advantage (bps)', position: 'insideBottom', offset: -2, fontSize: 11, fill: 'var(--muted-foreground)' }} />
                    <YAxis stroke="var(--muted-foreground)" tick={{ fontSize: 11 }} allowDecimals={false}
                      label={{ value: 'Days', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--muted-foreground)' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                      formatter={v => [`${v} days`, 'Count']}
                      labelFormatter={l => `${Number(l).toFixed(1)} bps`} />
                    <ReferenceLine x={0} stroke="var(--muted-foreground)" strokeDasharray="4 3" />
                    <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                      {bins.map((b, i) => <Cell key={i} fill={b.x >= 0 ? WIN_COLOR : LOSS_COLOR} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {summary && (
              <Card className="mt-4">
                <CardContent className="grid grid-cols-1 gap-4 pt-6 md:grid-cols-3">
                  <div>
                    <p className="text-xs text-muted-foreground">{pMeta.short} mean slippage</p>
                    <p className="mt-0.5 text-lg font-bold tabular-nums" style={{ color: pMeta.color }}>
                      {formatBps(summary.policy_mean_bps)} <span className="text-xs font-normal text-muted-foreground">± {summary.policy_std_bps.toFixed(0)}</span>
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">{bMeta.short} mean slippage</p>
                    <p className="mt-0.5 text-lg font-bold tabular-nums" style={{ color: bMeta.color }}>
                      {formatBps(summary.baseline_mean_bps)} <span className="text-xs font-normal text-muted-foreground">± {summary.baseline_std_bps.toFixed(0)}</span>
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Median advantage</p>
                    <p className="mt-0.5 text-lg font-bold tabular-nums" style={{ color: WIN_COLOR }}>{formatBps(summary.median_advantage_bps)}</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </>
        )}

        {!meta && !error && (
          <Card>
            <CardContent className="flex h-72 flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
              <p>Configure an evaluation and hit <span className="font-medium text-foreground">Run evaluation</span></p>
              <p className="text-xs">Each day is run with both policies on the identical price path — a paired comparison</p>
            </CardContent>
          </Card>
        )}

        <RftHoldoutCard />
      </div>
    </div>
  )
}

// Static evidence panel for the in-progress RFT effort (GitHub issue #15) -- the actual n=12
// paired result already collected, honestly, including the losses. A live, re-runnable version
// exists at /rft (real LLM calls per episode, capped low); this snapshot stays here so the
// Proof page isn't blocking on that slower, costlier path by default.
function RftHoldoutCard() {
  const rows = rftHoldout.scenarios.map((s, i) => ({ ...s, idx: i + 1, delta: s.rft - s.base }))
  return (
    <Card className="mt-4 border-dashed">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">RFT held-out result (in progress)</CardTitle>
            <CardDescription>
              {rftHoldout.model} vs. zero-shot Qwen3 8B, n={rftHoldout.nEpisodes} fresh scenarios, paired by seed —
              the actual data behind the snapshot below. Want a fresh run instead of this fixed one? See{' '}
              <a href="/rft" className="font-medium underline">the live RFT page</a> — it makes real LLM calls per
              episode (~30-90s each), capped low, so it's a separate page rather than a button here.
            </CardDescription>
          </div>
          <Badge variant="outline" className="shrink-0">Not yet significant</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="ticker" stroke="var(--muted-foreground)" tick={{ fontSize: 11 }} />
            <YAxis stroke="var(--muted-foreground)" tick={{ fontSize: 11 }}
              label={{ value: 'reward delta (rft - base)', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--muted-foreground)' }} />
            <Tooltip
              contentStyle={{ backgroundColor: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
              formatter={(v, _n, p) => [`${v >= 0 ? '+' : ''}${v.toFixed(3)}`, `${p.payload.side} · base=${p.payload.base.toFixed(2)} rft=${p.payload.rft.toFixed(2)}`]}
            />
            <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeDasharray="4 3" />
            <Bar dataKey="delta" radius={[3, 3, 0, 0]}>
              {rows.map((r, i) => <Cell key={i} fill={r.delta > 0 ? WIN_COLOR : r.delta < 0 ? LOSS_COLOR : 'var(--muted-foreground)'} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="mt-4 grid grid-cols-1 gap-4 border-t pt-4 md:grid-cols-4">
          <div>
            <p className="text-xs text-muted-foreground">Mean delta</p>
            <p className="mt-0.5 text-lg font-bold tabular-nums" style={{ color: WIN_COLOR }}>+{rftHoldout.meanDelta.toFixed(3)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Win / tie / loss</p>
            <p className="mt-0.5 text-lg font-bold tabular-nums">{rftHoldout.wins} / {rftHoldout.ties} / {rftHoldout.losses}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">t-statistic</p>
            <p className="mt-0.5 text-lg font-bold tabular-nums">{rftHoldout.tStat.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Training</p>
            <p className="mt-0.5 text-lg font-bold tabular-nums">{rftHoldout.trainingStepsApplied} steps</p>
          </div>
        </div>
        <p className="mt-4 text-xs text-muted-foreground">
          Honest read: small positive lean, far from statistically significant at this n. What's real is the
          pipeline itself — collect, train, promote checkpoint, evaluate — now runs end to end. See{' '}
          <a href="/pitch" className="font-medium underline">the full writeup</a> for the training curve and the bug
          we found and fixed to get here.
        </p>
      </CardContent>
    </Card>
  )
}
