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
import { apiUrl } from '@/lib/api'

const inputClass =
  'h-9 w-full rounded-md border border-input bg-background px-3 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

const WIN_COLOR = '#cc9166'
const LOSS_COLOR = '#c0553a'
const MAX_EPISODES = 8

export default function RftLive() {
  const [form, setForm] = useState({ ticker: 'AAPL', side: 'buy', adv_pct: 8, n_episodes: 4 })
  const [meta, setMeta] = useState(null)
  const [episodes, setEpisodes] = useState([])
  const [summary, setSummary] = useState(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState(null)
  const esRef = useRef(null)

  const close = useCallback(() => {
    if (esRef.current) { esRef.current.close(); esRef.current = null }
  }, [])
  useEffect(() => close, [close])

  const run = useCallback(() => {
    close()
    setError(null)
    setMeta(null)
    setSummary(null)
    setEpisodes([])
    setRunning(true)

    const params = new URLSearchParams({
      ticker: form.ticker, side: form.side,
      adv_pct: String(form.adv_pct), n_episodes: String(form.n_episodes),
    })
    const es = new EventSource(apiUrl(`/api/rft/stream?${params.toString()}`))
    esRef.current = es

    es.addEventListener('meta', e => setMeta(JSON.parse(e.data)))
    es.addEventListener('episode', e => {
      const d = JSON.parse(e.data)
      setEpisodes(prev => [...prev, d])
    })
    es.addEventListener('summary', e => {
      setSummary(JSON.parse(e.data))
      setRunning(false)
      close()
    })
    es.addEventListener('error', e => {
      if (e.data) { try { setError(JSON.parse(e.data).detail) } catch { setError('Stream error') } }
      else setError('Connection lost')
      setRunning(false)
      close()
    })
  }, [form, close])

  const ok = episodes.filter(e => !e.error)
  const progress = meta ? (episodes.length / meta.n_episodes) * 100 : 0
  const liveWins = ok.filter(e => e.delta > 0).length
  const liveMean = ok.length ? ok.reduce((s, e) => s + e.delta, 0) / ok.length : 0
  const significant = summary && Math.abs(summary.t_stat) >= 1.96

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6">
          <h2 className="text-[28px] font-light tracking-[-0.02em] text-foreground">
            RFT:{' '}
            <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 500, color: '#cc9166' }}>
              live, in progress.
            </span>
          </h2>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            A real, fine-tuned LLM (forked Qwen3 8B, trained via HUD) graded against its own un-trained, zero-shot
            self — paired by seed so both see the identical market path. Unlike the policies on the Proof page,{' '}
            <span className="font-medium text-foreground">each episode here is a live LLM call</span>, not an
            instant backend replay — expect roughly 30-90 seconds per episode, and capped at {MAX_EPISODES} per run
            to keep latency and cost bounded.
          </p>
        </div>

        <Card className="mb-6">
          <CardContent className="grid grid-cols-2 gap-4 pt-6 md:grid-cols-5">
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Ticker</label>
              <input className={inputClass} value={form.ticker}
                onChange={e => setForm({ ...form, ticker: e.target.value.toUpperCase() })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Side</label>
              <select className={inputClass} value={form.side} onChange={e => setForm({ ...form, side: e.target.value })}>
                <option value="buy">Buy</option><option value="sell">Sell</option>
              </select>
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground">Size (% ADV)</label>
              <input type="number" className={inputClass} min={1} max={20} step={1}
                value={form.adv_pct} onChange={e => setForm({ ...form, adv_pct: Number(e.target.value) })} />
            </div>
            <div>
              <label className="mb-1 block text-xs font-medium text-muted-foreground"># Episodes (max {MAX_EPISODES})</label>
              <input type="number" className={inputClass} min={1} max={MAX_EPISODES} step={1}
                value={form.n_episodes}
                onChange={e => setForm({ ...form, n_episodes: Math.min(MAX_EPISODES, Math.max(1, Number(e.target.value))) })} />
            </div>
            <div className="flex items-end">
              <Button className="w-full" onClick={run} disabled={running}>
                {running ? 'Running live…' : 'Run live'}
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
              <Badge variant="secondary">{meta.adv_pct}% ADV</Badge>
              <Badge variant="outline">RFT vs {meta.baseline}</Badge>
              {running && (
                <span className="flex items-center gap-1.5 text-xs font-medium text-[#c0553a]">
                  <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#c0553a]" />
                  episode {episodes.length}/{meta.n_episodes} — each one is a live LLM call, this can take a while
                </span>
              )}
            </div>

            {running && <Progress className="mb-4" value={progress} />}

            <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-3">
              <Card>
                <CardContent className="pt-6">
                  <p className="text-xs text-muted-foreground">Mean delta (rft − base)</p>
                  <p className="mt-1 text-3xl font-extrabold tabular-nums" style={{ color: (summary?.mean_delta ?? liveMean) >= 0 ? WIN_COLOR : LOSS_COLOR }}>
                    <AnimatedNumber value={summary?.mean_delta ?? liveMean} format={n => `${n >= 0 ? '+' : ''}${n.toFixed(3)}`} />
                  </p>
                  <p className="text-xs text-muted-foreground">of {summary?.n ?? ok.length} completed episodes</p>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-xs text-muted-foreground">Win / tie / loss</p>
                  <p className="mt-1 text-3xl font-extrabold tabular-nums">
                    {summary ? `${summary.wins}/${summary.ties}/${summary.losses}` : `${liveWins}/—/—`}
                  </p>
                  <p className="text-xs text-muted-foreground">paired by seed</p>
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
                      ? (significant ? <span style={{ color: WIN_COLOR }}>significant (|t| ≥ 1.96)</span>
                        : <span className="text-muted-foreground">not significant at this n — small-sample noise</span>)
                      : <span className="text-muted-foreground">running…</span>}
                  </p>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Per-episode reward delta</CardTitle>
                <CardDescription>RFT minus base, this run only — bars right of zero (gold) are episodes RFT won.</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={episodes.filter(e => !e.error).map((e, i) => ({ ...e, label: `ep ${i + 1}` }))}
                    margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="label" stroke="var(--muted-foreground)" tick={{ fontSize: 11 }} />
                    <YAxis stroke="var(--muted-foreground)" tick={{ fontSize: 11 }}
                      label={{ value: 'reward delta', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--muted-foreground)' }} />
                    <Tooltip
                      contentStyle={{ backgroundColor: 'var(--popover)', border: '1px solid var(--border)', borderRadius: 8, fontSize: 12 }}
                      formatter={(v, _n, p) => [`${v >= 0 ? '+' : ''}${v.toFixed(3)}`, `base=${p.payload.base_reward.toFixed(2)} rft=${p.payload.rft_reward.toFixed(2)}`]}
                    />
                    <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeDasharray="4 3" />
                    <Bar dataKey="delta" radius={[3, 3, 0, 0]}>
                      {episodes.filter(e => !e.error).map((e, i) => (
                        <Cell key={i} fill={e.delta > 0 ? WIN_COLOR : e.delta < 0 ? LOSS_COLOR : 'var(--muted-foreground)'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </>
        )}

        {!meta && !error && (
          <Card>
            <CardContent className="flex h-72 flex-col items-center justify-center gap-2 text-center text-sm text-muted-foreground">
              <p>Configure a run and hit <span className="font-medium text-foreground">Run live</span></p>
              <p className="text-xs">This calls the real trained model live through HUD's gateway — not a replay.</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
