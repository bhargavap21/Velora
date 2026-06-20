import { useEffect, useRef, useState } from 'react'
import Navbar from '../components/Navbar'
import ExecutionChart, { currentSlippageBps } from '../components/ExecutionChart'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'

const POLICY_LABELS = {
  twap: 'TWAP',
  ppo: 'PPO',
  llm: 'Claude LLM',
  fireworks: 'GPT-OSS (Fireworks)',
}

export default function ExecutionLive() {
  const [policy, setPolicy] = useState('twap')
  const [availablePolicies, setAvailablePolicies] = useState(['twap'])
  const [episode, setEpisode] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [slice, setSlice] = useState(0)
  const [done, setDone] = useState(false)
  const esRef = useRef(null)

  useEffect(() => {
    fetch('/api/policies')
      .then(res => {
        if (!res.ok) throw new Error('Failed to load policies')
        return res.json()
      })
      .then(data => {
        if (Array.isArray(data.available)) setAvailablePolicies(data.available)
      })
      .catch(() => {})
  }, [])

  const closeStream = () => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
  }

  const runEpisode = (selectedPolicy) => {
    closeStream()
    setLoading(true)
    setError(null)
    setEpisode(null)
    setSlice(0)
    setDone(false)

    const es = new EventSource(`/api/episode/stream?policy=${selectedPolicy}`)
    esRef.current = es

    es.addEventListener('meta', (e) => {
      const meta = JSON.parse(e.data)
      setEpisode({ ...meta, exec_prices: [], exec_quantities: [] })
      setSlice(0)
      setLoading(false)
    })

    es.addEventListener('slice', (e) => {
      const { i, exec_price, exec_quantity } = JSON.parse(e.data)
      setEpisode(prev => {
        if (!prev) return prev
        const exec_prices = [...prev.exec_prices]
        const exec_quantities = [...prev.exec_quantities]
        exec_prices[i] = exec_price
        exec_quantities[i] = exec_quantity
        return { ...prev, exec_prices, exec_quantities }
      })
      setSlice(i + 1)
    })

    es.addEventListener('done', (e) => {
      const data = JSON.parse(e.data)
      setEpisode(prev => prev ? {
        ...prev,
        exec_prices: data.exec_prices,
        exec_quantities: data.exec_quantities,
        final_reward: data.final_reward,
        filled_fraction: data.filled_fraction,
      } : prev)
      setDone(true)
      closeStream()
    })

    // Fires for both our custom server `error` event (has data) and native
    // connection errors (no data). EventSource would otherwise auto-reconnect, so we
    // always close the stream here.
    es.addEventListener('error', (e) => {
      if (e.data) {
        try {
          setError(JSON.parse(e.data).detail || 'Stream error')
        } catch {
          setError('Stream error')
        }
      } else if (es.readyState === EventSource.CLOSED) {
        setError('Connection to the live stream was lost')
      }
      setLoading(false)
      closeStream()
    })
  }

  useEffect(() => {
    runEpisode(policy)
    return closeStream
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const filledFraction = episode ? slice / episode.n_slices : 0
  const slippageBps = episode ? currentSlippageBps(episode, slice) : null
  const streaming = episode && !done && !loading

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">Live Execution</h2>
            <p className="mt-0.5 text-sm text-muted-foreground">
              {episode ? `${episode.schedule_label} — streamed live from the backend simulator` : 'Connecting to simulator…'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <select
              className="h-8 rounded-md border border-input bg-background px-2 text-xs"
              value={policy}
              disabled={loading || streaming}
              onChange={e => { setPolicy(e.target.value); runEpisode(e.target.value) }}
            >
              {availablePolicies.map(p => (
                <option key={p} value={p}>{POLICY_LABELS[p] ?? p.toUpperCase()}</option>
              ))}
            </select>
            {episode && <Badge variant="secondary">{episode.ticker}</Badge>}
            {episode && <Badge variant="secondary">{episode.side} {episode.total_shares.toLocaleString()} sh</Badge>}
            {loading ? (
              <span className="animate-pulse text-xs text-muted-foreground">Connecting…</span>
            ) : done ? (
              <span className="text-xs font-medium text-green-500">✓ Complete</span>
            ) : (
              <span className="flex items-center gap-1.5 text-xs font-medium text-red-500">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-red-500" />
                LIVE
              </span>
            )}
          </div>
        </div>

        {error && (
          <Card className="mb-4 border-red-500/50">
            <CardContent className="pt-6 text-sm text-red-500">{error}</CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Price path &amp; execution</CardTitle>
          </CardHeader>
          <CardContent>
            {episode ? <ExecutionChart episode={episode} currentSlice={slice} /> : (
              <div className="flex h-80 items-center justify-center text-sm text-muted-foreground">Connecting…</div>
            )}
          </CardContent>
        </Card>

        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
          <Card>
            <CardContent className="pt-6">
              <div className="mb-1 flex justify-between text-xs text-muted-foreground">
                <span>Filled</span>
                <span>{(filledFraction * 100).toFixed(0)}%</span>
              </div>
              <Progress value={filledFraction * 100} />
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <p className="text-xs text-muted-foreground">Slippage vs. VWAP</p>
              <p className={`mt-1 text-2xl font-bold ${slippageBps == null ? '' : slippageBps >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {slippageBps == null ? '—' : `${slippageBps >= 0 ? '+' : ''}${slippageBps.toFixed(1)} bps`}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="pt-6">
              <p className="text-xs text-muted-foreground">Slice</p>
              <p className="mt-1 text-2xl font-bold">{slice} / {episode ? episode.n_slices : '—'}</p>
            </CardContent>
          </Card>
        </div>

        <div className="mt-4 flex gap-2">
          <Button onClick={() => runEpisode(policy)} disabled={loading || streaming}>
            {done ? 'New episode' : streaming ? 'Streaming…' : 'Run episode'}
          </Button>
        </div>

        {episode?.llm_reasoning && (
          <Card className="mt-4">
            <CardContent className="pt-4">
              <p className="mb-1 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                {policy === 'fireworks' ? 'GPT-OSS reasoning' : 'Claude reasoning'}
              </p>
              <p className="text-sm leading-relaxed">{episode.llm_reasoning}</p>
              {episode.llm_primitive && (
                <p className="mt-2 text-xs text-muted-foreground">
                  Schedule primitive: <span className="font-medium text-foreground">{episode.llm_primitive}</span>
                  {episode.llm_pause_enabled && (
                    <span className="ml-2">· pause-on-adverse-move enabled ({episode.llm_pause_threshold_bps} bps)</span>
                  )}
                </p>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
