import { useEffect, useRef, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { openStream } from '../api'

const SEED_LABELS = {
  tsla_rsi:     'TSLA RSI Momentum',
  spy_ema_rsi:  'SPY EMA + RSI',
  nvda_sma_rsi: 'NVDA SMA + RSI',
  aapl_sma_rsi: 'AAPL SMA + RSI',
  tsla_sma_rsi: 'TSLA SMA + RSI',
}

function empty() {
  return { turns: [], finalSharpe: null, initialSharpe: null, totalReward: null, turnsTotal: null, ticker: null, complete: false }
}

function SharpeMeter({ initial, current }) {
  const max = Math.max(2, (current || initial || 0) * 1.5)
  const pct = Math.min(100, ((current || initial || 0) / max) * 100)
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs text-muted-foreground">
        <span>Sharpe</span>
        <span>
          {initial != null ? initial.toFixed(2) : '—'} → {current != null ? current.toFixed(2) : '—'}
        </span>
      </div>
      <Progress value={pct} />
    </div>
  )
}

function EpisodeCard({ seedName, data }) {
  const label = SEED_LABELS[seedName] || seedName
  const latestSharpe = data.turns.length > 0
    ? data.turns[data.turns.length - 1].sharpe_after
    : data.initialSharpe
  const turnsRef = useRef(null)

  useEffect(() => {
    if (turnsRef.current) {
      turnsRef.current.scrollTop = turnsRef.current.scrollHeight
    }
  }, [data.turns.length])

  return (
    <Card className={data.complete ? 'ring-1 ring-green-700/50' : ''}>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle>{label}</CardTitle>
            {data.ticker && <Badge variant="secondary">{data.ticker}</Badge>}
          </div>
          {data.complete ? (
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium text-green-500">✓ Complete</span>
              {data.totalReward != null && (
                <span className="text-xs text-muted-foreground">reward {data.totalReward.toFixed(2)}</span>
              )}
            </div>
          ) : (
            <span className="animate-pulse text-xs text-muted-foreground">Running…</span>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <SharpeMeter initial={data.initialSharpe} current={latestSharpe} />

        <div ref={turnsRef} className="max-h-48 space-y-2 overflow-y-auto">
          {data.turns.length === 0 ? (
            <p className="text-xs italic text-muted-foreground">Waiting for first turn…</p>
          ) : (
            data.turns.map((t, i) => (
              <div key={i} className="border-l-2 border-border pl-2 text-xs">
                <div className="flex items-center gap-2 text-foreground/80">
                  <span className="text-muted-foreground">Turn {t.turn}</span>
                  <span className="font-medium">{t.mutation_type}</span>
                  <span className="ml-auto text-primary">reward: {t.reward.toFixed(2)}</span>
                </div>
                {t.reasoning && (
                  <p className="mt-0.5 leading-snug text-muted-foreground">{t.reasoning}</p>
                )}
              </div>
            ))
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default function Live() {
  const { runId } = useParams()
  const navigate = useNavigate()
  const [episodes, setEpisodes] = useState({})
  const [disconnected, setDisconnected] = useState(false)
  const esRef = useRef(null)
  const episodesRef = useRef({})

  useEffect(() => {
    const es = openStream(runId)
    esRef.current = es

    es.onmessage = (e) => {
      try {
        const event = JSON.parse(e.data)
        setEpisodes(prev => {
          const next = { ...prev }
          const seed = event.seed_name
          if (!next[seed]) next[seed] = empty()
          const ep = { ...next[seed] }

          if (event.type === 'turn') {
            ep.turns = [...ep.turns, event]
            if (ep.initialSharpe == null) ep.initialSharpe = event.sharpe_before
          } else if (event.type === 'episode_complete') {
            ep.complete = true
            ep.finalSharpe = event.final_sharpe
            ep.initialSharpe = event.initial_sharpe
            ep.totalReward = event.total_reward
            ep.turnsTotal = event.turns_taken
            ep.ticker = event.ticker
          }
          next[seed] = ep
          episodesRef.current = next
          return next
        })
      } catch {}
    }

    es.onerror = () => {
      // EventSource fires onerror both on a real connection drop and on a clean
      // server-side close (the backend's stream generator returns once the run
      // is done) — the spec gives no way to distinguish them. If every episode
      // we've seen is already complete, this is the expected close, not a failure.
      const seeds = Object.values(episodesRef.current)
      const allComplete = seeds.length > 0 && seeds.every(ep => ep.complete)
      if (!allComplete) setDisconnected(true)
    }

    return () => es.close()
  }, [runId])

  const seeds = Object.keys(episodes)
  const completed = seeds.filter(s => episodes[s].complete).length
  const total = seeds.length
  const allDone = total > 0 && completed === total

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      {disconnected && (
        <Alert variant="destructive" className="rounded-none border-x-0 text-center">
          <AlertDescription className="mx-auto">
            Stream disconnected.{' '}
            <button onClick={() => window.location.reload()} className="underline">
              Reconnect
            </button>
          </AlertDescription>
        </Alert>
      )}
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">Live Episodes</h2>
            <p className="mt-0.5 text-sm text-muted-foreground">{completed} / {total || '…'} complete</p>
          </div>
          {allDone && (
            <Button onClick={() => navigate(`/results/${runId}`)}>
              View Results →
            </Button>
          )}
        </div>

        {seeds.length === 0 ? (
          <div className="py-24 text-center text-muted-foreground">Waiting for episodes to start…</div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {seeds.map(s => (
              <EpisodeCard key={s} seedName={s} data={episodes[s]} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
