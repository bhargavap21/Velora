import { useEffect, useRef, useState } from 'react'
import Navbar from '../components/Navbar'
import ExecutionChart, { currentSlippageBps } from '../components/ExecutionChart'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import episode from '../data/sample_episode.json'

const TICK_MS = 350

export default function ExecutionLive() {
  const [slice, setSlice] = useState(0)
  const [playing, setPlaying] = useState(true)
  const intervalRef = useRef(null)

  useEffect(() => {
    if (!playing) return
    intervalRef.current = setInterval(() => {
      setSlice(s => {
        if (s >= episode.n_slices) {
          setPlaying(false)
          return s
        }
        return s + 1
      })
    }, TICK_MS)
    return () => clearInterval(intervalRef.current)
  }, [playing])

  const filledFraction = slice / episode.n_slices
  const slippageBps = currentSlippageBps(episode, slice)
  const done = slice >= episode.n_slices

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="mx-auto max-w-5xl px-4 py-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold">Live Execution</h2>
            <p className="mt-0.5 text-sm text-muted-foreground">
              {episode.schedule_label} — replaying a real simulator episode
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary">{episode.ticker}</Badge>
            <Badge variant="secondary">{episode.side} {episode.total_shares.toLocaleString()} sh</Badge>
            {done ? (
              <span className="text-xs font-medium text-green-500">✓ Complete</span>
            ) : (
              <span className="animate-pulse text-xs text-muted-foreground">Running…</span>
            )}
          </div>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Price path &amp; execution</CardTitle>
          </CardHeader>
          <CardContent>
            <ExecutionChart episode={episode} currentSlice={slice} />
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
              <p className="mt-1 text-2xl font-bold">{slice} / {episode.n_slices}</p>
            </CardContent>
          </Card>
        </div>

        <div className="mt-4 flex gap-2">
          {done ? (
            <Button onClick={() => { setSlice(0); setPlaying(true) }}>Replay</Button>
          ) : (
            <Button variant="outline" onClick={() => setPlaying(p => !p)}>
              {playing ? 'Pause' : 'Resume'}
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
