import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { hudEval, ppoHoldout, headlineStats } from '../data/benchmarks'

export default function Home() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />

      <div className="mx-auto max-w-3xl px-4 py-24 text-center">
        <Badge variant="secondary" className="mb-4">HUD x YC Hackathon</Badge>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">Velora</h1>
        <p className="mt-4 text-lg text-muted-foreground">
          An RL environment for optimal trade execution.
        </p>
        <p className="mx-auto mt-3 max-w-xl text-sm text-muted-foreground">
          Dump a large order at once and you move the price against yourself. Spread it
          out wrong and you eat the day's drift instead. Velora trains and benchmarks
          agents — classical RL and LLM — on filling large orders against the same
          metric real execution desks are graded on: slippage vs. VWAP.
        </p>

        <div className="mt-8 flex items-center justify-center gap-3">
          <Button size="lg" onClick={() => navigate('/sandbox')}>
            Open sandbox →
          </Button>
          <Button size="lg" variant="outline" onClick={() => navigate('/execution-demo')}>
            Watch live demo
          </Button>
        </div>

        <div className="mx-auto mt-16 grid max-w-3xl grid-cols-2 gap-4 sm:grid-cols-4">
          {headlineStats.map(s => (
            <Card key={s.label}>
              <CardContent className="pt-6 text-center">
                <p className="text-2xl font-bold tracking-tight">{s.value}</p>
                <p className="mt-1 text-xs font-medium">{s.label}</p>
                <p className="mt-0.5 text-[11px] text-muted-foreground">{s.sub}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        <Card className="mt-6 text-left">
          <CardHeader>
            <div className="flex items-center justify-between gap-3">
              <div>
                <CardTitle className="text-base">HUD agent evaluation</CardTitle>
                <CardDescription className="mt-1">
                  {hudEval.model} run through the HUD environment. Each task is scored on
                  slippage vs. VWAP and normalized so 0.50 equals the benchmark —
                  Claude beat it on {hudEval.tasksBeatBenchmark} of {hudEval.tasksTotal} tasks.
                </CardDescription>
              </div>
              <a
                href={hudEval.jobUrl}
                target="_blank"
                rel="noreferrer"
                className="shrink-0 text-xs font-medium text-primary underline-offset-4 hover:underline"
              >
                View HUD job →
              </a>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {hudEval.tasks.map(t => {
                const pct = Math.max(0, Math.min(1, t.score)) * 100
                return (
                  <div key={t.id} className="flex items-center gap-3">
                    <span className="w-28 shrink-0 text-sm">{t.label}</span>
                    <div className="relative h-5 flex-1 overflow-hidden rounded bg-muted">
                      {/* benchmark marker at 0.50 */}
                      <div className="absolute inset-y-0 left-1/2 z-10 w-px bg-muted-foreground/60" />
                      <div
                        className={`h-full ${t.beat ? 'bg-green-500/80' : 'bg-amber-500/80'}`}
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className={`w-20 shrink-0 text-right text-sm font-semibold tabular-nums ${t.beat ? 'text-green-500' : 'text-amber-500'}`}>
                      {t.score.toFixed(3)}
                    </span>
                  </div>
                )
              })}
            </div>
            <div className="mt-3 flex items-center justify-between border-t pt-3 text-xs text-muted-foreground">
              <span>Dashed line = VWAP benchmark (0.50)</span>
              <span>
                Mean score <span className="font-semibold text-foreground">{hudEval.meanScore.toFixed(3)}</span>
                {' '}± {hudEval.stdScore.toFixed(3)}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card className="mt-4 text-left">
          <CardHeader>
            <CardTitle className="text-base">PPO on held-out days</CardTitle>
            <CardDescription className="mt-1">
              The trained PPO agent, evaluated on {ppoHoldout.nEpisodes} chronologically
              held-out {ppoHoldout.ticker} days it never saw in training, paired against the
              {' '}{ppoHoldout.baseline} baseline on the identical price path. It wins on{' '}
              <span className="font-semibold text-foreground">{(ppoHoldout.winRate * 100).toFixed(0)}%</span>{' '}
              of unseen days (median advantage{' '}
              <span className="font-semibold text-foreground">+{ppoHoldout.medianAdvantageBps.toFixed(2)} bps</span>),
              on an average order of ${(ppoHoldout.meanNotionalUsd / 1_000_000).toFixed(1)}M.
            </CardDescription>
          </CardHeader>
        </Card>

        <Card className="mt-4 text-left">
          <CardHeader>
            <CardTitle className="text-base">Who this is for</CardTitle>
            <CardDescription>
              Execution desks, prop trading firms, and asset managers measured on
              slippage vs. VWAP every day — plus quant teams who want a realistic,
              configurable RL sandbox to train and benchmark execution agents before
              risking real capital.
            </CardDescription>
          </CardHeader>
        </Card>
      </div>
    </div>
  )
}
