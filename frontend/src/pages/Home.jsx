import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2, Sparkles } from 'lucide-react'
import Navbar from '../components/Navbar'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { runEpisodes, parseStrategy, runStrategy } from '../api'

const SEEDS = [
  { key: 'tsla_rsi',    label: 'TSLA RSI Momentum' },
  { key: 'spy_ema_rsi', label: 'SPY EMA + RSI' },
  { key: 'nvda_sma_rsi', label: 'NVDA SMA + RSI' },
  { key: 'aapl_sma_rsi', label: 'AAPL SMA + RSI' },
  { key: 'tsla_sma_rsi', label: 'TSLA SMA + RSI' },
]

function describeCondition(c) {
  return `${c.indicator} ${c.operator} ${c.value}`
}

function ParsedStrategyPreview({ strategy, summary }) {
  return (
    <div className="space-y-3 rounded-lg border border-border bg-muted/30 p-3 text-sm">
      <p className="text-foreground/90">{summary}</p>
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant="secondary">{strategy.ticker}</Badge>
        {strategy.indicators.map((ind, i) => (
          <Badge key={i} variant="outline">{ind.type}({ind.period})</Badge>
        ))}
      </div>
      <div className="space-y-1 text-xs text-muted-foreground">
        <p>Entry: {strategy.entry_conditions.map(describeCondition).join(' AND ') || '—'}</p>
        <p>Exit: {strategy.exit_conditions.map(describeCondition).join(' OR ') || '—'}</p>
        <p>Stop loss: {(strategy.stop_loss * 100).toFixed(1)}% · Take profit: {(strategy.take_profit * 100).toFixed(1)}%</p>
      </div>
    </div>
  )
}

function SeedTab({ loading, error, onRun }) {
  const [selected, setSelected] = useState(new Set(SEEDS.map(s => s.key)))

  function toggleAll() {
    setSelected(selected.size === SEEDS.length ? new Set() : new Set(SEEDS.map(s => s.key)))
  }

  function toggle(key) {
    const next = new Set(selected)
    next.has(key) ? next.delete(key) : next.add(key)
    setSelected(next)
  }

  const allChecked = selected.size === SEEDS.length

  return (
    <div className="space-y-6">
      <div>
        <div className="mb-3 flex items-center justify-between">
          <span className="text-sm font-medium">Seed strategies</span>
          <button onClick={toggleAll} className="text-xs text-primary hover:underline">
            {allChecked ? 'Deselect all' : 'Select all'}
          </button>
        </div>
        <div className="space-y-3">
          {SEEDS.map(s => (
            <label key={s.key} className="flex items-center gap-3 cursor-pointer">
              <Checkbox checked={selected.has(s.key)} onCheckedChange={() => toggle(s.key)} />
              <span className="text-sm">{s.label}</span>
            </label>
          ))}
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Button
        onClick={() => onRun(selected.size === SEEDS.length ? ['all'] : [...selected])}
        disabled={loading || selected.size === 0}
        className="w-full"
        size="lg"
      >
        {loading ? (
          <>
            <Loader2 className="size-4 animate-spin" />
            Starting run…
          </>
        ) : (
          `Run ${selected.size} episode${selected.size !== 1 ? 's' : ''}`
        )}
      </Button>
    </div>
  )
}

function CustomTab({ loading, error, onRunStrategy }) {
  const [description, setDescription] = useState('')
  const [parsing, setParsing] = useState(false)
  const [parsed, setParsed] = useState(null)
  const [parseError, setParseError] = useState(null)

  async function handleParse() {
    if (!description.trim()) return
    setParsing(true)
    setParseError(null)
    setParsed(null)
    try {
      const result = await parseStrategy(description.trim())
      setParsed(result)
    } catch (e) {
      setParseError(e.message)
    } finally {
      setParsing(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <span className="mb-3 block text-sm font-medium">Describe a strategy</span>
        <Textarea
          value={description}
          onChange={(e) => { setDescription(e.target.value); setParsed(null) }}
          placeholder="e.g. Buy TSLA when RSI drops below 30, sell when it crosses back above 60. Use a 5% stop loss."
          className="min-h-24"
        />
      </div>

      {parseError && (
        <Alert variant="destructive">
          <AlertDescription>{parseError}</AlertDescription>
        </Alert>
      )}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {parsed && <ParsedStrategyPreview strategy={parsed.strategy} summary={parsed.summary} />}

      {parsed ? (
        <Button onClick={() => onRunStrategy(parsed.strategy)} disabled={loading} className="w-full" size="lg">
          {loading ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              Starting run…
            </>
          ) : (
            'Run this strategy'
          )}
        </Button>
      ) : (
        <Button onClick={handleParse} disabled={parsing || !description.trim()} className="w-full" size="lg" variant="outline">
          {parsing ? (
            <>
              <Loader2 className="size-4 animate-spin" />
              Parsing…
            </>
          ) : (
            <>
              <Sparkles className="size-4" />
              Parse strategy
            </>
          )}
        </Button>
      )}
    </div>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleRunSeeds(seeds) {
    setLoading(true)
    setError(null)
    try {
      const { run_id } = await runEpisodes(seeds)
      navigate(`/live/${run_id}`)
    } catch (e) {
      setError(e.message)
      setLoading(false)
    }
  }

  async function handleRunCustom(strategy) {
    setLoading(true)
    setError(null)
    try {
      const { run_id } = await runStrategy(strategy)
      navigate(`/live/${run_id}`)
    } catch (e) {
      setError(e.message)
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="flex min-h-[calc(100vh-3.5rem)] items-center justify-center px-4">
        <Card className="w-full max-w-md">
          <CardHeader>
            <CardTitle className="text-2xl">StratRL</CardTitle>
            <CardDescription>Claude-powered strategy optimizer</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="seeds">
              <TabsList className="mb-6 w-full">
                <TabsTrigger value="seeds" className="flex-1">Seed strategies</TabsTrigger>
                <TabsTrigger value="custom" className="flex-1">Describe your own</TabsTrigger>
              </TabsList>
              <TabsContent value="seeds">
                <SeedTab loading={loading} error={error} onRun={handleRunSeeds} />
              </TabsContent>
              <TabsContent value="custom">
                <CustomTab loading={loading} error={error} onRunStrategy={handleRunCustom} />
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
