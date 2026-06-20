import { useCallback, useEffect, useRef, useState } from 'react'
import Navbar from '../components/Navbar'
import ExecutionChart, { currentSlippageBps } from '../components/ExecutionChart'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { Separator } from '@/components/ui/separator'

const POLICY_LABELS = {
  twap: 'TWAP',
  ppo: 'PPO',
  llm: 'Claude LLM',
  fireworks: 'GPT-OSS (Fireworks)',
}

const inputClass =
  'h-9 w-full rounded-md border border-input bg-background px-3 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

function formatUsd(n) {
  if (n == null) return '—'
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(n)
}

function buildEpisodeUrl(config, sandboxConfig) {
  const params = new URLSearchParams()
  params.set('policy', config.policy)
  params.set('ticker', config.ticker)
  params.set('side', config.side)
  params.set('n_slices', String(config.n_slices))

  if (config.orderMode === 'capital') {
    params.set('capital_usd', String(config.capital_usd))
  } else {
    params.set('total_shares', String(config.total_shares))
  }

  if (config.dateMode === 'specific' && config.date) {
    params.set('date', config.date)
  } else if (config.dateMode === 'regime') {
    params.set('regime', config.regime)
  }

  if (config.seed !== '' && config.seed != null) {
    params.set('seed', String(config.seed))
  }

  return `/api/episode/stream?${params.toString()}`
}

export default function ExecutionSandbox() {
  const [sandboxConfig, setSandboxConfig] = useState(null)
  const [availablePolicies, setAvailablePolicies] = useState(['twap'])
  const [config, setConfig] = useState(null)
  const [episode, setEpisode] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [slice, setSlice] = useState(0)
  const [done, setDone] = useState(false)
  const esRef = useRef(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/config').then(r => r.json()),
      fetch('/api/policies').then(r => r.json()),
    ])
      .then(([cfg, policies]) => {
        setSandboxConfig(cfg)
        if (Array.isArray(policies.available)) setAvailablePolicies(policies.available)
        setConfig({
          ticker: cfg.defaults.ticker,
          side: cfg.defaults.side,
          orderMode: 'capital',
          total_shares: cfg.defaults.total_shares,
          capital_usd: cfg.defaults.capital_usd,
          n_slices: cfg.defaults.n_slices,
          timeframeId: cfg.timeframe_presets.find(p => p.n_slices === cfg.defaults.n_slices)?.id ?? '15min',
          dateMode: 'regime',
          date: cfg.ticker_info[cfg.defaults.ticker]?.date_range.end ?? '',
          regime: cfg.defaults.regime,
          seed: '',
          policy: cfg.defaults.policy,
        })
      })
      .catch(() => setError('Failed to load sandbox configuration'))
  }, [])

  const tickerInfo = sandboxConfig?.ticker_info?.[config?.ticker]
  const estimatedShares =
    config?.orderMode === 'capital' && tickerInfo?.last_close
      ? Math.max(
          sandboxConfig.constraints.total_shares.min,
          Math.floor(config.capital_usd / tickerInfo.last_close / 100) * 100,
        )
      : config?.total_shares

  const closeStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
  }, [])

  const runEpisode = useCallback(() => {
    if (!config || !sandboxConfig) return
    closeStream()
    setLoading(true)
    setError(null)
    setEpisode(null)
    setSlice(0)
    setDone(false)

    const es = new EventSource(buildEpisodeUrl(config, sandboxConfig))
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
  }, [config, sandboxConfig, closeStream])

  useEffect(() => closeStream, [closeStream])

  const updateConfig = (patch) => {
    setConfig(prev => {
      const next = { ...prev, ...patch }
      if (patch.ticker && sandboxConfig?.ticker_info?.[patch.ticker]) {
        next.date = sandboxConfig.ticker_info[patch.ticker].date_range.end
      }
      if (patch.dateMode === 'specific' && sandboxConfig?.ticker_info?.[next.ticker]) {
        next.date = sandboxConfig.ticker_info[next.ticker].date_range.end
      }
      return next
    })
  }

  const filledFraction = episode ? slice / episode.n_slices : 0
  const slippageBps = episode ? currentSlippageBps(episode, slice) : null
  const streaming = episode && !done && !loading

  const ppoDisabled = config && config.n_slices !== 26

  if (!sandboxConfig || !config) {
    return (
      <div className="min-h-screen bg-background text-foreground">
        <Navbar />
        <div className="mx-auto max-w-6xl px-4 py-16 text-center text-sm text-muted-foreground">
          Loading sandbox…
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navbar />
      <div className="mx-auto max-w-6xl px-4 py-8">
        <div className="mb-6">
          <h2 className="text-xl font-bold">Execution Sandbox</h2>
          <p className="mt-0.5 text-sm text-muted-foreground">
            Configure ticker, order size, timeframe, and replay date — then run any policy against real or synthetic intraday paths.
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
                <select
                  className={inputClass}
                  value={config.ticker}
                  onChange={e => updateConfig({ ticker: e.target.value })}
                >
                  {sandboxConfig.tickers.map(t => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Side</label>
                <div className="flex gap-2">
                  {['buy', 'sell'].map(s => (
                    <Button
                      key={s}
                      type="button"
                      size="sm"
                      variant={config.side === s ? 'default' : 'outline'}
                      className="flex-1 capitalize"
                      onClick={() => updateConfig({ side: s })}
                    >
                      {s}
                    </Button>
                  ))}
                </div>
              </div>

              <Separator />

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Order size</label>
                <div className="mb-2 flex gap-2">
                  {[
                    { id: 'capital', label: 'Notional ($)' },
                    { id: 'shares', label: 'Shares' },
                  ].map(mode => (
                    <Button
                      key={mode.id}
                      type="button"
                      size="sm"
                      variant={config.orderMode === mode.id ? 'default' : 'outline'}
                      className="flex-1 text-xs"
                      onClick={() => updateConfig({ orderMode: mode.id })}
                    >
                      {mode.label}
                    </Button>
                  ))}
                </div>
                {config.orderMode === 'capital' ? (
                  <>
                    <input
                      type="number"
                      className={inputClass}
                      min={sandboxConfig.constraints.capital_usd.min}
                      max={sandboxConfig.constraints.capital_usd.max}
                      step={sandboxConfig.constraints.capital_usd.step}
                      value={config.capital_usd}
                      onChange={e => updateConfig({ capital_usd: Number(e.target.value) })}
                    />
                    <p className="mt-1 text-xs text-muted-foreground">
                      ≈ {estimatedShares?.toLocaleString()} shares at ${tickerInfo?.last_close?.toFixed(2)} ref
                    </p>
                  </>
                ) : (
                  <input
                    type="number"
                    className={inputClass}
                    min={sandboxConfig.constraints.total_shares.min}
                    max={sandboxConfig.constraints.total_shares.max}
                    step={sandboxConfig.constraints.total_shares.step}
                    value={config.total_shares}
                    onChange={e => updateConfig({ total_shares: Number(e.target.value) })}
                  />
                )}
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Timeframe</label>
                <select
                  className={inputClass}
                  value={config.timeframeId}
                  onChange={e => {
                    const preset = sandboxConfig.timeframe_presets.find(p => p.id === e.target.value)
                    updateConfig({ timeframeId: e.target.value, n_slices: preset.n_slices })
                  }}
                >
                  {sandboxConfig.timeframe_presets.map(p => (
                    <option key={p.id} value={p.id}>
                      {p.label} ({p.n_slices} slices · {sandboxConfig.session.open}–{sandboxConfig.session.close} ET)
                    </option>
                  ))}
                </select>
              </div>

              <Separator />

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Replay day</label>
                <select
                  className={`${inputClass} mb-2`}
                  value={config.dateMode}
                  onChange={e => updateConfig({ dateMode: e.target.value })}
                >
                  <option value="random">Random day</option>
                  <option value="regime">Regime sample</option>
                  <option value="specific">Specific date</option>
                </select>

                {config.dateMode === 'regime' && (
                  <select
                    className={inputClass}
                    value={config.regime}
                    onChange={e => updateConfig({ regime: e.target.value })}
                  >
                    {sandboxConfig.regime_samples.map(r => (
                      <option key={r.id} value={r.id}>{r.label}</option>
                    ))}
                  </select>
                )}

                {config.dateMode === 'specific' && tickerInfo && (
                  <>
                    <input
                      type="date"
                      className={inputClass}
                      min={tickerInfo.date_range.start}
                      max={tickerInfo.date_range.end}
                      value={config.date || tickerInfo.date_range.end}
                      onChange={e => updateConfig({ date: e.target.value })}
                    />
                    <p className="mt-2 text-xs text-muted-foreground">
                      SIP data available {tickerInfo.date_range.start} → {tickerInfo.date_range.end}
                    </p>
                  </>
                )}

                {config.dateMode === 'regime' && config.regime !== 'random' && tickerInfo?.sample_dates && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    Example:{' '}
                    {tickerInfo.sample_dates.find(s => s.regime === config.regime)?.date ?? '—'}{' '}
                    ({tickerInfo.sample_dates.find(s => s.regime === config.regime)?.return_pct ?? 0}% day return)
                  </p>
                )}
              </div>

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Seed (optional)</label>
                <input
                  type="number"
                  className={inputClass}
                  placeholder="Reproducible sampling"
                  value={config.seed}
                  onChange={e => updateConfig({ seed: e.target.value })}
                />
              </div>

              <Separator />

              <div>
                <label className="mb-1 block text-xs font-medium text-muted-foreground">Policy</label>
                <select
                  className={inputClass}
                  value={config.policy}
                  onChange={e => updateConfig({ policy: e.target.value })}
                >
                  {availablePolicies.map(p => (
                    <option key={p} value={p} disabled={p === 'ppo' && ppoDisabled}>
                      {POLICY_LABELS[p] ?? p.toUpperCase()}
                      {p === 'ppo' && ppoDisabled ? ' (requires 15-min slices)' : ''}
                    </option>
                  ))}
                </select>
              </div>

              <Button className="w-full" onClick={runEpisode} disabled={loading || streaming}>
                {loading ? 'Connecting…' : streaming ? 'Streaming…' : episode ? 'Re-run scenario' : 'Run sandbox'}
              </Button>
            </CardContent>
          </Card>

          <div>
            {error && (
              <Card className="mb-4 border-red-500/50">
                <CardContent className="pt-6 text-sm text-red-500">{error}</CardContent>
              </Card>
            )}

            {episode && (
              <div className="mb-4 flex flex-wrap items-center gap-2">
                <Badge variant="secondary">{episode.ticker}</Badge>
                <Badge variant="secondary">{episode.date}</Badge>
                <Badge variant="secondary">{episode.side} {episode.total_shares.toLocaleString()} sh</Badge>
                <Badge variant="secondary">{formatUsd(episode.notional_usd)}</Badge>
                <Badge variant={episode.data_source === 'real_minute' ? 'default' : 'outline'}>
                  {episode.data_source === 'real_minute' ? 'Real minute bars' : 'Synthetic path'}
                </Badge>
                {episode.regime && episode.regime !== 'random' && (
                  <Badge variant="outline">{episode.regime.replace('_', ' ')}</Badge>
                )}
                {streaming ? (
                  <span className="flex items-center gap-1.5 text-xs font-medium text-red-500">
                    <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-red-500" />
                    LIVE
                  </span>
                ) : done ? (
                  <span className="text-xs font-medium text-green-500">✓ Complete</span>
                ) : null}
              </div>
            )}

            <Card>
              <CardHeader>
                <CardTitle className="text-base">Price path &amp; execution</CardTitle>
                {episode && (
                  <CardDescription>
                    {episode.schedule_label} · Open ${episode.open_price.toFixed(2)} → Close ${episode.close_price.toFixed(2)}
                    · ADV {Math.round(episode.adv).toLocaleString()}
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent>
                {episode ? (
                  <ExecutionChart episode={episode} currentSlice={slice} />
                ) : (
                  <div className="flex h-80 flex-col items-center justify-center gap-2 text-sm text-muted-foreground">
                    <p>Configure a scenario and hit Run sandbox</p>
                    <p className="text-xs">Try high-vol TSLA, a calm SPY day, or pin an exact historical date</p>
                  </div>
                )}
              </CardContent>
            </Card>

            {episode && (
              <>
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
                  <Button onClick={runEpisode} disabled={loading || streaming}>
                    {done ? 'Re-run' : streaming ? 'Streaming…' : 'Run'}
                  </Button>
                </div>

                {episode.llm_reasoning && (
                  <Card className="mt-4">
                    <CardContent className="pt-4">
                      <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">Agent reasoning</p>
                      <p className="text-sm leading-relaxed">{episode.llm_reasoning}</p>
                    </CardContent>
                  </Card>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
