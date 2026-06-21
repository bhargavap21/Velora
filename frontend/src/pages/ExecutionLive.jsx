import { useEffect, useRef, useState } from 'react'
import Navbar from '../components/Navbar'
import ExecutionChart, { currentSlippageBps } from '../components/ExecutionChart'
import { Progress } from '@/components/ui/progress'
import { apiUrl } from '@/lib/api'

const POLICY_LABELS = {
  naive_twap: 'TWAP (equal-time)',
  twap: 'VWAP-match',
  ppo: 'PPO',
  llm: 'Claude LLM',
  fireworks: 'GPT-OSS (Fireworks)',
}

/* ── Shared card primitive ── */
function Panel({ className = '', children }) {
  return (
    <div className={`rounded-[10px] border border-[#1e2028] bg-[#0d0e12] ${className}`}>
      {children}
    </div>
  )
}

/* ── Pill tag ── */
function Pill({ children }) {
  return (
    <span className="inline-flex items-center rounded-full border border-[#464853] px-2.5 py-[3px] text-[12px] font-medium text-[#9194a1]">
      {children}
    </span>
  )
}

export default function ExecutionLive() {
  const [policy, setPolicy] = useState('ppo')
  const [availablePolicies, setAvailablePolicies] = useState(['twap'])
  const [episode, setEpisode] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [slice, setSlice] = useState(0)
  const [done, setDone] = useState(false)
  const esRef = useRef(null)

  useEffect(() => {
    fetch(apiUrl('/api/policies'))
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

    const es = new EventSource(apiUrl(`/api/episode/stream?policy=${selectedPolicy}`))
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

  const slippageColor =
    slippageBps == null ? '#777a88'
    : slippageBps >= 0  ? '#cc9166'
    : '#c0553a'

  return (
    <div
      className="min-h-screen bg-[#000000] text-[#e2e3e9]"
      style={{ fontFamily: "'Inter', sans-serif" }}
    >
      <Navbar />

      <div className="mx-auto max-w-5xl px-6 py-8">

        {/* ── Page header ── */}
        <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-[20px] font-semibold leading-tight text-[#e2e3e9]">
              Live Execution
            </h1>
            <p className="mt-1 text-[13px] text-[#777a88]">
              {episode
                ? `${episode.schedule_label} — streamed live from the backend simulator`
                : 'Connecting to simulator…'}
            </p>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-2">
            <select
              className="h-8 rounded-[2px] border border-[#2e3038] bg-[#121317] px-2.5 text-[13px] font-medium text-[#e2e3e9] outline-none transition-colors hover:border-[#464853] focus:border-[#cc9166] focus:ring-1 focus:ring-[#cc9166]/30 disabled:opacity-50"
              value={policy}
              disabled={loading || streaming}
              onChange={e => { setPolicy(e.target.value); runEpisode(e.target.value) }}
            >
              {availablePolicies.map(p => (
                <option key={p} value={p}>{POLICY_LABELS[p] ?? p.toUpperCase()}</option>
              ))}
            </select>

            {episode && <Pill>{episode.ticker}</Pill>}
            {episode && <Pill>{episode.side} {episode.total_shares.toLocaleString()} sh</Pill>}

            {loading ? (
              <span className="animate-pulse text-[12px] text-[#777a88]">Connecting…</span>
            ) : done ? (
              <span className="text-[12px] font-medium text-[#cc9166]">Complete</span>
            ) : (
              <span className="flex items-center gap-1.5 text-[12px] font-medium text-[#c0553a]">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-[#c0553a]" />
                LIVE
              </span>
            )}
          </div>
        </div>

        {/* ── Error ── */}
        {error && (
          <Panel className="mb-4 border-[#c0553a]/50">
            <div className="px-5 py-4 text-[13px] text-[#c0553a]">{error}</div>
          </Panel>
        )}

        {/* ── Main chart ── */}
        <Panel className="mb-4">
          <div className="border-b border-[#1e2028] px-6 py-4">
            <p className="text-[13px] font-medium text-[#e2e3e9]">Price path &amp; execution</p>
            <p className="mt-0.5 text-[12px] text-[#5e616e]">
              Market price, VWAP benchmark, and agent execution cost
            </p>
          </div>
          <div className="px-4 py-4">
            {episode ? (
              <ExecutionChart episode={episode} currentSlice={slice} />
            ) : (
              <div className="flex h-80 items-center justify-center text-[13px] text-[#5e616e]">
                Connecting…
              </div>
            )}
          </div>
        </Panel>

        {/* ── Metric cards ── */}
        <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">

          {/* Filled */}
          <Panel className="p-5">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.06em] text-[#5e616e]">
              Filled
            </p>
            <p className="mb-3 text-[28px] font-medium leading-none text-[#e2e3e9]"
               style={{ fontVariantNumeric: 'tabular-nums' }}>
              {(filledFraction * 100).toFixed(0)}%
            </p>
            <Progress value={filledFraction * 100} className="h-[2px] bg-[#2e3038]" />
          </Panel>

          {/* Slippage */}
          <Panel className="p-5">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.06em] text-[#5e616e]">
              Slippage vs. VWAP
            </p>
            <p
              className="text-[28px] font-semibold leading-none"
              style={{ color: slippageColor, fontVariantNumeric: 'tabular-nums' }}
            >
              {slippageBps == null
                ? '—'
                : `${slippageBps >= 0 ? '+' : ''}${slippageBps.toFixed(1)} bps`}
            </p>
          </Panel>

          {/* Slice */}
          <Panel className="p-5">
            <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.06em] text-[#5e616e]">
              Slice
            </p>
            <p
              className="text-[28px] font-semibold leading-none text-[#e2e3e9]"
              style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              {slice}
              <span className="ml-1 text-[16px] font-normal text-[#5e616e]">
                / {episode ? episode.n_slices : '—'}
              </span>
            </p>
          </Panel>
        </div>

        {/* ── Action row ── */}
        <div className="mt-4 flex gap-2">
          <button
            onClick={() => runEpisode(policy)}
            disabled={loading || streaming}
            className="rounded-[2px] bg-white px-4 py-[9px] text-[14px] font-medium text-[#08080a] transition-opacity hover:opacity-90 disabled:opacity-50"
          >
            {done ? 'New episode' : streaming ? 'Streaming…' : 'Run episode'}
          </button>
        </div>

        {/* ── LLM reasoning ── */}
        {episode?.llm_reasoning && (
          <Panel className="mt-4 p-5">
            <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.06em] text-[#5e616e]">
              {policy === 'fireworks' ? 'GPT-OSS reasoning' : 'Claude reasoning'}
            </p>
            <p className="text-[14px] leading-[1.6] text-[#acafb9]">
              {episode.llm_reasoning}
            </p>
            {episode.llm_primitive && (
              <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-[#1e2028] pt-4 text-[12px] text-[#5e616e]">
                <span>
                  Schedule primitive:{' '}
                  <span className="font-medium text-[#e2e3e9]">{episode.llm_primitive}</span>
                </span>
                {episode.llm_pause_enabled && (
                  <span>
                    Pause on adverse move &gt;{' '}
                    <span className="font-medium text-[#cc9166]">
                      {episode.llm_pause_threshold_bps} bps
                    </span>
                  </span>
                )}
              </div>
            )}
          </Panel>
        )}
      </div>
    </div>
  )
}
