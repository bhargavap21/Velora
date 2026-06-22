import { useCallback, useEffect, useState } from 'react'
import {
  BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip,
  LabelList, ResponsiveContainer,
} from 'recharts'
import { annotate } from '@/lib/annotate'
import { apiUrl } from '@/lib/api'

const OBS_LABELS = [
  'inventory_remaining',
  'time_remaining',
  'current_price',
  'vwap_so_far',
  'spread',
  'volatility',
  'volume_imbalance',
  'participation_rate_lag',
]

// Terminal green palette
const BG         = '#050a05'
const GREEN      = '#4ade80'
const GREEN_SOFT = '#86efac'
const GREEN_DIM  = '#166534'
const GREEN_GRID = '#0f2f0f'
const ACTION_BAR = '#86efac'

function ObsBarChart({ obs, action, annotation }) {
  const chartData = [
    ...obs.map((v, i) => ({
      dim: OBS_LABELS[i],
      // chart domain [0, 1] — obs are already normalised by the gym
      value: Math.min(1, Math.max(0, v)),
      raw: v,
      isAction: false,
      isActive: annotation.activeDims.includes(i),
    })),
    {
      dim: 'action (rate)',
      // action is participation multiplier in [0, 2]; halve for chart domain
      value: Math.min(1, Math.max(0, action / 2)),
      raw: action,
      isAction: true,
      isActive: true,
    },
  ]

  return (
    <ResponsiveContainer width="100%" height={310}>
      <BarChart
        layout="vertical"
        data={chartData}
        margin={{ top: 4, right: 64, bottom: 4, left: 168 }}
      >
        <CartesianGrid horizontal={false} stroke={GREEN_GRID} />
        <XAxis
          type="number"
          domain={[0, 1]}
          tick={{ fill: GREEN_DIM, fontSize: 10, fontFamily: 'monospace' }}
          tickLine={false}
          axisLine={{ stroke: GREEN_GRID }}
        />
        <YAxis
          type="category"
          dataKey="dim"
          width={163}
          tick={{ fill: GREEN, fontSize: 10, fontFamily: 'monospace' }}
          tickLine={false}
          axisLine={{ stroke: GREEN_GRID }}
        />
        <Tooltip
          cursor={{ fill: '#0a1a0a' }}
          contentStyle={{
            backgroundColor: '#0a1a0a',
            border: `1px solid ${GREEN_GRID}`,
            borderRadius: 4,
            fontSize: 11,
            fontFamily: 'monospace',
            color: GREEN,
          }}
          formatter={(_, __, props) => [props.payload.raw.toFixed(4), '']}
          labelStyle={{ color: GREEN }}
        />
        <Bar dataKey="value" radius={[0, 2, 2, 0]} isAnimationActive={false}>
          {chartData.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.isAction ? ACTION_BAR : entry.isActive ? GREEN : GREEN_DIM}
              fillOpacity={entry.isAction ? 1 : entry.isActive ? 0.85 : 0.35}
            />
          ))}
          <LabelList
            dataKey="raw"
            position="right"
            formatter={v => (v != null && typeof v === 'number' ? v.toFixed(3) : '')}
            style={{ fill: GREEN, fontSize: 10, fontFamily: 'monospace' }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export default function PolicyReplayPanel({ runId: runIdProp }) {
  const [runId, setRunId]         = useState(runIdProp ?? '')
  const [episodeLog, setLog]      = useState(null)
  const [step, setStep]           = useState(0)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState(null)
  const [dragOver, setDragOver]   = useState(false)

  // Auto-fetch if a runId prop is supplied (e.g. linked from another page)
  useEffect(() => {
    if (runIdProp) fetchLog(runIdProp)
  }, [runIdProp]) // eslint-disable-line react-hooks/exhaustive-deps

  // Arrow-key scrubbing
  useEffect(() => {
    if (!episodeLog) return
    const last = episodeLog.steps.length - 1
    const onKey = e => {
      if (e.key === 'ArrowRight') setStep(s => Math.min(s + 1, last))
      if (e.key === 'ArrowLeft')  setStep(s => Math.max(s - 1, 0))
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [episodeLog])

  const fetchLog = useCallback(async id => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(apiUrl(`/api/episode/log?run_id=${encodeURIComponent(id)}`))
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail ?? `HTTP ${res.status}`)
      }
      setLog(await res.json())
      setStep(0)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const handleDrop = useCallback(e => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer?.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = ev => {
      try {
        const data = JSON.parse(ev.target.result)
        if (!Array.isArray(data?.steps)) throw new Error('Missing "steps" array')
        setLog(data)
        setStep(0)
        setError(null)
      } catch (err) {
        setError(`Invalid file: ${err.message}`)
      }
    }
    reader.readAsText(file)
  }, [])

  const current   = episodeLog?.steps[step]
  const annotation = current ? annotate(current.obs) : null

  return (
    <div
      style={{ backgroundColor: BG, color: GREEN, minHeight: '100%' }}
      className="font-mono w-full"
      onDragOver={e => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      {/* Terminal chrome */}
      <div style={{
        borderBottom: `1px solid ${GREEN_GRID}`,
        padding: '10px 20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ fontSize: 11, letterSpacing: '0.15em' }}>▸ POLICY REPLAY</span>
        {episodeLog && (
          <span style={{ fontSize: 11, color: GREEN_SOFT }}>
            step {step + 1} / {episodeLog.steps.length}
          </span>
        )}
      </div>

      <div style={{ padding: '24px 20px', maxWidth: 880, margin: '0 auto' }}>

        {/* ── Load panel ─────────────────────────────────────── */}
        {!episodeLog && (
          <div style={{ marginBottom: 24 }}>
            <div
              style={{
                border: `1px dashed ${dragOver ? GREEN : GREEN_GRID}`,
                borderRadius: 6,
                padding: '28px 20px',
                textAlign: 'center',
                fontSize: 12,
                color: dragOver ? GREEN : GREEN_DIM,
                marginBottom: 14,
                transition: 'border-color 0.12s, color 0.12s',
              }}
            >
              drop episode_log.json here
            </div>

            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: GREEN_DIM, whiteSpace: 'nowrap' }}>
                or fetch by run_id:
              </span>
              <input
                value={runId}
                onChange={e => setRunId(e.target.value)}
                placeholder="e.g. a3f9bc12"
                onKeyDown={e => e.key === 'Enter' && runId && fetchLog(runId)}
                style={{
                  flex: 1,
                  background: '#0a1a0a',
                  border: `1px solid ${GREEN_GRID}`,
                  borderRadius: 4,
                  color: GREEN,
                  fontFamily: 'monospace',
                  fontSize: 12,
                  padding: '6px 10px',
                  outline: 'none',
                }}
              />
              <button
                onClick={() => runId && fetchLog(runId)}
                disabled={!runId || loading}
                style={{
                  background: 'transparent',
                  border: `1px solid ${!runId || loading ? GREEN_GRID : GREEN}`,
                  color: !runId || loading ? GREEN_DIM : GREEN,
                  fontFamily: 'monospace',
                  fontSize: 11,
                  padding: '6px 14px',
                  borderRadius: 4,
                  cursor: !runId || loading ? 'not-allowed' : 'pointer',
                  letterSpacing: '0.1em',
                }}
              >
                {loading ? 'loading…' : 'LOAD'}
              </button>
            </div>
          </div>
        )}

        {error && (
          <div style={{ color: '#f97316', fontSize: 12, marginBottom: 16 }}>
            ✗ {error}
          </div>
        )}

        {/* ── Step view ──────────────────────────────────────── */}
        {episodeLog && current && annotation && (
          <>
            {/* Annotation */}
            <div style={{ marginBottom: 20, display: 'flex', alignItems: 'center', gap: 12 }}>
              <span style={{
                border: `1px solid ${annotation.color}`,
                color: annotation.color,
                fontSize: 10,
                letterSpacing: '0.15em',
                padding: '3px 10px',
                borderRadius: 3,
                flexShrink: 0,
              }}>
                {annotation.label}
              </span>
              <span style={{ fontSize: 13, color: GREEN_SOFT }}>{annotation.text}</span>
            </div>

            {/* Reward */}
            <div style={{ fontSize: 11, color: GREEN_DIM, marginBottom: 20 }}>
              reward:{' '}
              <span style={{ color: current.reward >= 0 ? GREEN : '#f97316' }}>
                {current.reward >= 0 ? '+' : ''}{current.reward.toFixed(4)}
              </span>
            </div>

            {/* Chart */}
            <div style={{ marginBottom: 24 }}>
              <div style={{ fontSize: 10, letterSpacing: '0.12em', color: GREEN_DIM, marginBottom: 8 }}>
                OBS VECTOR + ACTION
              </div>
              <ObsBarChart obs={current.obs} action={current.action} annotation={annotation} />
            </div>

            {/* Scrubber */}
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 10, letterSpacing: '0.12em', color: GREEN_DIM, marginBottom: 8 }}>
                TIMESTEP — use ← → to step
              </div>
              <input
                type="range"
                min={0}
                max={episodeLog.steps.length - 1}
                value={step}
                onChange={e => setStep(Number(e.target.value))}
                style={{ width: '100%', accentColor: GREEN, cursor: 'pointer' }}
              />
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                fontSize: 10,
                color: GREEN_DIM,
                marginTop: 4,
              }}>
                <span>t=0</span>
                <span style={{ color: GREEN }}>t={step}</span>
                <span>t={episodeLog.steps.length - 1}</span>
              </div>
            </div>

            <button
              onClick={() => { setLog(null); setError(null); setRunId(runIdProp ?? '') }}
              style={{
                fontSize: 10,
                color: GREEN_DIM,
                background: 'none',
                border: 'none',
                cursor: 'pointer',
                letterSpacing: '0.1em',
                padding: 0,
              }}
            >
              ↺ load different episode
            </button>
          </>
        )}
      </div>
    </div>
  )
}
