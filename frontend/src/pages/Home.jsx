import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'
import AnimatedNumber from '../components/AnimatedNumber'
import { policyMeta, formatUsd, formatBps, slippageBpsUpTo, usdFromBps } from '@/lib/execution'
import { hudEval, ppoHoldout, headlineStats } from '@/data/benchmarks'

// A pinned, reproducible example scenario for the landing headline.
const HERO = 'policies=ppo,naive_twap&ticker=TSLA&side=buy&adv_pct=10&regime=high_vol&seed=7'
const HERO_FALLBACK = 'policies=twap,naive_twap&ticker=TSLA&side=buy&adv_pct=10&regime=high_vol&seed=7'

function heroStat(data) {
  if (!data?.policies?.length) return null
  const [a, b] = data.policies
  const n = data.n_slices
  const slipA = slippageBpsUpTo(data, a.exec_prices, a.exec_quantities, n)
  const slipB = slippageBpsUpTo(data, b.exec_prices, b.exec_quantities, n)
  const advBps = slipA - slipB
  return {
    advBps,
    advUsd: usdFromBps(advBps, data.notional_usd),
    notional: data.notional_usd,
    agent: policyMeta(a.policy),
    baseline: policyMeta(b.policy),
    ticker: data.ticker,
    advPct: data.order_adv_pct,
  }
}

// ─── SVG market path for the hero panel (viewBox 0 0 600 260) ──────────────────
const HERO_MARKS = [[95, 92], [221, 112], [347, 82], [474, 68], [505, 33]]
const HERO_MKT =
  'M 0,170 L 32,128 L 63,183 L 95,92  L 126,148 L 158,207 L 190,168 L 221,112 L 253,140 L 284,190 L 316,133 L 347,82  L 379,157 L 411,200 L 442,123 L 474,68  L 505,33  L 537,102 L 568,148 L 600,107'

function HeroPanel({ stat }) {
  const value = stat ? Math.abs(stat.advBps) : null
  return (
    <div
      className="relative w-full overflow-hidden rounded-[10px]"
      style={{ boxShadow: '0 0 140px -20px rgba(174,147,87,0.40)' }}
    >
      <div className="relative bg-[#000000]">
        <div className="absolute left-5 top-5 z-10">
          <p className="text-[10px] font-medium uppercase tracking-[0.07em] text-[#464853]">
            {stat ? `${stat.agent.short} vs ${stat.baseline.short} · ${stat.ticker}` : 'Slippage vs. VWAP'}
          </p>
          <p
            className="mt-1 text-[44px] font-light leading-none text-[#cc9166]"
            style={{ letterSpacing: '-0.025em', fontVariantNumeric: 'tabular-nums' }}
          >
            {value == null ? '—' : <AnimatedNumber value={value} format={n => `+${n.toFixed(1)} bps`} />}
          </p>
          {stat && (
            <p className="mt-1 text-[12px] font-light text-[#5e616e]">
              ≈ {formatUsd(Math.abs(stat.advUsd))} on a {formatUsd(stat.notional)} order
            </p>
          )}
        </div>

        <svg viewBox="0 0 600 260" className="w-full" style={{ height: 300, display: 'block' }} preserveAspectRatio="none">
          <defs>
            <linearGradient id="hMoltGold" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#fff0cc" stopOpacity="0.95" />
              <stop offset="20%" stopColor="#ae9357" stopOpacity="0.80" />
              <stop offset="55%" stopColor="#ae9357" stopOpacity="0.30" />
              <stop offset="100%" stopColor="#ae9357" stopOpacity="0" />
            </linearGradient>
            <filter id="hGlow">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          {[95, 221, 347, 474, 505].map(x => (
            <line key={x} x1={x} y1="0" x2={x} y2="260" stroke="#1e2028" strokeWidth="0.8" strokeDasharray="3 5" />
          ))}
          {[70, 130, 190].map(y => (
            <line key={y} x1="0" y1={y} x2="600" y2={y} stroke="#1a1b20" strokeWidth="0.8" />
          ))}
          <path d="M 0,170 L 32,128 L 63,183 L 95,92  L 126,148 L 158,207 L 190,168 L 221,112 L 253,140 L 284,190 L 316,133 L 347,82  L 379,157 L 411,200 L 442,123 L 474,68  L 505,33  L 537,102 L 568,148 L 600,107 L 600,260 L 0,260 Z" fill="url(#hMoltGold)" />
          <path d={HERO_MKT} fill="none" stroke="#d4d6dd" strokeWidth="1.5" filter="url(#hGlow)" />
          {HERO_MARKS.map(([x, y], i) => (
            <circle key={i} cx={x} cy={y} r="4" fill="#cc9166" opacity="0.95" />
          ))}
        </svg>
      </div>

      <div className="flex justify-between bg-[#000000] px-4 pb-3 pt-1">
        {['09:30', '11:00', '12:30', '14:00', '16:00'].map(t => (
          <span key={t} className="text-[11px] font-light text-[#464853]">{t}</span>
        ))}
      </div>
    </div>
  )
}

export default function Home() {
  const navigate = useNavigate()
  const [stat, setStat] = useState(null)

  useEffect(() => {
    const load = (q) => fetch(`/api/compare?${q}`).then(r => { if (!r.ok) throw new Error(); return r.json() })
    load(HERO)
      .catch(() => load(HERO_FALLBACK))
      .then(d => setStat(heroStat(d)))
      .catch(() => {})
  }, [])

  return (
    <div className="min-h-screen bg-[#000000] text-[#e2e3e9]" style={{ fontFamily: "'Inter', sans-serif" }}>

      {/* Announcement */}
      <div className="flex items-center justify-center gap-2 border-b border-[#2e3038] bg-[#08080a] px-4 py-2.5">
        <span className="text-[12px] font-light tracking-[-0.01em] text-[#777a88]">
          Now running Claude Sonnet 4 in the live execution demo
        </span>
        <button
          onClick={() => navigate('/execution-demo')}
          className="text-[12px] font-medium text-[#cc9166] transition-opacity hover:opacity-75"
        >
          Watch it trade →
        </button>
      </div>

      <Navbar />

      {/* ── Hero ─────────────────────────────────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-8 pb-8 pt-20">
        <div className="grid grid-cols-1 items-center gap-14 lg:grid-cols-[1fr_1.4fr]">
          <div>
            <h1 className="mb-5">
              <span className="block font-light leading-[1.08] tracking-[-0.02em] text-[#e2e3e9]"
                style={{ fontSize: 'clamp(40px, 4vw, 56px)' }}>
                A sharper edge
              </span>
              <span className="block leading-[1.0] tracking-[-0.02em] text-[#cc9166]"
                style={{
                  fontFamily: "'Playfair Display', Georgia, serif",
                  fontSize: 'clamp(44px, 4.5vw, 64px)', fontWeight: 500, fontStyle: 'italic',
                }}>
                in every trade.
              </span>
            </h1>
            <p className="mb-8 max-w-[440px] text-[15px] font-light leading-[1.7] tracking-[-0.015em] text-[#777a88]">
              Dump a large order at once and you move the price against yourself.
              Spread it out wrong and you eat the day's drift. Velora benchmarks agents —
              classical RL and LLM — against the metric real desks live by: slippage vs. VWAP.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => navigate('/showdown')}
                className="rounded-[2px] bg-white px-5 py-[10px] text-[14px] font-medium text-[#08080a] transition-opacity hover:opacity-90"
              >
                Watch the showdown →
              </button>
              <button
                onClick={() => navigate('/proof')}
                className="rounded-[2px] border border-[#464853] px-5 py-[10px] text-[14px] font-medium text-[#e2e3e9] transition-colors hover:border-[#9194a1]"
              >
                See the proof
              </button>
            </div>
          </div>
          <HeroPanel stat={stat} />
        </div>
      </section>

      {/* ── Stats strip (real metrics) ───────────────────────────────────────── */}
      <section className="border-y border-[#2e3038]">
        <div className="mx-auto grid max-w-6xl grid-cols-2 divide-x divide-[#2e3038] md:grid-cols-4">
          {headlineStats.map(({ label, value, sub }) => (
            <div key={label} className="group cursor-default px-8 py-8 text-center transition-colors hover:bg-[#08080a]">
              <p className="text-[22px] font-semibold text-[#e2e3e9] transition-colors group-hover:text-white"
                 style={{ fontVariantNumeric: 'tabular-nums' }}>
                {value}
              </p>
              <p className="mt-1 text-[12px] font-medium text-[#9194a1]">{label}</p>
              <p className="mt-0.5 text-[11px] font-light tracking-[-0.01em] text-[#5e616e]">{sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── The benchmark: real HUD eval ─────────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-8 py-20">
        <div className="mb-12">
          <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.08em] text-[#5e616e]">The benchmark</p>
          <h2 className="text-[36px] font-light leading-[1.15] text-[#e2e3e9]" style={{ letterSpacing: '-0.018em' }}>
            Scored on a HUD environment,{' '}
            <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400, color: '#cc9166' }}>
              not a backtest.
            </span>
          </h2>
          <p className="mt-4 max-w-2xl text-[15px] font-light leading-[1.65] tracking-[-0.012em] text-[#777a88]">
            The {hudEval.model} execution agent, evaluated through the HUD gateway on {hudEval.date}.
            Reward is normalized so 0.50 equals matching VWAP — above 0.50 beats the desk's default.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.4fr_1fr]">
          {/* HUD eval card */}
          <div className="overflow-hidden rounded-[10px] border border-[#2e3038] bg-[#1c1d22]">
            <div className="flex items-center justify-between border-b border-[#2e3038] bg-[#08080a] px-6 py-4">
              <div>
                <p className="text-[13px] font-medium text-[#e2e3e9]">HUD agent evaluation</p>
                <p className="mt-0.5 text-[11px] font-light text-[#5e616e]">Per-task normalized reward · {hudEval.runtime}</p>
              </div>
              <div className="text-right">
                <p className="text-[28px] font-light leading-none text-[#cc9166]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {hudEval.meanScore.toFixed(3)}
                </p>
                <p className="mt-1 text-[11px] font-light text-[#5e616e]">mean · {hudEval.tasksBeatBenchmark}/{hudEval.tasksTotal} beat VWAP</p>
              </div>
            </div>
            <div className="space-y-5 p-6">
              {hudEval.tasks.map(task => {
                // Map score onto a bar; 0.5 (VWAP) sits at the midpoint, 1.0 fills the bar.
                const pct = Math.max(4, Math.min(100, task.score * 100))
                const color = task.beat ? '#cc9166' : '#c0553a'
                return (
                  <div key={task.id}>
                    <div className="mb-1.5 flex items-center justify-between">
                      <span className="text-[12px] font-medium text-[#acafb9]">{task.label}</span>
                      <span className="text-[12px] font-medium" style={{ color, fontVariantNumeric: 'tabular-nums' }}>
                        {task.score.toFixed(3)} {task.beat ? '· beat' : '· missed'}
                      </span>
                    </div>
                    <div className="relative h-[4px] rounded-full bg-[#2e3038]">
                      {/* VWAP midpoint marker at 0.50 */}
                      <span className="absolute top-[-3px] h-[10px] w-px bg-[#5e616e]" style={{ left: '50%' }} />
                      <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
                    </div>
                  </div>
                )
              })}
              <p className="pt-1 text-[11px] font-light text-[#464853]">
                Vertical marker = VWAP benchmark (0.50). Bars past it beat the desk's default.
              </p>
            </div>
            <div className="border-t border-[#2e3038] px-6 py-3">
              <a
                href={hudEval.jobUrl}
                target="_blank"
                rel="noreferrer"
                className="text-[12px] font-medium text-[#cc9166] transition-opacity hover:opacity-75"
              >
                View the HUD job →
              </a>
            </div>
          </div>

          {/* PPO held-out card */}
          <div className="flex flex-col justify-between overflow-hidden rounded-[10px] border border-[#2e3038] bg-[#1c1d22]">
            <div className="p-6">
              <p className="text-[10px] font-medium uppercase tracking-[0.07em] text-[#464853]">PPO · held-out days</p>
              <p className="mt-3 text-[44px] font-light leading-none text-[#acafb9]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                {(ppoHoldout.winRate * 100).toFixed(0)}%
              </p>
              <p className="mt-2 text-[13px] font-light leading-[1.6] text-[#777a88]">
                win-rate vs the {ppoHoldout.baseline} baseline across {ppoHoldout.nEpisodes} {ppoHoldout.ticker} days the
                model never trained on — same paired price paths, the only variable is the policy.
              </p>
            </div>
            <div className="grid grid-cols-3 divide-x divide-[#2e3038] border-t border-[#2e3038]">
              <div className="px-5 py-4">
                <p className="text-[11px] font-light text-[#5e616e]">Median edge</p>
                <p className="mt-1 text-[16px] font-semibold text-[#cc9166]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {formatBps(ppoHoldout.medianAdvantageBps)}
                </p>
              </div>
              <div className="px-5 py-4">
                <p className="text-[11px] font-light text-[#5e616e]">Significance</p>
                <p className="mt-1 text-[16px] font-semibold text-[#cc9166]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  t = {ppoHoldout.tStat.toFixed(2)}
                </p>
              </div>
              <div className="px-5 py-4">
                <p className="text-[11px] font-light text-[#5e616e]">Avg order</p>
                <p className="mt-1 text-[16px] font-semibold text-[#e2e3e9]" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {formatUsd(ppoHoldout.meanNotionalUsd, { notation: 'compact', maximumFractionDigits: 1 })}
                </p>
              </div>
            </div>
            <button
              onClick={() => navigate('/proof')}
              className="border-t border-[#2e3038] px-6 py-3 text-left text-[12px] font-medium text-[#cc9166] transition-colors hover:bg-[#08080a]"
            >
              Reproduce it live →
            </button>
          </div>
        </div>
      </section>

      {/* ── Three product pillars ────────────────────────────────────────────── */}
      <section className="border-t border-[#2e3038]">
        <div className="mx-auto max-w-6xl px-8 py-24">
          <div className="mb-16 text-center">
            <h2 className="mx-auto max-w-3xl text-[44px] font-light leading-[1.1] text-[#e2e3e9]" style={{ letterSpacing: '-0.022em' }}>
              Three ways to{' '}
              <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400, color: '#cc9166' }}>
                interrogate the agents.
              </span>
            </h2>
            <p className="mx-auto mt-5 max-w-xl text-[15px] font-light leading-[1.65] tracking-[-0.012em] text-[#777a88]">
              Real OHLCV data, a standard Almgren–Chriss impact model, and slippage-first metrics —
              for the people who live and die by execution quality.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {[
              {
                title: 'Showdown',
                to: '/showdown',
                desc: 'Race the RL agent against the naive baseline on the same price path, slice by slice. The only variable is the policy.',
              },
              {
                title: 'Proof',
                to: '/proof',
                desc: 'Win-rate and bps advantage across dozens of held-out days the model never trained on — the actual statistical evidence.',
              },
              {
                title: 'Sandbox',
                to: '/sandbox',
                desc: 'Configure any ticker, order size, regime and policy set, and overlay them on real intraday data in one chart.',
              },
            ].map(c => (
              <button
                key={c.title}
                onClick={() => navigate(c.to)}
                className="group overflow-hidden rounded-[10px] border border-[#2e3038] bg-[#1c1d22] p-7 text-left transition-colors hover:border-[#464853]"
              >
                <p className="mb-2 text-[18px] font-medium tracking-[-0.012em] text-[#e2e3e9]"
                   style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 500 }}>
                  {c.title}
                </p>
                <p className="text-[13px] font-light leading-[1.65] tracking-[-0.01em] text-[#777a88]">{c.desc}</p>
                <p className="mt-5 text-[12px] font-medium text-[#cc9166] transition-opacity group-hover:opacity-75">
                  Open {c.title.toLowerCase()} →
                </p>
              </button>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}
