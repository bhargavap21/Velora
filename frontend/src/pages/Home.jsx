import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'

// ─── SVG paths (viewBox 0 0 600 240) ──────────────────────────────────────────
const P = {
  mkt:    'M 0,170 L 32,128 L 63,183 L 95,92  L 126,148 L 158,207 L 190,168 L 221,112 L 253,140 L 284,190 L 316,133 L 347,82  L 379,157 L 411,200 L 442,123 L 474,68  L 505,33  L 537,102 L 568,148 L 600,107',
  mktA:   'M 0,170 L 32,128 L 63,183 L 95,92  L 126,148 L 158,207 L 190,168 L 221,112 L 253,140 L 284,190 L 316,133 L 347,82  L 379,157 L 411,200 L 442,123 L 474,68  L 505,33  L 537,102 L 568,148 L 600,107 L 600,240 L 0,240 Z',
  bench:  'M 63,137 L 95,140 L 126,145 L 158,155 L 190,158 L 221,152 L 253,148 L 284,155 L 316,152 L 347,145 L 379,148 L 411,153 L 442,150 L 474,145 L 505,140 L 537,143 L 568,147 L 600,145',
  agent:  'M 63,145 L 95,153 L 126,160 L 158,172 L 190,175 L 221,167 L 253,163 L 284,170 L 316,167 L 347,160 L 379,163 L 411,168 L 442,165 L 474,160 L 505,155 L 537,158 L 568,162 L 600,160',
  agentA: 'M 63,145 L 95,153 L 126,160 L 158,172 L 190,175 L 221,167 L 253,163 L 284,170 L 316,167 L 347,160 L 379,163 L 411,168 L 442,165 L 474,160 L 505,155 L 537,158 L 568,162 L 600,160 L 600,240 L 63,240 Z',
  marks:  [[95,92],[221,112],[347,82],[474,68],[505,33]],
}

// ─── Comparison chart (viewBox 0 0 1200 180) ───────────────────────────────────
const CMP = {
  mkt:   'M 0,125 L 63,96  L 126,134 L 190,70  L 253,110 L 316,151 L 379,124 L 442,84  L 505,104 L 568,139 L 632,99  L 695,63  L 758,116 L 821,146 L 884,92  L 947,54  L 1011,29  L 1074,77  L 1137,110 L 1200,81',
  bench: 'M 126,102 L 190,104 L 253,108 L 316,115 L 379,117 L 442,112 L 505,110 L 568,115 L 632,112 L 695,108 L 758,110 L 821,113 L 884,111 L 947,108 L 1011,104 L 1074,106 L 1137,109 L 1200,108',
  twap:  'M 126,105 L 190,108 L 253,112 L 316,118 L 379,120 L 442,115 L 505,113 L 568,117 L 632,115 L 695,110 L 758,113 L 821,116 L 884,113 L 947,110 L 1011,105 L 1074,108 L 1137,111 L 1200,110',
  ppo:   'M 126,112 L 190,117 L 253,123 L 316,132 L 379,135 L 442,128 L 505,125 L 568,131 L 632,127 L 695,122 L 758,125 L 821,129 L 884,126 L 947,122 L 1011,117 L 1074,119 L 1137,123 L 1200,122',
  llm:   'M 126,124 L 190,129 L 253,135 L 316,143 L 379,146 L 442,139 L 505,136 L 568,142 L 632,138 L 695,132 L 758,135 L 821,139 L 884,136 L 947,132 L 1011,126 L 1074,128 L 1137,132 L 1200,130',
}

// ─── Fills ────────────────────────────────────────────────────────────────────
const FILLS = [
  { time: '09:32:14', qty: '500 sh',   price: '$145.21', delta: '−0.8 bps' },
  { time: '09:35:02', qty: '750 sh',   price: '$145.15', delta: '−1.4 bps' },
  { time: '09:38:47', qty: '500 sh',   price: '$145.08', delta: '−2.1 bps' },
  { time: '09:41:33', qty: '1,000 sh', price: '$145.12', delta: '−1.7 bps' },
  { time: '09:44:22', qty: '750 sh',   price: '$145.19', delta: '−1.0 bps' },
]

// ─── Hero product panel ────────────────────────────────────────────────────────
function HeroPanel() {
  return (
    <div
      className="relative w-full overflow-hidden rounded-[10px]"
      style={{ boxShadow: '0 0 140px -20px rgba(174,147,87,0.40)' }}
    >
      {/* Chart — full-bleed black, metric overlaid top-left */}
      <div className="relative bg-[#000000]">
        <div className="absolute left-5 top-5 z-10">
          <p className="text-[10px] font-medium uppercase tracking-[0.07em] text-[#464853]">Slippage vs. VWAP</p>
          <p
            className="mt-1 text-[44px] font-light leading-none text-[#cc9166]"
            style={{ letterSpacing: '-0.025em', fontVariantNumeric: 'tabular-nums' }}
          >
            −2.1 bps
          </p>
        </div>

        <svg viewBox="0 0 600 260" className="w-full" style={{ height: 300, display: 'block' }} preserveAspectRatio="none">
          <defs>
            <linearGradient id="hMoltGold" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#fff0cc" stopOpacity="0.95" />
              <stop offset="20%"  stopColor="#ae9357" stopOpacity="0.80" />
              <stop offset="55%"  stopColor="#ae9357" stopOpacity="0.30" />
              <stop offset="100%" stopColor="#ae9357" stopOpacity="0"    />
            </linearGradient>
            <filter id="hGlow">
              <feGaussianBlur stdDeviation="2.5" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          {/* Dashed vertical markers at peaks */}
          {[95,221,347,474,505].map(x => (
            <line key={x} x1={x} y1="0" x2={x} y2="260" stroke="#1e2028" strokeWidth="0.8" strokeDasharray="3 5" />
          ))}
          {/* Subtle horizontal grid */}
          {[70,130,190].map(y => (
            <line key={y} x1="0" y1={y} x2="600" y2={y} stroke="#1a1b20" strokeWidth="0.8" />
          ))}
          {/* Gold fill area — bright at peaks */}
          <path d="M 0,170 L 32,128 L 63,183 L 95,92  L 126,148 L 158,207 L 190,168 L 221,112 L 253,140 L 284,190 L 316,133 L 347,82  L 379,157 L 411,200 L 442,123 L 474,68  L 505,33  L 537,102 L 568,148 L 600,107 L 600,260 L 0,260 Z" fill="url(#hMoltGold)" />
          {/* White glowing line */}
          <path d={P.mkt} fill="none" stroke="#d4d6dd" strokeWidth="1.5" filter="url(#hGlow)" />
          {/* Gold peak markers */}
          {P.marks.map(([x,y], i) => (
            <circle key={i} cx={x} cy={y} r="4" fill="#cc9166" opacity="0.95" />
          ))}
        </svg>
      </div>

      {/* Time axis */}
      <div className="flex justify-between bg-[#000000] px-4 pb-3 pt-1">
        {['09:30','11:00','12:30','14:00','16:00'].map(t => (
          <span key={t} className="text-[11px] font-light text-[#464853]">{t}</span>
        ))}
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────
export default function Home() {
  const navigate = useNavigate()

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
                    style={{ fontFamily: "'Playfair Display', Georgia, serif",
                             fontSize: 'clamp(44px, 4.5vw, 64px)', fontWeight: 500, fontStyle: 'italic' }}>
                in every trade.
              </span>
            </h1>
            <p className="mb-8 max-w-[420px] text-[15px] font-light leading-[1.7] tracking-[-0.015em] text-[#777a88]">
              Dump a large order at once and you move the price against yourself.
              Spread it out wrong and you eat the day's drift. Velora benchmarks agents —
              classical RL and LLM — against the metric real desks live by.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <button
                onClick={() => navigate('/execution-demo')}
                className="rounded-[2px] bg-white px-5 py-[10px] text-[14px] font-medium text-[#08080a] transition-opacity hover:opacity-90"
              >
                Watch live demo →
              </button>
              <span className="text-[13px] font-light tracking-[-0.01em] text-[#464853]">
                Slippage vs. VWAP · Real OHLCV data
              </span>
            </div>
          </div>
          <HeroPanel />
        </div>
      </section>

      {/* ── Stats strip ──────────────────────────────────────────────────────── */}
      <section className="border-y border-[#2e3038]">
        <div className="mx-auto max-w-6xl grid grid-cols-3 divide-x divide-[#2e3038]">
          {[
            { val: '3',          sub: 'Agent types: TWAP · PPO · LLM' },
            { val: 'VWAP',       sub: 'Industry-standard slippage metric' },
            { val: 'Real OHLCV', sub: 'Live Alpaca market data' },
          ].map(({ val, sub }) => (
            <div key={val} className="group cursor-default px-10 py-8 text-center transition-colors hover:bg-[#08080a]">
              <p className="text-[22px] font-semibold text-[#e2e3e9] transition-colors group-hover:text-white">{val}</p>
              <p className="mt-1 text-[12px] font-light tracking-[-0.01em] text-[#5e616e]">{sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Environment section — NO BOXES ────────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-8 py-20">

        {/* Section header */}
        <div className="mb-16">
          <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.08em] text-[#5e616e]">The environment</p>
          <h2 className="text-[36px] font-light leading-[1.15] text-[#e2e3e9]" style={{ letterSpacing: '-0.018em' }}>
            Everything the execution desk{' '}
            <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400, color: '#cc9166' }}>
              actually needs.
            </span>
          </h2>
        </div>

        {/* 2-column — chart | fills — no card boxes */}
        <div className="grid grid-cols-1 gap-20 md:grid-cols-2 mb-24">

          {/* ── Chart column ── */}
          <div>
            {/* Big metric */}
            <div className="mb-6">
              <p className="text-[11px] font-medium uppercase tracking-[0.07em] text-[#5e616e]">Slippage vs. VWAP</p>
              <p className="mt-1.5 text-[42px] font-light leading-none tracking-[-0.025em] text-[#cc9166]"
                 style={{ fontVariantNumeric: 'tabular-nums' }}>
                −2.1 bps
              </p>
            </div>

            {/* Full-bleed chart — no box */}
            <div className="relative" style={{ height: 260 }}>
              {/* Radial gold glow from chart area */}
              <div className="pointer-events-none absolute inset-0"
                   style={{ background: 'radial-gradient(ellipse 85% 65% at 60% 95%, rgba(174,147,87,0.22) 0%, transparent 65%)' }} />
              <svg viewBox="0 0 600 240" className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="moltGold" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor="#fff0cc" stopOpacity="0.95" />
                    <stop offset="20%"  stopColor="#ae9357" stopOpacity="0.80" />
                    <stop offset="55%"  stopColor="#ae9357" stopOpacity="0.28" />
                    <stop offset="100%" stopColor="#ae9357" stopOpacity="0"    />
                  </linearGradient>
                  <filter id="gChartGlow">
                    <feGaussianBlur stdDeviation="2.5" result="blur" />
                    <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                  </filter>
                </defs>
                {/* Dashed vertical markers */}
                {[95,221,347,474,505].map(x => (
                  <line key={x} x1={x} y1="0" x2={x} y2="240" stroke="#2e3038" strokeWidth="0.5" strokeDasharray="3 4" />
                ))}
                {[80,160].map(y => <line key={y} x1="0" y1={y} x2="600" y2={y} stroke="#2e3038" strokeWidth="0.5" />)}
                {/* Market price area — gold concentrates at the high spikes */}
                <path d={P.mktA} fill="url(#moltGold)" />
                {/* Market price line — near-white with glow, like Slash */}
                <path d={P.mkt}  fill="none" stroke="#d4d6dd" strokeWidth="1.5" filter="url(#gChartGlow)" />
                {/* Gold accent dots at peaks */}
                {P.marks.map(([x,y], i) => (
                  <circle key={i} cx={x} cy={y} r="5" fill="#cc9166" opacity="0.95" />
                ))}
              </svg>
            </div>

            {/* Time axis */}
            <div className="mt-2 flex justify-between">
              {['09:30','11:00','12:30','14:00','16:00'].map(t => (
                <span key={t} className="text-[11px] font-light text-[#464853]">{t}</span>
              ))}
            </div>

            {/* Caption */}
            <div className="mt-8 border-t border-[#2e3038] pt-6">
              <p className="text-[18px] font-medium tracking-[-0.015em] text-[#e2e3e9]">Benchmarked against VWAP</p>
              <p className="mt-2 text-[14px] font-light leading-[1.65] tracking-[-0.012em] text-[#777a88]">
                Every execution slice scored against the industry-standard benchmark in real time.
                Gold dots mark individual fill events.
              </p>
            </div>
          </div>

          {/* ── Fills column — no card boxes ── */}
          <div>
            {/* Big metric */}
            <div className="mb-6">
              <p className="text-[11px] font-medium uppercase tracking-[0.07em] text-[#5e616e]">Episode complete</p>
              <p className="mt-1.5 text-[42px] font-light leading-none tracking-[-0.025em] text-[#cc9166]"
                 style={{ fontVariantNumeric: 'tabular-nums' }}>
                −4.8 bps
              </p>
              <p className="mt-1 text-[13px] font-light tracking-[-0.01em] text-[#777a88]">vs. VWAP benchmark</p>
            </div>

            {/* Fills list — bare rows, Slash transaction-row style */}
            <div className="border-t border-[#2e3038]">
              <div className="flex items-center justify-between py-2.5">
                <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[#464853]">Execution fills · AAPL Buy</span>
                <span className="text-[11px] font-light text-[#464853]">10,000 sh total</span>
              </div>
              {FILLS.map(({ time, qty, price, delta }) => (
                <div
                  key={time}
                  className="group flex items-center justify-between border-t border-[#2e3038] py-3 transition-all duration-150 hover:border-[#cc9166]/25 hover:bg-[#cc9166]/[0.03] -mx-2 px-2 rounded-[2px]"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-[12px] font-medium text-[#464853]" style={{ fontVariantNumeric: 'tabular-nums' }}>{time}</span>
                    <span className="text-[12px] font-light tracking-[-0.01em] text-[#9194a1]">{qty}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-[13px] font-medium text-[#e2e3e9]" style={{ fontVariantNumeric: 'tabular-nums' }}>{price}</span>
                    <span className="min-w-[70px] text-right text-[12px] font-medium text-[#cc9166]" style={{ fontVariantNumeric: 'tabular-nums' }}>{delta}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Caption */}
            <div className="mt-8 border-t border-[#2e3038] pt-6">
              <p className="text-[18px] font-medium tracking-[-0.015em] text-[#e2e3e9]">Fill-by-fill tracking</p>
              <p className="mt-2 text-[14px] font-light leading-[1.65] tracking-[-0.012em] text-[#777a88]">
                Every execution slice captured, priced, and benchmarked against the real-time market VWAP.
              </p>
            </div>
          </div>
        </div>

        {/* ── Full-width comparison — no box ──────────────────────────────────── */}
        <div>
          {/* Header row */}
          <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="text-[11px] font-medium uppercase tracking-[0.07em] text-[#5e616e]">Agent comparison</p>
              <p className="mt-1 text-[36px] font-light leading-[1.1] tracking-[-0.02em] text-[#e2e3e9]">
                Three strategies.{' '}
                <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400, color: '#cc9166' }}>
                  One benchmark.
                </span>
              </p>
            </div>
            <div className="flex items-center gap-8">
              {[
                { label: 'TWAP',       value: '−0.5 bps', color: '#464853' },
                { label: 'PPO',        value: '+2.1 bps', color: '#acafb9' },
                { label: 'Claude LLM', value: '+4.8 bps', color: '#cc9166' },
              ].map(({ label, value, color }) => (
                <div key={label}>
                  <p className="text-[11px] font-medium uppercase tracking-[0.06em]" style={{ color }}>{label}</p>
                  <p className="mt-0.5 text-[18px] font-light" style={{ color, fontVariantNumeric: 'tabular-nums' }}>{value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Full-bleed chart — no box */}
          <div className="relative" style={{ height: 220 }}>
            <div className="pointer-events-none absolute inset-0"
                 style={{ background: 'radial-gradient(ellipse 65% 90% at 50% 100%, rgba(174,147,87,0.14) 0%, transparent 55%)' }} />
            <svg viewBox="0 0 1200 180" className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
              <defs>
                <linearGradient id="llmMolt" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#fff0cc" stopOpacity="0.55" />
                  <stop offset="35%"  stopColor="#ae9357" stopOpacity="0.35" />
                  <stop offset="100%" stopColor="#ae9357" stopOpacity="0"    />
                </linearGradient>
                <filter id="llmGlw">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>
              {/* Dashed vertical grid */}
              {[300,600,900].map(x => (
                <line key={x} x1={x} y1="0" x2={x} y2="180" stroke="#2e3038" strokeWidth="0.5" strokeDasharray="3 4" />
              ))}
              {[60,120].map(y => <line key={y} x1="0" y1={y} x2="1200" y2={y} stroke="#2e3038" strokeWidth="0.5" />)}
              {/* Market — barely visible */}
              <path d={CMP.mkt}   fill="none" stroke="#2e3038" strokeWidth="0.6" />
              {/* Benchmark dashed */}
              <path d={CMP.bench} fill="none" stroke="#5e616e" strokeWidth="1" strokeDasharray="4 3" opacity="0.5" />
              {/* TWAP */}
              <path d={CMP.twap}  fill="none" stroke="#464853" strokeWidth="1.5" />
              {/* PPO */}
              <path d={CMP.ppo}   fill="none" stroke="#acafb9" strokeWidth="2" />
              {/* LLM — molten gold area + line */}
              <path d={`${CMP.llm} L 1200,180 L 126,180 Z`} fill="url(#llmMolt)" />
              <path d={CMP.llm}   fill="none" stroke="#cc9166" strokeWidth="2.5" filter="url(#llmGlw)" />
            </svg>
          </div>

          {/* Axis labels */}
          <div className="mt-2 flex justify-between">
            {['09:30','11:00','12:30','14:00','16:00'].map(t => (
              <span key={t} className="text-[11px] font-light text-[#464853]">{t}</span>
            ))}
          </div>

          {/* Caption row */}
          <div className="mt-8 flex flex-wrap items-center justify-between gap-4 border-t border-[#2e3038] pt-6">
            <div className="flex flex-wrap gap-2">
              {['TWAP', 'PPO · RL', 'Claude LLM'].map(tag => (
                <span key={tag} className="rounded-full border border-[#464853] px-3 py-1 text-[12px] font-medium text-[#9194a1]">
                  {tag}
                </span>
              ))}
            </div>
            <p className="max-w-sm text-[14px] font-light leading-[1.65] tracking-[-0.012em] text-[#777a88]">
              TWAP sets the floor. PPO learns from the market. Claude reasons about order flow
              in natural language. All scored against identical conditions.
            </p>
          </div>
        </div>
      </section>

      {/* ── Who this is for — 3-column feature cards ─────────────────────────── */}
      <section className="border-t border-[#2e3038]">
        <div className="mx-auto max-w-6xl px-8 py-24">

          {/* Centered heading */}
          <div className="mb-16 text-center">
            <h2 className="mx-auto max-w-3xl text-[48px] font-light leading-[1.1] text-[#e2e3e9]"
                style={{ letterSpacing: '-0.022em' }}>
              Built for every desk that{' '}
              <span style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400, color: '#cc9166' }}>
                lives by the benchmark.
              </span>
            </h2>
            <p className="mx-auto mt-5 max-w-xl text-[15px] font-light leading-[1.65] tracking-[-0.012em] text-[#777a88]">
              Configurable RL and LLM agents, real OHLCV data, and slippage-first metrics —
              built for the people who live and die by execution quality.
            </p>
          </div>

          {/* 3-column feature cards */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">

            {/* Card 1: Execution desks — fills table visual */}
            <div className="group overflow-hidden rounded-[10px] border border-[#2e3038] bg-[#1c1d22] transition-colors hover:border-[#464853]">
              <div className="bg-[#08080a] p-5" style={{ height: 220 }}>
                <p className="mb-3 text-[10px] font-medium uppercase tracking-[0.07em] text-[#464853]">
                  Execution fills · AAPL Buy
                </p>
                <div className="rounded-[6px] border border-[#2e3038] overflow-hidden">
                  {[
                    { time: '09:32', qty: '500 sh',   price: '$145.21', bps: '−0.8 bps' },
                    { time: '09:35', qty: '750 sh',   price: '$145.15', bps: '−1.4 bps' },
                    { time: '09:38', qty: '500 sh',   price: '$145.08', bps: '−2.1 bps' },
                    { time: '09:41', qty: '1,000 sh', price: '$145.12', bps: '−1.7 bps' },
                  ].map(({ time, qty, price, bps }) => (
                    <div key={time} className="flex items-center justify-between border-b border-[#2e3038] last:border-0 px-3 py-2">
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-[#464853]" style={{ fontVariantNumeric: 'tabular-nums' }}>{time}</span>
                        <span className="text-[10px] font-light text-[#5e616e]">{qty}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] text-[#9194a1]" style={{ fontVariantNumeric: 'tabular-nums' }}>{price}</span>
                        <span className="text-[10px] font-medium text-[#cc9166]" style={{ fontVariantNumeric: 'tabular-nums' }}>{bps}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="p-6 pt-5">
                <p className="mb-2 text-[15px] font-semibold tracking-[-0.012em] text-[#e2e3e9]">Execution desks & prop firms</p>
                <p className="text-[13px] font-light leading-[1.65] tracking-[-0.01em] text-[#777a88]">
                  Measured on slippage vs. VWAP every day. Stress-test RL and LLM agents against
                  real market data before you commit capital.
                </p>
              </div>
            </div>

            {/* Card 2: Quant research — agent comparison bars */}
            <div className="group overflow-hidden rounded-[10px] border border-[#2e3038] bg-[#1c1d22] transition-colors hover:border-[#464853]">
              <div className="bg-[#08080a] p-5" style={{ height: 220 }}>
                <p className="mb-4 text-[10px] font-medium uppercase tracking-[0.07em] text-[#464853]">
                  Agent performance vs. VWAP
                </p>
                <div className="space-y-4">
                  {[
                    { label: 'Claude LLM', value: 4.8, pct: 96, color: '#cc9166' },
                    { label: 'PPO · RL',   value: 2.1, pct: 58, color: '#acafb9' },
                    { label: 'TWAP',       value: 0.5, pct: 18, color: '#464853' },
                  ].map(({ label, value, pct, color }) => (
                    <div key={label}>
                      <div className="mb-1.5 flex items-center justify-between">
                        <span className="text-[11px] font-medium" style={{ color }}>{label}</span>
                        <span className="text-[11px] font-medium" style={{ color, fontVariantNumeric: 'tabular-nums' }}>
                          {value > 0 ? '+' : ''}{value} bps
                        </span>
                      </div>
                      <div className="h-[3px] rounded-full bg-[#2e3038]">
                        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
                      </div>
                    </div>
                  ))}
                </div>
                <p className="mt-5 text-[10px] font-light text-[#464853]">30-episode rolling average · AAPL</p>
              </div>
              <div className="p-6 pt-5">
                <p className="mb-2 text-[15px] font-semibold tracking-[-0.012em] text-[#e2e3e9]">Quant research teams</p>
                <p className="text-[13px] font-light leading-[1.65] tracking-[-0.01em] text-[#777a88]">
                  Configurable, reproducible RL environment on real OHLCV data. Tune participation rates
                  and adverse-move logic without touching live markets.
                </p>
              </div>
            </div>

            {/* Card 3: Strategy developers — sparkline metric visual */}
            <div className="group overflow-hidden rounded-[10px] border border-[#2e3038] bg-[#1c1d22] transition-colors hover:border-[#464853]">
              <div className="relative bg-[#08080a] p-5" style={{ height: 220 }}>
                <p className="text-[10px] font-medium uppercase tracking-[0.07em] text-[#464853]">Slippage · episode history</p>
                <p className="mt-2 text-[36px] font-light leading-none tracking-[-0.025em] text-[#cc9166]"
                   style={{ fontVariantNumeric: 'tabular-nums' }}>
                  −2.3 bps
                </p>
                <p className="mt-1 text-[11px] font-light text-[#777a88]">avg. vs. VWAP benchmark</p>
                {/* Mini sparkline */}
                <div className="absolute bottom-0 left-0 right-0">
                  <svg viewBox="0 0 280 80" className="w-full" style={{ height: 80, display: 'block' }} preserveAspectRatio="none">
                    <defs>
                      <linearGradient id="spkGold" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%"   stopColor="#fff0cc" stopOpacity="0.8" />
                        <stop offset="40%"  stopColor="#ae9357" stopOpacity="0.5" />
                        <stop offset="100%" stopColor="#ae9357" stopOpacity="0"   />
                      </linearGradient>
                    </defs>
                    <path
                      d="M 0,65 L 28,55 L 56,60 L 84,42 L 112,50 L 140,35 L 168,45 L 196,28 L 224,38 L 252,20 L 280,30 L 280,80 L 0,80 Z"
                      fill="url(#spkGold)"
                    />
                    <path
                      d="M 0,65 L 28,55 L 56,60 L 84,42 L 112,50 L 140,35 L 168,45 L 196,28 L 224,38 L 252,20 L 280,30"
                      fill="none" stroke="#cc9166" strokeWidth="1.5"
                    />
                  </svg>
                </div>
              </div>
              <div className="p-6 pt-5">
                <p className="mb-2 text-[15px] font-semibold tracking-[-0.012em] text-[#e2e3e9]">Portfolio managers</p>
                <p className="text-[13px] font-light leading-[1.65] tracking-[-0.01em] text-[#777a88]">
                  Validate execution strategies across episodes before deploying to real portfolios.
                  Track slippage trends, not just single-run snapshots.
                </p>
              </div>
            </div>

          </div>
        </div>
      </section>

    </div>
  )
}
