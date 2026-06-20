import { useNavigate } from 'react-router-dom'
import Navbar from '../components/Navbar'

// ─── SVG path data ─────────────────────────────────────────────────────────────
// All paths share viewBox "0 0 600 240". y-scale: price 144.9→146.1 maps y 220→20.
// Price path (AAPL buy episode, 20 slices):
const P = {
  mkt:   'M 0,170 L 32,128 L 63,183 L 95,92  L 126,148 L 158,207 L 190,168 L 221,112 L 253,140 L 284,190 L 316,133 L 347,82  L 379,157 L 411,200 L 442,123 L 474,68  L 505,33  L 537,102 L 568,148 L 600,107',
  mktA:  'M 0,170 L 32,128 L 63,183 L 95,92  L 126,148 L 158,207 L 190,168 L 221,112 L 253,140 L 284,190 L 316,133 L 347,82  L 379,157 L 411,200 L 442,123 L 474,68  L 505,33  L 537,102 L 568,148 L 600,107 L 600,240 L 0,240 Z',
  bench: 'M 63,137 L 95,140 L 126,145 L 158,155 L 190,158 L 221,152 L 253,148 L 284,155 L 316,152 L 347,145 L 379,148 L 411,153 L 442,150 L 474,145 L 505,140 L 537,143 L 568,147 L 600,145',
  agent: 'M 63,145 L 95,153 L 126,160 L 158,172 L 190,175 L 221,167 L 253,163 L 284,170 L 316,167 L 347,160 L 379,163 L 411,168 L 442,165 L 474,160 L 505,155 L 537,158 L 568,162 L 600,160',
  agentA:'M 63,145 L 95,153 L 126,160 L 158,172 L 190,175 L 221,167 L 253,163 L 284,170 L 316,167 L 347,160 L 379,163 L 411,168 L 442,165 L 474,160 L 505,155 L 537,158 L 568,162 L 600,160 L 600,240 L 63,240 Z',
  // Markers: (x,y) on market price path at key execution slices
  marks: [[95,92],[221,112],[347,82],[474,68],[505,33]],
}

// Comparison chart viewBox "0 0 1200 180"
// Three agent curves: TWAP (steel), PPO (pearl), LLM (ember-gold)
const CMP = {
  mkt:   'M 0,125 L 63,96  L 126,134 L 190,70  L 253,110 L 316,151 L 379,124 L 442,84  L 505,104 L 568,139 L 632,99  L 695,63  L 758,116 L 821,146 L 884,92  L 947,54  L 1011,29  L 1074,77  L 1137,110 L 1200,81',
  bench: 'M 126,102 L 190,104 L 253,108 L 316,115 L 379,117 L 442,112 L 505,110 L 568,115 L 632,112 L 695,108 L 758,110 L 821,113 L 884,111 L 947,108 L 1011,104 L 1074,106 L 1137,109 L 1200,108',
  twap:  'M 126,105 L 190,108 L 253,112 L 316,118 L 379,120 L 442,115 L 505,113 L 568,117 L 632,115 L 695,110 L 758,113 L 821,116 L 884,113 L 947,110 L 1011,105 L 1074,108 L 1137,111 L 1200,110',
  ppo:   'M 126,112 L 190,117 L 253,123 L 316,132 L 379,135 L 442,128 L 505,125 L 568,131 L 632,127 L 695,122 L 758,125 L 821,129 L 884,126 L 947,122 L 1011,117 L 1074,119 L 1137,123 L 1200,122',
  llm:   'M 126,124 L 190,129 L 253,135 L 316,143 L 379,146 L 442,139 L 505,136 L 568,142 L 632,138 L 695,132 L 758,135 L 821,139 L 884,136 L 947,132 L 1011,126 L 1074,128 L 1137,132 L 1200,130',
}

// ─── Hero execution panel ───────────────────────────────────────────────────────
function HeroPanel() {
  return (
    <div
      className="relative w-full overflow-hidden rounded-[10px] border border-[#2e3038] bg-[#1c1d22]"
      style={{ boxShadow: '0 0 120px -30px rgba(174,147,87,0.25), 0 0 0 1px #2e3038' }}
    >
      {/* Header */}
      <div className="flex items-center justify-between border-b border-[#2e3038] px-5 py-3.5">
        <div className="flex items-center gap-2.5">
          <span className="text-[13px] font-semibold text-[#e2e3e9]">AAPL</span>
          <span className="text-[#464853] text-[11px]">·</span>
          <span className="text-[12px] text-[#777a88]">Buy 10,000 sh · TWAP</span>
        </div>
        <span className="flex items-center gap-1.5 text-[12px] font-medium text-[#cc9166]">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#cc9166]" />
          Live
        </span>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-3 divide-x divide-[#2e3038] border-b border-[#2e3038]">
        {[
          { label: 'Slippage vs. VWAP', value: '−2.1 bps', gold: true },
          { label: 'Filled',            value: '65%',       gold: false },
          { label: 'Slice',             value: '13 / 20',   gold: false },
        ].map(({ label, value, gold }) => (
          <div key={label} className="px-5 py-3">
            <p className="text-[10px] font-medium uppercase tracking-[0.07em] text-[#464853]">{label}</p>
            <p className="mt-1 text-[15px] font-semibold" style={{ color: gold ? '#cc9166' : '#e2e3e9', fontVariantNumeric: 'tabular-nums' }}>
              {value}
            </p>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="bg-[#08080a] px-0 pb-0 pt-0" style={{ position: 'relative' }}>
        <svg viewBox="0 0 600 240" className="w-full" style={{ height: 200, display: 'block' }} preserveAspectRatio="none">
          <defs>
            <linearGradient id="hMkt" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#464853" stopOpacity="0.1" />
              <stop offset="100%" stopColor="#464853" stopOpacity="0"   />
            </linearGradient>
            <linearGradient id="hAgent" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#cc9166" stopOpacity="0.35" />
              <stop offset="50%"  stopColor="#ae9357" stopOpacity="0.14" />
              <stop offset="100%" stopColor="#ae9357" stopOpacity="0"    />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          {[60,120,180].map(y => <line key={y} x1="0" y1={y} x2="600" y2={y} stroke="#2e3038" strokeWidth="0.5" />)}
          <path d={P.mktA}   fill="url(#hMkt)" />
          <path d={P.mkt}    fill="none" stroke="#464853" strokeWidth="0.8" />
          <path d={P.bench}  fill="none" stroke="#cc9166" strokeWidth="1.5" strokeDasharray="4 3" />
          <path d={P.agentA} fill="url(#hAgent)" />
          <path d={P.agent}  fill="none" stroke="#acafb9" strokeWidth="2" filter="url(#glow)" />
          {P.marks.map(([x,y], i) => (
            <circle key={i} cx={x} cy={y} r="4" fill="#cc9166" opacity="0.85" filter="url(#glow)" />
          ))}
        </svg>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 border-t border-[#2e3038] bg-[#08080a] px-5 py-2.5">
        {[
          { color:'#5e616e', label:'Market price', dash:false },
          { color:'#cc9166', label:'VWAP benchmark', dash:true },
          { color:'#acafb9', label:'Agent avg. price', dash:false },
        ].map(({ color, label, dash }) => (
          <div key={label} className="flex items-center gap-1.5">
            <svg width="18" height="2"><line x1="0" y1="1" x2="18" y2="1" stroke={color} strokeWidth="1.5" strokeDasharray={dash ? '4 3' : undefined} /></svg>
            <span className="text-[11px] text-[#5e616e]">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── Feature tile: cinematic execution chart ───────────────────────────────────
function ChartTile() {
  return (
    <div className="relative flex flex-col overflow-hidden rounded-[10px] bg-[#08080a]" style={{ minHeight: 420 }}>
      {/* Cinematic chart fills the whole tile */}
      <div className="flex-1 relative">
        {/* Ambient gold glow */}
        <div
          className="pointer-events-none absolute inset-0"
          style={{ background: 'radial-gradient(ellipse 80% 60% at 60% 80%, rgba(174,147,87,0.13) 0%, transparent 70%)' }}
        />
        <svg
          viewBox="0 0 600 240"
          className="w-full h-full"
          style={{ position: 'absolute', inset: 0, height: '100%' }}
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="tMkt" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#464853" stopOpacity="0.12" />
              <stop offset="100%" stopColor="#464853" stopOpacity="0"    />
            </linearGradient>
            <linearGradient id="tAgent" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#cc9166" stopOpacity="0.5"  />
              <stop offset="40%"  stopColor="#ae9357" stopOpacity="0.2"  />
              <stop offset="100%" stopColor="#ae9357" stopOpacity="0"    />
            </linearGradient>
            <filter id="tGlow">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
          {[60,120,180].map(y => <line key={y} x1="0" y1={y} x2="600" y2={y} stroke="#2e3038" strokeWidth="0.5" />)}
          <path d={P.mktA}   fill="url(#tMkt)"   />
          <path d={P.mkt}    fill="none" stroke="#2e3038" strokeWidth="1" />
          <path d={P.bench}  fill="none" stroke="#cc9166" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.7" />
          <path d={P.agentA} fill="url(#tAgent)" />
          <path d={P.agent}  fill="none" stroke="#cc9166" strokeWidth="2.5" filter="url(#tGlow)" />
          {P.marks.map(([x,y], i) => (
            <circle key={i} cx={x} cy={y} r="5" fill="#cc9166" opacity="0.9" filter="url(#tGlow)" />
          ))}
        </svg>
      </div>

      {/* Bottom text */}
      <div className="relative z-10 border-t border-[#2e3038] p-7">
        <p className="mb-2 text-[19px] font-semibold text-[#e2e3e9]">Slippage vs. VWAP</p>
        <p className="text-[14px] leading-[1.6] text-[#777a88]">
          Measure execution quality against the industry-standard benchmark. Every episode scored and logged.
        </p>
      </div>
    </div>
  )
}

// ─── Feature tile: execution fills ────────────────────────────────────────────
const FILLS = [
  { time: '09:32:14', qty: '500 sh',   price: '$145.21', delta: '−0.8 bps' },
  { time: '09:35:02', qty: '750 sh',   price: '$145.15', delta: '−1.4 bps' },
  { time: '09:38:47', qty: '500 sh',   price: '$145.08', delta: '−2.1 bps' },
  { time: '09:41:33', qty: '1,000 sh', price: '$145.12', delta: '−1.7 bps' },
  { time: '09:44:22', qty: '750 sh',   price: '$145.19', delta: '−1.0 bps' },
]

function FillsTile() {
  return (
    <div className="relative flex flex-col overflow-hidden rounded-[10px] bg-[#08080a]" style={{ minHeight: 420 }}>
      {/* Subtle background glow */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: 'radial-gradient(ellipse 70% 50% at 50% 30%, rgba(46,48,56,0.6) 0%, transparent 70%)' }}
      />

      {/* Notification card */}
      <div className="relative z-10 p-6 pt-8">
        <div className="rounded-[10px] border border-[#2e3038] bg-[#1c1d22] p-4"
             style={{ boxShadow: '0 8px 32px rgba(0,0,0,0.5)' }}>
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-[13px]">🎯</span>
              <span className="text-[13px] font-medium text-[#e2e3e9]">Episode complete</span>
            </div>
            <span className="text-[11px] text-[#5e616e]">just now</span>
          </div>
          <p className="text-[28px] font-semibold text-[#cc9166]" style={{ fontVariantNumeric: 'tabular-nums' }}>
            −4.8 bps
          </p>
          <p className="mt-0.5 text-[12px] text-[#777a88]">vs. VWAP benchmark</p>
        </div>
      </div>

      {/* Fills list */}
      <div className="relative z-10 flex-1 px-6 pb-2">
        <div className="rounded-[10px] border border-[#2e3038] bg-[#1c1d22] overflow-hidden">
          <div className="border-b border-[#2e3038] px-4 py-3">
            <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-[#5e616e]">
              Execution fills · AAPL Buy
            </p>
          </div>
          {FILLS.map(({ time, qty, price, delta }) => (
            <div key={time} className="flex items-center justify-between px-4 py-2.5 border-b border-[#2e3038] last:border-0">
              <div className="flex items-center gap-3">
                <span className="text-[11px] font-medium text-[#464853]" style={{ fontVariantNumeric: 'tabular-nums' }}>{time}</span>
                <span className="text-[12px] text-[#9194a1]">{qty}</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="text-[12px] font-medium text-[#e2e3e9]" style={{ fontVariantNumeric: 'tabular-nums' }}>{price}</span>
                <span className="text-[11px] font-medium text-[#cc9166]" style={{ fontVariantNumeric: 'tabular-nums' }}>{delta}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom text */}
      <div className="relative z-10 border-t border-[#2e3038] p-7">
        <p className="mb-2 text-[19px] font-semibold text-[#e2e3e9]">Fill-by-fill tracking</p>
        <p className="text-[14px] leading-[1.6] text-[#777a88]">
          Every execution slice captured, priced, and benchmarked against the real-time market VWAP.
        </p>
      </div>
    </div>
  )
}

// ─── Feature tile: full-width agent comparison ─────────────────────────────────
function ComparisonTile() {
  return (
    <div className="relative overflow-hidden rounded-[10px] bg-[#08080a]" style={{ minHeight: 420 }}>
      {/* Ambient glow */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: 'radial-gradient(ellipse 60% 80% at 50% 100%, rgba(174,147,87,0.1) 0%, transparent 60%)' }}
      />

      {/* Chart fills upper portion */}
      <div className="relative" style={{ height: 260 }}>
        <svg
          viewBox="0 0 1200 180"
          className="w-full h-full"
          style={{ position: 'absolute', inset: 0 }}
          preserveAspectRatio="none"
        >
          <defs>
            <linearGradient id="cMkt" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#2e3038" stopOpacity="0.4"  />
              <stop offset="100%" stopColor="#2e3038" stopOpacity="0"    />
            </linearGradient>
            <linearGradient id="cLLM" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#cc9166" stopOpacity="0.3"  />
              <stop offset="100%" stopColor="#cc9166" stopOpacity="0"    />
            </linearGradient>
            <filter id="cGlow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>

          {/* Grid */}
          {[45,90,135].map(y => <line key={y} x1="0" y1={y} x2="1200" y2={y} stroke="#2e3038" strokeWidth="0.5" />)}

          {/* Market price */}
          <path d={CMP.mkt} fill="none" stroke="#2e3038" strokeWidth="0.8" />

          {/* TWAP — steel, barely above benchmark */}
          <path d={CMP.twap} fill="none" stroke="#464853" strokeWidth="1.5" />

          {/* VWAP benchmark dashed */}
          <path d={CMP.bench} fill="none" stroke="#9194a1" strokeWidth="1" strokeDasharray="4 3" opacity="0.5" />

          {/* PPO — pearl */}
          <path d={CMP.ppo}  fill="none" stroke="#acafb9" strokeWidth="2" />

          {/* LLM — ember gold with glow, area fill */}
          <path d={`${CMP.llm} L 1200,180 L 126,180 Z`} fill="url(#cLLM)" />
          <path d={CMP.llm}  fill="none" stroke="#cc9166" strokeWidth="2.5" filter="url(#cGlow)" />

          {/* End labels */}
          <text x="1205" y="113" fill="#464853" fontSize="10" dominantBaseline="middle" fontFamily="Inter, sans-serif">TWAP</text>
          <text x="1205" y="125" fill="#acafb9" fontSize="10" dominantBaseline="middle" fontFamily="Inter, sans-serif">PPO</text>
          <text x="1205" y="133" fill="#cc9166" fontSize="10" dominantBaseline="middle" fontFamily="Inter, sans-serif" fontWeight="600">LLM</text>
        </svg>

        {/* Floating metric chips */}
        <div className="absolute right-8 top-6 flex flex-col gap-2 z-10">
          {[
            { label: 'TWAP', value: '−0.5 bps', color: '#464853' },
            { label: 'PPO',  value: '+2.1 bps', color: '#acafb9' },
            { label: 'Claude LLM', value: '+4.8 bps', color: '#cc9166' },
          ].map(({ label, value, color }) => (
            <div key={label} className="flex items-center gap-2 rounded-[6px] border border-[#2e3038] bg-[#1c1d22]/90 px-3 py-1.5 backdrop-blur-sm">
              <span className="text-[11px] font-medium" style={{ color }}>{label}</span>
              <span className="text-[12px] font-semibold" style={{ color, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom text */}
      <div className="relative z-10 border-t border-[#2e3038] p-8 lg:flex lg:items-start lg:justify-between">
        <div>
          <div className="mb-3 flex flex-wrap gap-2">
            {['TWAP', 'PPO · RL', 'Claude LLM'].map(tag => (
              <span key={tag} className="rounded-full border border-[#464853] px-3 py-1 text-[12px] font-medium text-[#9194a1]">
                {tag}
              </span>
            ))}
          </div>
          <p className="text-[19px] font-semibold text-[#e2e3e9]">Three strategies. One benchmark.</p>
        </div>
        <p className="mt-3 max-w-sm text-[14px] leading-[1.65] text-[#777a88] lg:mt-0">
          TWAP sets the floor. PPO learns from the market. Claude reasons about order flow in
          natural language. All scored against identical conditions.
        </p>
      </div>
    </div>
  )
}

// ─── Page ───────────────────────────────────────────────────────────────────────
export default function Home() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-[#000000] text-[#e2e3e9]" style={{ fontFamily: "'Inter', sans-serif" }}>

      {/* Announcement bar */}
      <div className="flex items-center justify-center gap-2 border-b border-[#2e3038] bg-[#08080a] px-4 py-2.5 text-center">
        <span className="text-[12px] text-[#777a88]">
          Now running Claude Sonnet 4 in the live execution demo
        </span>
        <button
          onClick={() => navigate('/execution-demo')}
          className="text-[12px] font-medium text-[#cc9166] transition-opacity hover:opacity-80"
        >
          Watch it trade →
        </button>
      </div>

      <Navbar />

      {/* ── Hero ─────────────────────────────────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-8 pb-8 pt-20">
        <div className="grid grid-cols-1 items-center gap-14 lg:grid-cols-[1fr_1.4fr]">

          {/* Text */}
          <div>
            <h1 className="mb-5">
              <span className="block font-light leading-[1.08] tracking-[-0.02em] text-[#e2e3e9]" style={{ fontSize: 'clamp(40px, 4vw, 56px)' }}>
                A sharper edge
              </span>
              <span
                className="block leading-[1.0] tracking-[-0.02em] text-[#cc9166]"
                style={{
                  fontFamily: "'Playfair Display', Georgia, serif",
                  fontSize: 'clamp(44px, 4.5vw, 64px)',
                  fontWeight: 500,
                  fontStyle: 'italic',
                }}
              >
                in every trade.
              </span>
            </h1>

            <p className="mb-8 max-w-[420px] text-[15px] leading-[1.7] text-[#777a88]">
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
              <span className="text-[13px] text-[#464853]">Slippage vs. VWAP · Real OHLCV data</span>
            </div>
          </div>

          {/* Hero panel */}
          <HeroPanel />
        </div>
      </section>

      {/* ── Stats strip ──────────────────────────────────────────────────────── */}
      <section className="border-y border-[#2e3038]">
        <div className="mx-auto max-w-6xl grid grid-cols-3 divide-x divide-[#2e3038] px-0">
          {[
            { val: '3',          sub: 'Agent types: TWAP · PPO · LLM' },
            { val: 'VWAP',       sub: 'Industry-standard slippage metric' },
            { val: 'Real OHLCV', sub: 'Live Alpaca market data' },
          ].map(({ val, sub }) => (
            <div key={val} className="px-10 py-8 text-center">
              <p className="text-[22px] font-semibold text-[#e2e3e9]">{val}</p>
              <p className="mt-1 text-[12px] text-[#5e616e]">{sub}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Foundation tiles ──────────────────────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-8 py-16">
        <div className="mb-10">
          <p className="mb-3 text-[11px] font-medium uppercase tracking-[0.08em] text-[#5e616e]">The environment</p>
          <h2
            className="text-[36px] font-light leading-[1.15] text-[#e2e3e9]"
            style={{ letterSpacing: '-0.018em' }}
          >
            Everything the execution desk{' '}
            <span
              style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400, color: '#cc9166' }}
            >
              actually needs.
            </span>
          </h2>
        </div>

        {/* 2-up tiles */}
        <div className="mb-4 grid grid-cols-1 gap-4 md:grid-cols-2">
          <ChartTile />
          <FillsTile />
        </div>

        {/* Full-width comparison tile */}
        <ComparisonTile />
      </section>

      {/* ── Who this is for ──────────────────────────────────────────────────── */}
      <section className="border-t border-[#2e3038]">
        <div className="mx-auto max-w-6xl px-8 py-20">
          <div className="grid grid-cols-1 gap-12 lg:grid-cols-2 lg:items-start">
            <div>
              <h2
                className="mb-4 text-[36px] font-light leading-[1.15] text-[#e2e3e9]"
                style={{ letterSpacing: '-0.018em' }}
              >
                Built for the desks that live
                <br />
                <span
                  style={{ fontFamily: "'Playfair Display', Georgia, serif", fontStyle: 'italic', fontWeight: 400, color: '#cc9166' }}
                >
                  and die by slippage.
                </span>
              </h2>
              <div className="flex flex-wrap gap-2 pt-2">
                {['Execution desks', 'Prop trading', 'Asset managers', 'Quant research'].map(t => (
                  <span key={t} className="rounded-full border border-[#2e3038] px-3 py-1 text-[12px] text-[#777a88]">{t}</span>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              {[
                {
                  h: 'Execution desks & prop firms',
                  b: 'Measured on slippage vs. VWAP every day. Velora gives you a sandbox to stress-test RL and LLM agents against historical market data before you commit capital.'
                },
                {
                  h: 'Quant research teams',
                  b: 'A configurable, reproducible RL environment built on real OHLCV data. Tune participation rates, schedule primitives, and adverse-move pause logic without touching live markets.'
                },
              ].map(({ h, b }) => (
                <div key={h} className="rounded-[10px] border border-[#2e3038] bg-[#1c1d22] p-6">
                  <p className="mb-2 text-[15px] font-semibold text-[#e2e3e9]">{h}</p>
                  <p className="text-[14px] leading-[1.65] text-[#777a88]">{b}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
