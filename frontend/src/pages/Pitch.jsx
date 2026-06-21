import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, ArrowRight, ExternalLink, StickyNote, X } from 'lucide-react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, Cell,
} from 'recharts'
import { ppoPooled, ppoHoldout, hudEval, headlineStats } from '@/data/benchmarks'

// ─── shared bits ──────────────────────────────────────────────────────────────

const GOLD = '#cc9166'
const PEARL = '#acafb9'
const DIM = '#5e616e'
const FAINT = '#464853'

function Kicker({ children }) {
  return (
    <p className="text-[11px] font-medium uppercase tracking-[0.12em]" style={{ color: GOLD }}>
      {children}
    </p>
  )
}

function Title({ children }) {
  return (
    <h1 className="mt-3 text-[40px] font-light leading-[1.08] text-white" style={{ letterSpacing: '-0.02em' }}>
      {children}
    </h1>
  )
}

function Sub({ children }) {
  return <p className="mt-4 max-w-2xl text-[15px] font-light leading-relaxed" style={{ color: DIM }}>{children}</p>
}

function Stat({ label, value, sub }) {
  return (
    <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] px-5 py-4">
      <p className="text-[10px] font-medium uppercase tracking-[0.07em]" style={{ color: FAINT }}>{label}</p>
      <p className="mt-1 text-[26px] font-light text-white" style={{ fontVariantNumeric: 'tabular-nums' }}>{value}</p>
      {sub && <p className="mt-0.5 text-[12px] font-light" style={{ color: DIM }}>{sub}</p>}
    </div>
  )
}

function Callout({ tone = 'gold', children }) {
  const border = tone === 'gold' ? 'border-[#cc9166]/25' : 'border-white/10'
  const bg = tone === 'gold' ? 'bg-[#cc9166]/[0.06]' : 'bg-white/[0.03]'
  return (
    <div className={`mt-6 rounded-[10px] border ${border} ${bg} px-5 py-4`}>
      <p className="text-[13px] font-light leading-relaxed" style={{ color: tone === 'gold' ? '#e8c9a8' : DIM }}>
        {children}
      </p>
    </div>
  )
}

function Table({ rows, cols }) {
  return (
    <div className="mt-6 overflow-hidden rounded-[10px] border border-white/[0.06]">
      <table className="w-full text-left text-[13px]">
        <thead>
          <tr className="border-b border-white/[0.06] bg-white/[0.02]">
            {cols.map((c) => (
              <th key={c} className="px-4 py-2.5 font-medium uppercase tracking-[0.06em]" style={{ color: FAINT, fontSize: 10 }}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-white/[0.04] last:border-0">
              {r.map((cell, j) => (
                <td key={j} className="px-4 py-3 font-light" style={{ color: j === 0 ? '#d8d9de' : DIM }}>
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Pill({ children }) {
  return (
    <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-[11px] font-light" style={{ color: DIM }}>
      {children}
    </span>
  )
}

// ─── slides ───────────────────────────────────────────────────────────────────

// A representative intraday price path, purely decorative — same hand-drawn style as the
// landing page hero, just here to fill the title slide's empty lower half.
const TITLE_MKT =
  'M 0,210 L 60,178 L 120,222 L 180,150 L 240,196 L 300,236 L 360,184 L 420,128 L 480,160 ' +
  'L 540,206 L 600,142 L 660,98 L 720,150 L 780,210 L 840,232 L 900,164 L 960,108 L 1020,52 L 1080,90'

function Slide1() {
  return (
    <div className="relative flex h-full flex-col items-center justify-center text-center">
      <svg
        className="pointer-events-none absolute inset-x-0 bottom-0 h-[46%] w-full"
        viewBox="0 0 1080 260"
        preserveAspectRatio="none"
      >
        <defs>
          <linearGradient id="titleFade" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={GOLD} stopOpacity="0.22" />
            <stop offset="100%" stopColor={GOLD} stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${TITLE_MKT} L 1080,260 L 0,260 Z`} fill="url(#titleFade)" />
        <path d={TITLE_MKT} fill="none" stroke={GOLD} strokeOpacity="0.45" strokeWidth="1.5" />
      </svg>

      <Kicker>HUD Hackathon · Execution RL</Kicker>
      <h1 className="mt-4 text-[64px] font-light leading-none text-white" style={{ letterSpacing: '-0.03em' }}>
        Velora
      </h1>
      <p className="mt-3 text-[18px] font-light" style={{ color: '#d8d9de' }}>
        RL environments for optimal trade execution
      </p>
      <p className="mt-6 max-w-xl text-[14px] font-light italic" style={{ color: DIM }}>
        "Prove it on held-out market data. Reproduce it live."
      </p>
      <div className="mt-10 flex gap-3">
        <Pill>Team Velora · Bhargava Perumalla, Dheeraj Tallapragada</Pill>
        <Pill>HUD × Modal × Fireworks</Pill>
      </div>

      <div className="relative mt-16 grid grid-cols-4 gap-3">
        <Stat label="Mean advantage" value={`+${ppoPooled.meanAdvantageBps.toFixed(1)} bps`} sub="PPO vs VWAP-match" />
        <Stat label="Win rate" value={`${Math.round(ppoPooled.winRate * 100)}%`} sub={`paired, n=${ppoPooled.nEpisodes}`} />
        <Stat label="t-statistic" value={ppoPooled.tStat.toFixed(1)} sub="held-out, statistically significant" />
        <Stat label="Tickers trained" value="51" sub="Modal · 2.03M timesteps" />
      </div>
    </div>
  )
}

function Slide2() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>The Problem</Kicker>
      <Title>Institutional desks execute $100M+ orders daily.</Title>
      <Sub>Slippage vs. VWAP — the volume-weighted average price — is how execution quality is graded on real trading desks.</Sub>

      <div className="mt-6 rounded-[10px] border border-[#cc9166]/20 bg-[#cc9166]/[0.05] p-4">
        <p className="text-[10px] font-medium uppercase tracking-[0.07em]" style={{ color: GOLD }}>Example</p>
        <p className="mt-1.5 text-[14px] font-light leading-relaxed" style={{ color: '#e8c9a8' }}>
          A desk needs to buy 500,000 shares of NVDA — about 6% of its average daily volume — before the close. Every
          slice is a decision: trade more now, or wait?
        </p>
      </div>

      <p className="mt-6 text-[11px] font-medium uppercase tracking-[0.07em]" style={{ color: FAINT }}>
        Why the obvious approaches fail
      </p>
      <div className="mt-3 grid grid-cols-3 gap-3">
        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="text-[13px] font-light text-white">Equal-time slices (TWAP)</p>
          <p className="mt-2 text-[12px] font-light leading-relaxed" style={{ color: DIM }}>
            Trades the same size at 9:35am, when liquidity is thin, as at 3:45pm, when it's thick — pushing the price
            harder than necessary during quiet stretches.
          </p>
        </div>
        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="text-[13px] font-light text-white">Follow the historical curve (VWAP-match)</p>
          <p className="mt-2 text-[12px] font-light leading-relaxed" style={{ color: DIM }}>
            Better, but static — the schedule is fixed before the open and blind to what's actually happening today: a
            volume spike, a headline, a sudden liquidity air-pocket.
          </p>
        </div>
        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="text-[13px] font-light text-white">Rule-based / human desks</p>
          <p className="mt-2 text-[12px] font-light leading-relaxed" style={{ color: DIM }}>
            Can't continuously re-weigh dozens of changing signals across ~26 decision points without either
            over-engineering brittle rules or under-reacting in the moment.
          </p>
        </div>
      </div>

      <Callout>
        What we do instead: train a policy that observes the order's real-time state at every slice and adapts the
        schedule as the day actually unfolds — not a plan committed once and forgotten.
      </Callout>
    </div>
  )
}

function Slide3() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>What We Built</Kicker>
      <Title>A full RL loop: environment → train → eval → deploy → demo</Title>
      <Table
        cols={['Layer', 'What']}
        rows={[
          ['Simulator', 'Real Alpaca SIP data + Almgren-Chriss impact model'],
          ['Gym env', 'ExecutionEnv — closed-loop, slice-by-slice'],
          ['HUD env', 'MCP tools: read_market_context() → submit_schedule() → graded reward'],
          ['Agents', 'PPO (RL), Claude, Fireworks gpt-oss'],
          ['Proof', 'Held-out eval, paired seeds, t-stats'],
          ['Demo', 'Live sandbox, Showdown, Proof pages'],
        ]}
      />
      <Callout>This isn't a mock dashboard — every number on the Proof page is reproducible one-click.</Callout>
    </div>
  )
}

const PROOF_BARS = [
  { name: 'Pooled · 4 tickers, n=480', bps: ppoPooled.meanAdvantageBps, win: ppoPooled.winRate },
  { name: 'AAPL holdout · n=60', bps: ppoHoldout.meanAdvantageBps, win: ppoHoldout.winRate },
]

function Slide4() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>The Proof</Kicker>
      <Title>PPO beats VWAP-match on held-out days it never trained on.</Title>

      <div className="mt-6 grid grid-cols-[1fr_1.1fr] gap-4">
        <div className="grid grid-cols-2 gap-3">
          <Stat label="Mean advantage" value={`+${ppoPooled.meanAdvantageBps.toFixed(1)} bps`} sub={`vs VWAP-match, n=${ppoPooled.nEpisodes}`} />
          <Stat label="Win rate" value={`${Math.round(ppoPooled.winRate * 100)}%`} sub="paired by seed" />
          <Stat label="t-statistic" value={ppoPooled.tStat.toFixed(1)} sub="statistically significant" />
          <Stat label="AAPL example" value={`+${ppoHoldout.meanAdvantageBps.toFixed(0)} bps`} sub={`${Math.round(ppoHoldout.winRate * 100)}% win · 60 eps`} />
        </div>

        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-3">
          <p className="text-[10px] font-medium uppercase tracking-[0.07em]" style={{ color: FAINT }}>
            Mean advantage vs VWAP-match (bps)
          </p>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={PROOF_BARS} margin={{ top: 14, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis dataKey="name" stroke={FAINT} tick={{ fontSize: 10, fill: DIM }} />
              <YAxis stroke={FAINT} tick={{ fontSize: 10, fill: DIM }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
                labelStyle={{ color: '#d8d9de' }}
                formatter={(v) => [`+${v.toFixed(1)} bps`, 'Advantage']}
              />
              <ReferenceLine y={0} stroke={FAINT} />
              <Bar dataKey="bps" radius={[3, 3, 0, 0]}>
                {PROOF_BARS.map((_, i) => <Cell key={i} fill={GOLD} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <Callout>
        Critical honesty line — judges respect this: edge is order-size dependent. It shows at institutional sizes
        (~8% ADV). At 10k-share HUD tasks, impact is tiny and scheduling barely matters.
      </Callout>
    </div>
  )
}

function Slide5() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>Live Demo Storyboard</Kicker>
      <Title>3-minute demo — pick one path</Title>
      <div className="mt-6 grid grid-cols-3 gap-3">
        {[
          { t: 'A · Reproduce the proof', d: 'Proof page → AAPL, 8% ADV, 60 episodes, VWAP-match baseline. Click run, show win rate / mean bps / t-stat live.', q: '"Same seeds, same price path for every policy — no cherry-picking."', time: '~90s · safest' },
          { t: 'B · Head-to-head', d: 'Showdown → TSLA buy, 8% ADV, seed 7. PPO vs VWAP-match slippage chart side-by-side. Point at the participation curve.', q: null, time: '~90s' },
          { t: 'C · HUD agent eval', d: 'Home HUD card → PPO vs Claude scores. Click the hud.ai job links — verifiable traces, not fabricated.', q: `PPO ${hudEval.agents[0].meanScore} vs Claude ${hudEval.agents[1].meanScore} on 3 tasks`, time: '~60s' },
        ].map((p) => (
          <div key={p.t} className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-4">
            <p className="text-[10px] font-medium uppercase tracking-[0.06em]" style={{ color: GOLD }}>{p.time}</p>
            <p className="mt-2 text-[14px] font-light text-white">{p.t}</p>
            <p className="mt-2 text-[12px] font-light leading-relaxed" style={{ color: DIM }}>{p.d}</p>
            {p.q && <p className="mt-2 text-[11px] font-light italic" style={{ color: '#e8c9a8' }}>{p.q}</p>}
          </div>
        ))}
      </div>
      <Callout tone="dim">Backup if demo fails: screenshot + job URLs + "run locally with one command."</Callout>
    </div>
  )
}

function Slide6() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>Two Agents, One Environment</Kicker>
      <Title>Why we built both PPO and LLM agents</Title>
      <div className="mt-6 rounded-[10px] border border-white/[0.08] bg-white/[0.02] p-5 text-center">
        <p className="text-[12px] font-light" style={{ color: DIM }}>Velora RL Environment</p>
        <p className="text-[11px] font-light" style={{ color: FAINT }}>simulator + VWAP reward + MCP tools</p>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-4">
        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="text-[13px] font-light" style={{ color: PEARL }}>PPO + MLP · specialist</p>
          <p className="mt-2 text-[11px] font-medium uppercase tracking-[0.06em]" style={{ color: FAINT }}>Observes, every slice</p>
          <ul className="mt-1.5 space-y-1 text-[12px] font-light" style={{ color: DIM }}>
            <li>Order state — shares remaining, time remaining</li>
            <li>Market state — last-slice return, position in the volume curve, interim slippage so far</li>
            <li>Regime — log average-daily-volume, today's volatility</li>
            <li>Volume surprise — realized volume so far vs. the expected curve</li>
          </ul>
          <p className="mt-2.5 text-[11px] font-medium uppercase tracking-[0.06em]" style={{ color: FAINT }}>Decides</p>
          <p className="mt-1 text-[12px] font-light" style={{ color: DIM }}>
            One number per slice: a participation multiplier from 0 to 2 — 1.0 trades at the textbook rate, above or
            below throttles up or down.
          </p>
        </div>
        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="text-[13px] font-light" style={{ color: GOLD }}>LLM agents · generalist</p>
          <ul className="mt-2 space-y-1 text-[12px] font-light" style={{ color: DIM }}>
            <li>Natural-language constraints</li>
            <li>Generalizes to unseen tickers</li>
            <li>Explainable schedules</li>
            <li>RFT path (Fireworks / HUD)</li>
          </ul>
        </div>
      </div>
      <Callout>
        Differentiated insight from our sanity eval: zero-shot Claude always picks "follow the volume curve" —
        mathematically identical to VWAP-match. That's not a bug; it's the rational open-loop choice with no future
        price info. RFT's job is to beat that trivial floor, not PPO's ceiling.
      </Callout>
    </div>
  )
}

function Slide7() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>What LLMs Are Actually For</Kicker>
      <Title>PPO optimizes numbers. LLMs obey instructions.</Title>
      <div className="mt-6 rounded-[10px] border border-[#cc9166]/20 bg-[#cc9166]/[0.05] p-5">
        <p className="text-[15px] font-light italic leading-relaxed" style={{ color: '#e8c9a8' }}>
          "Work 400k shares of $X by 2pm, stay under 5% participation, de-risk before earnings at noon."
        </p>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="text-[13px] font-light" style={{ color: PEARL }}>PPO</p>
          <p className="mt-2 text-[12px] font-light" style={{ color: DIM }}>No input channel for any of that.</p>
        </div>
        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-4">
          <p className="text-[13px] font-light" style={{ color: GOLD }}>RFT'd LLM</p>
          <p className="mt-2 text-[12px] font-light" style={{ color: DIM }}>Reads it, schedules, explains why.</p>
        </div>
      </div>

      <p className="mt-5 text-[11px] font-medium uppercase tracking-[0.07em]" style={{ color: FAINT }}>
        More constraints a PM hands the desk
      </p>
      <div className="mt-2 space-y-1.5">
        {[
          'Cap participation at 5% so we don’t move the tape.',
          'Pause trading for 15 minutes around the 10am CPI print.',
          'If price moves more than 2% against us, slow down and flag it.',
          'Explain why you front-loaded the first hour.',
        ].map((c) => (
          <div key={c} className="rounded-[10px] border border-white/[0.05] bg-white/[0.015] px-4 py-2 text-[12px] font-light" style={{ color: DIM }}>
            {c}
          </div>
        ))}
      </div>

      <Callout>
        Honest ceiling: open-loop LLM ≈ VWAP-match today. Closed-loop + RFT → beat that floor. PPO remains the bps
        ceiling (~31 bps).
      </Callout>
    </div>
  )
}

function Slide8() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>Sponsor Integration</Kicker>
      <Title>Built on the stack the hackathon sponsors provide</Title>
      <Table
        cols={['Sponsor', 'How we use it']}
        rows={[
          ['HUD', 'MCP environment, hud eval, agent traces, GRPO/RFT path (issue #15)'],
          ['Modal', 'PPO training on 51 tickers, 2.03M timesteps, data cache Volume'],
          ['Fireworks', 'gpt-oss open-weight agent + RFT reward function path'],
        ]}
      />
      <Callout>
        Same environment for eval and training — HUD's two-yield scenario pattern. PPO trained on Modal in hours; we
        trained and evaluated an RFT checkpoint end-to-end tonight on a forked Tinker-hosted Qwen3 8B model via HUD's
        TrainingClient — see the held-out result on the next slide.
      </Callout>
    </div>
  )
}

function Slide9() {
  const rows = [
    'Market data (Alpaca SIP)',
    'Simulator (impact model, volume curve)',
    'ExecutionEnv (gym)  ←  PPO policy (Modal-trained)',
    'HUD env.py (MCP)  ←  Claude / gpt-oss / RFT’d model',
    'Reward: slippage vs VWAP (verifiable, deterministic)',
    'Frontend: Proof · Showdown · Sandbox (live SSE)',
  ]
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>Architecture</Kicker>
      <Title>One environment, every layer reproducible</Title>
      <div className="mt-6 flex flex-col items-center gap-1.5">
        {rows.map((r, i) => (
          <div key={r} className="flex flex-col items-center gap-1.5">
            <div className="w-full max-w-xl rounded-[10px] border border-white/[0.07] bg-white/[0.02] px-5 py-2.5 text-center text-[13px] font-light text-white">
              {r}
            </div>
            {i < rows.length - 1 && <div className="h-3 w-px" style={{ background: FAINT }} />}
          </div>
        ))}
      </div>
      <Callout tone="dim">
        Participation cap prevents pathological impact blow-ups; slippage clipped defensively; held-out chronological
        split (last 20% of each ticker's history) — not a random subsample.
      </Callout>
    </div>
  )
}

// Real measurements from tonight's bounded training run (15 groups, forked Qwen3 8B on
// Tinker via HUD's TrainingClient) -- mean reward per group of 8 rollouts, in order. Group
// 13 is null: every rollout in that group failed, so it was skipped rather than trained on
// (see hud_rft_pipeline.py / cookbooks/rl-training -- a group needs at least one successful
// submit_schedule to give the trainer anything to learn from).
const RFT_CURVE = [
  { g: 1, reward: 0.172 }, { g: 2, reward: 0.098 }, { g: 3, reward: 0.130 }, { g: 4, reward: 0.165 },
  { g: 5, reward: 0.136 }, { g: 6, reward: 0.241 }, { g: 7, reward: 0.063 }, { g: 8, reward: 0.066 },
  { g: 9, reward: 0.172 }, { g: 10, reward: 0.068 }, { g: 11, reward: 0.189 }, { g: 12, reward: 0.163 },
  { g: 13, reward: null }, { g: 14, reward: 0.112 }, { g: 15, reward: 0.059 },
]

function Slide10() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>What's Next — RFT</Kicker>
      <Title>Phases 1-4 complete. The signal isn't significant yet (GitHub #15).</Title>

      <div className="mt-5 grid grid-cols-[1.05fr_1fr] gap-4">
        <div className="space-y-2">
          {[
            { done: true, t: 'Environment + verifiable reward, hardened for trainable signal' },
            { done: true, t: 'PPO proves learnable alpha exists (+30.9 bps, t=18.5)' },
            { done: true, t: 'Forked trainable Qwen3 8B (Tinker); TrainingClient.step() verified end-to-end, real checkpoints landing' },
            { done: true, t: '15-group training run complete — best checkpoint (step-000007) promoted to active head' },
            { done: true, t: 'Held-out comparison run: base (zero-shot) vs RFT, n=12, paired by seed' },
          ].map((s) => (
            <div key={s.t} className="flex items-start gap-3 rounded-[10px] border border-white/[0.06] bg-white/[0.02] px-3.5 py-2.5">
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: '#7ec98f' }} />
              <p className="flex-1 text-[12px] font-light leading-snug" style={{ color: '#d8d9de' }}>{s.t}</p>
              <span className="shrink-0 text-[10px] font-medium uppercase tracking-[0.06em]" style={{ color: '#7ec98f' }}>Done</span>
            </div>
          ))}

          <div className="mt-1 grid grid-cols-3 gap-2">
            <Stat label="Mean delta" value="+0.023" sub="rft − base, n=12" />
            <Stat label="Win/tie/loss" value="4/5/3" sub="held-out, paired" />
            <Stat label="t-statistic" value="0.28" sub="not significant" />
          </div>
        </div>

        <div className="rounded-[10px] border border-white/[0.06] bg-white/[0.02] p-3">
          <p className="text-[10px] font-medium uppercase tracking-[0.07em]" style={{ color: FAINT }}>
            Mean reward per training group (live run, n=15)
          </p>
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={RFT_CURVE} margin={{ top: 14, right: 8, bottom: 0, left: -16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis dataKey="g" stroke={FAINT} tick={{ fontSize: 10, fill: DIM }} label={{ value: 'group', position: 'insideBottom', offset: -2, fontSize: 10, fill: FAINT }} />
              <YAxis stroke={FAINT} tick={{ fontSize: 10, fill: DIM }} domain={[0, 0.3]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#0a0a0a', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, fontSize: 12 }}
                labelFormatter={(g) => `group ${g}`}
                formatter={(v) => [v == null ? 'skipped (0/8 success)' : v.toFixed(3), 'mean reward']}
              />
              <Line type="monotone" dataKey="reward" stroke={GOLD} strokeWidth={2} dot={{ r: 3, fill: GOLD }} connectNulls />
            </LineChart>
          </ResponsiveContainer>
          <p className="mt-1 text-[10px] font-light" style={{ color: FAINT }}>
            No clean upward trend over 14 gradient steps — consistent with the held-out result on the left.
          </p>
        </div>
      </div>

      <Callout>
        Honest framing for judges: RFT ran live end-to-end tonight — collect, train, promote, evaluate, all real. The
        held-out delta is small and positive (+0.023) but the t-stat (0.28) is nowhere near significant at n=12. We
        are not claiming a win over zero-shot. What's proven: the training pipeline itself works, including a real
        bug we found and fixed (a missing token-id flag silently blocked every training call). Reaching significance
        needs more training steps and a larger held-out n — exactly the next iteration, not a different approach.
      </Callout>
    </div>
  )
}

function Slide11() {
  return (
    <div className="flex h-full flex-col justify-center">
      <Kicker>Why Velora Matters</Kicker>
      <Title>Making AI agents trustworthy in high-stakes finance</Title>
      <div className="mt-6 grid grid-cols-2 gap-3">
        <Stat label="Verifiable rewards" value="Not vibes" sub="slippage in bps from real simulator state" />
        <Stat label="Honest benchmarks" value="VWAP-match" sub="the strong baseline, not a straw-man" />
        <Stat label="Reproducible eval" value="Paired seeds" sub="held-out days, public HUD job links" />
        <Stat label="Explainable schedules" value="PM trust" sub="natural-language constraints, compliance-ready" />
      </div>
      <Callout>
        Any quantitative skill you can simulate and grade, you can RL-teach into a frontier model. Execution is a
        clean, high-value first instance.
      </Callout>
    </div>
  )
}

function Slide12() {
  return (
    <div className="flex h-full flex-col items-center justify-center text-center">
      <Kicker>Close</Kicker>
      <Title>Velora: prove it, reproduce it, improve it.</Title>
      <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
        <Pill>Live demo</Pill>
        <Pill>GitHub repo</Pill>
        <Pill>HUD job links</Pill>
      </div>
      <Callout>One ask: try the Proof page with your own ticker and seed.</Callout>
      <div className="mt-6 flex gap-3">
        <a
          href="https://hud.ai"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-1.5 rounded-full border border-white/10 px-4 py-2 text-[12px] font-light text-white/80 hover:border-white/20"
        >
          hud.ai jobs <ExternalLink size={12} />
        </a>
      </div>
    </div>
  )
}

const SLIDES = [
  { Comp: Slide1, notes: 'Memorize the one-liner before this slide loads on screen.' },
  { Comp: Slide2, notes: 'Set up Originality: real finance problem, verifiable reward, agent interface.' },
  { Comp: Slide3, notes: 'This is the Completion slide — say "every number on Proof is reproducible one-click."' },
  { Comp: Slide4, notes: 'Say the honesty line about order-size dependence out loud. Never say "PPO always wins."' },
  { Comp: Slide5, notes: 'Pick ONE path live — Path A is safest. Have the backup screenshot ready.' },
  { Comp: Slide6, notes: 'Lead with the sanity-eval insight — it is your most original finding.' },
  { Comp: Slide7, notes: 'Put the PM prompt on screen verbatim. Latency is not the blocker — say so if asked.' },
  { Comp: Slide8, notes: 'Judges love seeing the sponsor stack used for real, not just name-dropped.' },
  { Comp: Slide9, notes: 'Walk top to bottom once, slowly. Mention the chronological holdout split.' },
  { Comp: Slide10, notes: 'RFT trained and was evaluated tonight, end to end. Present the null result plainly — small positive lean, not significant — and frame the win as the validated pipeline + the bug we found and fixed.' },
  { Comp: Slide11, notes: 'The Most Utopian pitch. Slow down here.' },
  { Comp: Slide12, notes: 'End on the ask, not a recap.' },
]

// ─── deck shell ───────────────────────────────────────────────────────────────

export default function Pitch() {
  const navigate = useNavigate()
  const [i, setI] = useState(0)
  const [showNotes, setShowNotes] = useState(false)
  const n = SLIDES.length

  const go = useCallback((delta) => setI((cur) => Math.min(n - 1, Math.max(0, cur + delta))), [n])

  useEffect(() => {
    function onKey(e) {
      if (e.key === 'ArrowRight' || e.key === ' ') go(1)
      else if (e.key === 'ArrowLeft') go(-1)
      else if (e.key.toLowerCase() === 'n') setShowNotes((s) => !s)
      else if (e.key === 'Escape') navigate('/')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [go, navigate])

  const { Comp, notes } = SLIDES[i]

  return (
    <div className="fixed inset-0 flex flex-col bg-black font-[var(--font-geist)]">
      {/* top bar */}
      <div className="flex items-center justify-between px-6 py-4">
        <button onClick={() => navigate('/')} className="text-[12px] font-light" style={{ color: FAINT }}>
          ← exit
        </button>
        <p className="text-[11px] font-light tabular-nums" style={{ color: FAINT }}>
          {String(i + 1).padStart(2, '0')} / {String(n).padStart(2, '0')}
        </p>
        <button onClick={() => setShowNotes((s) => !s)} className="flex items-center gap-1.5 text-[12px] font-light" style={{ color: showNotes ? GOLD : FAINT }}>
          <StickyNote size={13} /> notes
        </button>
      </div>

      {/* progress bar */}
      <div className="h-px w-full bg-white/[0.06]">
        <div className="h-px transition-all" style={{ width: `${((i + 1) / n) * 100}%`, background: GOLD }} />
      </div>

      {/* slide */}
      <div className="relative flex-1 overflow-y-auto px-16 py-10">
        <div className="mx-auto min-h-full max-w-4xl">
          <Comp />
        </div>

        <button
          onClick={() => go(-1)}
          disabled={i === 0}
          className="absolute left-4 top-1/2 -translate-y-1/2 rounded-full border border-white/10 p-2 text-white/60 hover:text-white disabled:opacity-20"
        >
          <ArrowLeft size={16} />
        </button>
        <button
          onClick={() => go(1)}
          disabled={i === n - 1}
          className="absolute right-4 top-1/2 -translate-y-1/2 rounded-full border border-white/10 p-2 text-white/60 hover:text-white disabled:opacity-20"
        >
          <ArrowRight size={16} />
        </button>
      </div>

      {/* dots */}
      <div className="flex items-center justify-center gap-1.5 pb-5">
        {SLIDES.map((_, idx) => (
          <button
            key={idx}
            onClick={() => setI(idx)}
            className="h-1.5 rounded-full transition-all"
            style={{ width: idx === i ? 18 : 6, background: idx === i ? GOLD : 'rgba(255,255,255,0.15)' }}
          />
        ))}
      </div>

      {/* speaker notes drawer */}
      {showNotes && (
        <div className="absolute bottom-16 left-1/2 w-full max-w-2xl -translate-x-1/2 rounded-[10px] border border-white/10 bg-[#0a0a0a] p-4 shadow-2xl">
          <div className="flex items-start justify-between gap-3">
            <p className="text-[12px] font-light leading-relaxed" style={{ color: '#d8d9de' }}>
              {notes}
            </p>
            <button onClick={() => setShowNotes(false)} className="text-white/40 hover:text-white">
              <X size={14} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
