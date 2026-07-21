import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Loader2, X, ExternalLink } from 'lucide-react'

// ── Reuse chart helpers already defined here ─────────────────

const LAYER_COLORS = [
  'rgb(129, 236, 255)',
  'rgb(170, 255, 220)',
  'rgb(255, 183, 77)',
  'rgb(255, 113, 108)',
  'rgb(167, 139, 250)',
  'rgb(255, 213, 79)',
]

// ── Simple SVG line chart ────────────────────────────────────

function TinyLine({ data, color }) {
  if (!data || data.length < 2) return <div className="h-12 opacity-20 text-[10px] text-center text-on-surface-variant pt-4">No data</div>
  const w = 260; const h = 56; const pad = 8
  const cw = w - pad * 2; const ch = h - pad * 2
  const mn = Math.min(...data); const mx = Math.max(...data); const range = mx - mn || 1
  const pts = data.map((v, i) => [
    pad + (data.length > 1 ? (i / (data.length - 1)) * cw : cw / 2),
    pad + ch - ((v - mn) / range) * ch,
  ])
  const d = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const area = `${d} L${pts[pts.length-1][0].toFixed(1)},${h-pad} L${pts[0][0].toFixed(1)},${h-pad} Z`
  const gid = `tg-${color.replace(/[^a-z0-9]/gi,'')}-${Math.random().toString(36).slice(2,6)}`
  return (
    <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#${gid})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={pts[pts.length-1][0]} cy={pts[pts.length-1][1]} r="3.5" fill={color} />
    </svg>
  )
}

function DualLineChart({ data1, data2, label1, label2, color1, color2, title, yLabel, animDelay = 0 }) {
  const allData = [...(data1 || []), ...(data2 || [])]
  if (!allData.length) return (
    <div className="glass-panel rounded-xl p-4 flex items-center justify-center h-48">
      <p className="text-on-surface-variant/40 text-xs font-mono">Awaiting data…</p>
    </div>
  )
  const w = 340; const h = 140; const pad = { t: 24, r: 12, b: 28, l: 38 }
  const cw = w - pad.l - pad.r; const ch = h - pad.t - pad.b
  const mn = Math.min(...allData) * 0.97; const mx = Math.max(...allData) * 1.03; const range = mx - mn || 1

  const makePath = (data, fillId) => {
    if (!data || !data.length) return { line: '', area: '' }
    const pts = data.map((v, i) => [
      pad.l + (data.length > 1 ? (i / (data.length - 1)) * cw : cw / 2),
      pad.t + ch - ((v - mn) / range) * ch,
    ])
    const line = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
    const area = `${line} L${pts[pts.length-1][0].toFixed(1)},${h-pad.b} L${pts[0][0].toFixed(1)},${h-pad.b} Z`
    return { line, area, last: pts[pts.length - 1], lastVal: data[data.length - 1] }
  }
  const p1 = makePath(data1, 'tg1'); const p2 = makePath(data2, 'tg2')
  const yTicks = [mn, (mn + mx) / 2, mx]

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: animDelay }}
      className="glass-panel rounded-xl p-4"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] uppercase tracking-widest font-bold text-outline">{title}</span>
        <div className="flex gap-3">
          <span className="flex items-center gap-1 text-[9px] font-mono" style={{ color: color1 }}>
            <span className="w-3 h-0.5 inline-block rounded" style={{ background: color1 }} /> {label1}
            {p1.lastVal !== undefined && <span className="ml-1 opacity-70">{p1.lastVal.toFixed ? p1.lastVal.toFixed(4) : p1.lastVal}</span>}
          </span>
          <span className="flex items-center gap-1 text-[9px] font-mono" style={{ color: color2 }}>
            <span className="w-3 h-0.5 inline-block rounded border-t-2 border-dashed" style={{ borderColor: color2 }} /> {label2}
            {p2.lastVal !== undefined && <span className="ml-1 opacity-70">{p2.lastVal.toFixed ? p2.lastVal.toFixed(4) : p2.lastVal}</span>}
          </span>
        </div>
      </div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible">
        <defs>
          <linearGradient id="tdl1" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color1} stopOpacity="0.2" />
            <stop offset="100%" stopColor={color1} stopOpacity="0.01" />
          </linearGradient>
          <linearGradient id="tdl2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color2} stopOpacity="0.1" />
            <stop offset="100%" stopColor={color2} stopOpacity="0.01" />
          </linearGradient>
        </defs>
        {yTicks.map((v, i) => {
          const y = pad.t + ch - ((v - mn) / range) * ch
          return (
            <g key={i}>
              <line x1={pad.l} x2={w - pad.r} y1={y} y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
              <text x={pad.l - 4} y={y + 3} textAnchor="end" className="font-mono" fontSize="7" fill="rgba(255,255,255,0.3)">{v.toFixed(2)}</text>
            </g>
          )
        })}
        {p1.area && <path d={p1.area} fill="url(#tdl1)" />}
        {p2.area && <path d={p2.area} fill="url(#tdl2)" />}
        {p1.line && <path d={p1.line} fill="none" stroke={color1} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />}
        {p2.line && <path d={p2.line} fill="none" stroke={color2} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="5,3" />}
        {p1.last && <circle cx={p1.last[0]} cy={p1.last[1]} r="3.5" fill={color1} />}
        {p2.last && <circle cx={p2.last[0]} cy={p2.last[1]} r="3" fill={color2} />}
        <text x={pad.l - 28} y={pad.t + ch / 2} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.3)" transform={`rotate(-90,${pad.l - 28},${pad.t + ch / 2})`}>{yLabel}</text>
      </svg>
    </motion.div>
  )
}

function SRPerLayerChart({ srPerLayer, srSteps, title, animDelay = 0 }) {
  if (!srPerLayer || !Object.keys(srPerLayer).length) return (
    <div className="glass-panel rounded-xl p-4 flex items-center justify-center h-48">
      <p className="text-on-surface-variant/40 text-xs font-mono">Awaiting SR data…</p>
    </div>
  )
  const layers = Object.keys(srPerLayer).sort()
  const allVals = layers.flatMap(l => srPerLayer[l])
  const mn = Math.min(...allVals) * 0.97; const mx = Math.max(...allVals) * 1.03; const range = mx - mn || 1
  const w = 340; const h = 140; const pad = { t: 24, r: 12, b: 28, l: 38 }
  const cw = w - pad.l - pad.r; const ch = h - pad.t - pad.b

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: animDelay }}
      className="glass-panel rounded-xl p-4"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] uppercase tracking-widest font-bold text-outline">{title}</span>
        <div className="flex gap-2 flex-wrap">
          {layers.map((l, i) => (
            <span key={l} className="flex items-center gap-1 text-[9px] font-mono" style={{ color: LAYER_COLORS[i % LAYER_COLORS.length] }}>
              <span className="w-2.5 h-0.5 inline-block rounded" style={{ background: LAYER_COLORS[i % LAYER_COLORS.length] }} />
              {l.replace('layer_', 'L')}
              <span className="opacity-60">{srPerLayer[l]?.slice(-1)[0]?.toFixed(3)}</span>
            </span>
          ))}
        </div>
      </div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible">
        {[mn, (mn+mx)/2, mx].map((v, i) => {
          const y = pad.t + ch - ((v - mn) / range) * ch
          return (
            <g key={i}>
              <line x1={pad.l} x2={w-pad.r} y1={y} y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
              <text x={pad.l-4} y={y+3} textAnchor="end" fontSize="7" fill="rgba(255,255,255,0.3)">{v.toFixed(2)}</text>
            </g>
          )
        })}
        {layers.map((l, li) => {
          const data = srPerLayer[l]
          if (!data || data.length < 1) return null
          const pts = data.map((v, i) => [
            pad.l + (data.length > 1 ? (i / (data.length - 1)) * cw : cw / 2),
            pad.t + ch - ((v - mn) / range) * ch,
          ])
          const d = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
          const color = LAYER_COLORS[li % LAYER_COLORS.length]
          return <path key={l} d={d} fill="none" stroke={color} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        })}
      </svg>
    </motion.div>
  )
}

function SROverallChart({ steps, values, title, animDelay = 0 }) {
  if (!values || !values.length) return (
    <div className="glass-panel rounded-xl p-4 flex items-center justify-center h-48">
      <p className="text-on-surface-variant/40 text-xs font-mono">Awaiting SR data…</p>
    </div>
  )
  const w = 340; const h = 140; const pad = { t: 24, r: 12, b: 28, l: 38 }
  const cw = w - pad.l - pad.r; const ch = h - pad.t - pad.b
  const mn = Math.min(...values) * 0.97; const mx = Math.max(...values) * 1.03; const range = mx - mn || 1
  const pts = values.map((v, i) => [
    pad.l + (values.length > 1 ? (i / (values.length - 1)) * cw : cw / 2),
    pad.t + ch - ((v - mn) / range) * ch,
  ])
  const d = pts.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`).join(' ')
  const area = `${d} L${pts[pts.length-1][0].toFixed(1)},${h-pad.b} L${pts[0][0].toFixed(1)},${h-pad.b} Z`
  const last = values[values.length - 1]

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: animDelay }}
      className="glass-panel rounded-xl p-4"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10px] uppercase tracking-widest font-bold text-outline">{title}</span>
        <span className="text-[10px] font-mono text-[#a78bfa]">avg {last?.toFixed(4)}</span>
      </div>
      <svg width="100%" height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible">
        <defs>
          <linearGradient id="tsr-overall" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.01" />
          </linearGradient>
        </defs>
        {[mn, (mn+mx)/2, mx].map((v, i) => {
          const y = pad.t + ch - ((v - mn) / range) * ch
          return (
            <g key={i}>
              <line x1={pad.l} x2={w-pad.r} y1={y} y2={y} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
              <text x={pad.l-4} y={y+3} textAnchor="end" fontSize="7" fill="rgba(255,255,255,0.3)">{v.toFixed(3)}</text>
            </g>
          )
        })}
        <path d={area} fill="url(#tsr-overall)" />
        <path d={d} fill="none" stroke="#a78bfa" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={pts[pts.length-1][0]} cy={pts[pts.length-1][1]} r="3.5" fill="#a78bfa" />
        <text x={pad.l-28} y={pad.t+ch/2} textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.3)" transform={`rotate(-90,${pad.l-28},${pad.t+ch/2})`}>Stable Rank</text>
        {steps && steps.length > 0 && (
          <text x={pad.l + cw / 2} y={h - 2} textAnchor="middle" fontSize="7" fill="rgba(255,255,255,0.3)">Training Steps</text>
        )}
      </svg>
    </motion.div>
  )
}

function PlaceholderCard({ label }) {
  return (
    <div className="glass-panel rounded-xl p-4 flex items-center justify-center h-48 border border-dashed border-outline/10">
      <p className="text-on-surface-variant/30 text-xs font-mono uppercase tracking-wider">{label}</p>
    </div>
  )
}

// ── Text Experiment 1 Block ──────────────────────────────────

function TextExp1Block({ metrics, loading, status, progress, onTrain, onCancel, hyperparams }) {
  const hasResults = status === 'completed' && metrics
  const isTraining = status === 'training'
  const [notebookOpen, setNotebookOpen] = useState(false)

  const plotData = isTraining && progress
    ? { trainLoss: progress.trainLoss, valLoss: progress.valLoss,
        trainAcc: progress.trainAcc, valAcc: progress.valAcc,
        srPerLayer: progress.srPerLayer, srOverall: progress.srOverall, srSteps: progress.srSteps }
    : hasResults ? metrics : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="col-span-12"
    >
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-baseline gap-4">
          <h2 className="text-xl font-headline font-bold text-primary uppercase tracking-wider">
            Text Experiment 1
          </h2>
          <p className="text-sm text-on-surface-variant font-body">
            {isTraining && progress
              ? `Training — Epoch ${progress.currentEpoch} / ${progress.totalEpochs}`
              : 'Vanilla Transformer — No residual, no LayerNorm.'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setNotebookOpen(true)}
            className="bg-[#20beff]/10 hover:bg-[#20beff]/20 text-[#20beff] border border-[#20beff]/20 px-4 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_20px_rgba(32,190,255,0.15)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap"
          >
            <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="#20beff">
              <path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.281.18.046.149.034.233-.07.352l-6.871 7.058 7.261 7.669c.093.117.117.2.07.282z"/>
            </svg>
            NOTEBOOK
          </button>
          {loading ? (
            <button onClick={onCancel} className="bg-error hover:bg-red-600 text-white px-6 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_25px_rgba(255,113,108,0.3)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap">
              <Loader2 size={14} className="animate-spin" /> STOP TRAINING
            </button>
          ) : (
            <button onClick={onTrain} className="bg-primary hover:bg-primary-dim text-on-primary px-6 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_25px_rgba(129,236,255,0.3)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap">
              TEXT EXP1 : TRAIN
            </button>
          )}
        </div>
      </div>

      {/* Notebook Modal */}
      {notebookOpen && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={() => setNotebookOpen(false)}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.25 }}
            className="glass-panel rounded-2xl border border-outline/10 shadow-2xl w-full max-w-lg mx-4 overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-outline/10">
              <h2 className="text-sm font-headline font-bold text-on-surface uppercase tracking-wider">Text Exp 1 — Kaggle Notebook</h2>
              <button onClick={() => setNotebookOpen(false)} className="p-2 text-on-surface-variant/60 hover:text-on-surface-variant hover:bg-surface-container rounded-lg transition-all">
                <X size={18} />
              </button>
            </div>
            <div className="p-6 flex flex-col items-center gap-5">
              <div className="w-16 h-16 rounded-2xl bg-[#20beff]/10 border border-[#20beff]/20 flex items-center justify-center">
                <svg viewBox="0 0 24 24" className="w-8 h-8" fill="#20beff"><path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.281.18.046.149.034.233-.07.352l-6.871 7.058 7.261 7.669c.093.117.117.2.07.282z"/></svg>
              </div>
              <div className="text-center">
                <p className="text-on-surface text-base font-headline font-bold mb-1">Kaggle Notebook</p>
                <p className="text-on-surface-variant/60 text-xs font-body max-w-xs">View training code for Text Experiment 1 on Kaggle.</p>
              </div>
              <div className="flex flex-wrap gap-2 justify-center">
                {['Python', 'PyTorch', 'AG News', 'GPU T4'].map(tag => (
                  <span key={tag} className="px-3 py-1 rounded-full bg-[#20beff]/10 border border-[#20beff]/20 text-[10px] font-mono text-[#20beff] uppercase tracking-wider">{tag}</span>
                ))}
              </div>
              <a href="https://www.kaggle.com/code/ssimranjit302/textexp1" target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-2 bg-[#20beff] hover:bg-[#18a8e6] text-white px-6 py-3 rounded-lg font-bold text-sm uppercase tracking-wide transition-all hover:shadow-[0_0_25px_rgba(32,190,255,0.3)] hover:scale-105 active:scale-95">
                <ExternalLink size={16} /> Open in Kaggle
              </a>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* ── Live Metric Cards ─────────────────────────────── */}
      {(isTraining || hasResults) && plotData && (() => {
        const tl = plotData.trainLoss?.slice(-1)[0]
        const vl = plotData.valLoss?.slice(-1)[0]
        const ta = plotData.trainAcc?.slice(-1)[0]
        const va = plotData.valAcc?.slice(-1)[0]
        const sr = plotData.srOverall?.slice(-1)[0]
        const cards = [
          { label: 'Train Loss',  value: tl != null ? tl.toFixed(4) : '—',       color: '#ff716c', sub: 'CE loss' },
          { label: 'Val Loss',    value: vl != null ? vl.toFixed(4) : '—',       color: '#ffb74d', sub: 'CE loss' },
          { label: 'Train Acc',   value: ta != null ? ta.toFixed(2) + '%' : '—', color: '#81ecff', sub: 'top-1'   },
          { label: 'Val Acc',     value: va != null ? va.toFixed(2) + '%' : '—', color: '#4caf50', sub: 'top-1'   },
          { label: 'Stable Rank', value: sr != null ? sr.toFixed(4) : '—',       color: '#a78bfa', sub: 'avg SR'  },
        ]
        return (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-5">
            {cards.map(({ label, value, color, sub }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: i * 0.06 }}
                className="glass-panel rounded-xl p-3.5 flex flex-col gap-1 border border-outline/5"
                style={{ boxShadow: `0 0 16px ${color}10` }}
              >
                <span className="text-[9px] uppercase tracking-widest text-on-surface-variant/50">{label}</span>
                <span className="text-lg font-mono font-bold leading-none" style={{ color }}>
                  {isTraining ? (
                    <motion.span
                      animate={{ opacity: [1, 0.5, 1] }}
                      transition={{ duration: 1.2, repeat: Infinity }}
                    >{value}</motion.span>
                  ) : value}
                </span>
                <span className="text-[9px] text-on-surface-variant/30 font-mono">{sub}</span>
              </motion.div>
            ))}
          </div>
        )
      })()}

      {/* 4-plot grid */}
      {plotData ? (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-5">
          <DualLineChart
            data1={plotData.trainLoss} data2={plotData.valLoss}
            label1="Train Loss" label2="Val Loss"
            color1="#ff716c" color2="#ffb74d"
            title="Loss" yLabel="Loss" animDelay={0.1}
          />
          <DualLineChart
            data1={plotData.trainAcc} data2={plotData.valAcc}
            label1="Train Acc" label2="Val Acc"
            color1="#81ecff" color2="#4caf50"
            title="Accuracy" yLabel="Accuracy (%)" animDelay={0.2}
          />
          <SRPerLayerChart
            srPerLayer={plotData.srPerLayer}
            srSteps={plotData.srSteps}
            title="Stable Rank / Layer"
            animDelay={0.3}
          />
          <SROverallChart
            steps={plotData.srSteps}
            values={plotData.srOverall}
            title="Stable Rank / Steps"
            animDelay={0.4}
          />
        </section>
      ) : (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-5">
          {['Loss', 'Accuracy', 'Stable Rank / Layer', 'Stable Rank / Steps'].map(label => (
            <PlaceholderCard key={label} label={label} />
          ))}
        </section>
      )}

      {/* Hyperparams panel */}
      {(hasResults || isTraining) && hyperparams && (
        <div className="glass-panel p-4 rounded-xl mb-2">
          <h3 className="text-[10px] uppercase font-bold text-outline tracking-widest mb-3">Hyperparameters</h3>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {[
              { label: 'Embeddings', value: hyperparams.n_embd, color: 'text-primary' },
              { label: 'Heads', value: hyperparams.n_head, color: 'text-tertiary' },
              { label: 'Layers', value: hyperparams.n_layer, color: 'text-secondary' },
              { label: 'Learn Rate', value: hyperparams.learning_rate, color: 'text-[#ffb74d]' },
              { label: 'Epochs', value: hyperparams.epochs, color: 'text-[#a78bfa]' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex flex-col">
                <span className="text-[9px] uppercase tracking-wider text-on-surface-variant/50 mb-0.5">{label}</span>
                <span className={`text-sm font-mono font-bold ${color}`}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dataset info */}
      <div className="glass-panel p-3 rounded-xl border border-outline/5">
        <div className="flex items-center gap-3">
          <span className="text-[9px] uppercase tracking-widest text-outline font-bold">Dataset</span>
          <span className="text-xs font-mono text-on-surface-variant">AG News · 4 classes (World / Sports / Business / Sci-Tech) · 120K train · 7.6K test</span>
        </div>
      </div>

      {status === 'failed' && (
        <div className="glass-panel p-4 rounded-xl mb-2 border border-error/30 mt-2">
          <p className="text-sm text-error font-body">Training failed. Check backend logs.</p>
        </div>
      )}
      {status === 'cancelled' && (
        <div className="glass-panel p-4 rounded-xl mb-2 border border-yellow-500/30 mt-2">
          <p className="text-sm text-yellow-400 font-body">Training stopped. Click "TEXT EXP1 : TRAIN" to restart.</p>
        </div>
      )}
    </motion.div>
  )
}

// ── Text Experiment 2 Block ──────────────────────────────────

function TextExp2Block({ metrics, loading, status, progress, onTrain, onCancel, hyperparams }) {
  const hasResults = status === 'completed' && metrics
  const isTraining = status === 'training'
  const [notebookOpen, setNotebookOpen] = useState(false)

  const plotData = isTraining && progress
    ? { trainLoss: progress.trainLoss, valLoss: progress.valLoss,
        trainAcc: progress.trainAcc, valAcc: progress.valAcc,
        srPerLayer: progress.srPerLayer, srOverall: progress.srOverall, srSteps: progress.srSteps }
    : hasResults ? metrics : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="col-span-12"
    >
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-baseline gap-4">
          <h2 className="text-xl font-headline font-bold text-primary uppercase tracking-wider">
            Text Experiment 2
          </h2>
          <p className="text-sm text-on-surface-variant font-body">
            {isTraining && progress
              ? `Training — Epoch ${progress.currentEpoch} / ${progress.totalEpochs}`
              : 'Residual Transformer — Pre-LN and residual connections.'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setNotebookOpen(true)}
            className="bg-[#20beff]/10 hover:bg-[#20beff]/20 text-[#20beff] border border-[#20beff]/20 px-4 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_20px_rgba(32,190,255,0.15)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap"
          >
            <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="#20beff">
              <path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.281.18.046.149.034.233-.07.352l-6.871 7.058 7.261 7.669c.093.117.117.2.07.282z"/>
            </svg>
            NOTEBOOK
          </button>
          {loading ? (
            <button onClick={onCancel} className="bg-error hover:bg-red-600 text-white px-6 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_25px_rgba(255,113,108,0.3)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap">
              <Loader2 size={14} className="animate-spin" /> STOP TRAINING
            </button>
          ) : (
            <button onClick={onTrain} className="bg-primary hover:bg-primary-dim text-on-primary px-6 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_25px_rgba(129,236,255,0.3)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap">
              TEXT EXP2 : TRAIN
            </button>
          )}
        </div>
      </div>

      {/* Notebook Modal */}
      {notebookOpen && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={() => setNotebookOpen(false)}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.25 }}
            className="glass-panel rounded-2xl border border-outline/10 shadow-2xl w-full max-w-lg mx-4 overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-outline/10">
              <h2 className="text-sm font-headline font-bold text-on-surface uppercase tracking-wider">Text Exp 2 — Kaggle Notebook</h2>
              <button onClick={() => setNotebookOpen(false)} className="p-2 text-on-surface-variant/60 hover:text-on-surface-variant hover:bg-surface-container rounded-lg transition-all">
                <X size={18} />
              </button>
            </div>
            <div className="p-6 flex flex-col items-center gap-5">
              <div className="w-16 h-16 rounded-2xl bg-[#20beff]/10 border border-[#20beff]/20 flex items-center justify-center">
                <svg viewBox="0 0 24 24" className="w-8 h-8" fill="#20beff"><path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.281.18.046.149.034.233-.07.352l-6.871 7.058 7.261 7.669c.093.117.117.2.07.282z"/></svg>
              </div>
              <div className="text-center">
                <p className="text-on-surface text-base font-headline font-bold mb-1">Kaggle Notebook</p>
                <p className="text-on-surface-variant/60 text-xs font-body max-w-xs">View training code for Text Experiment 2 on Kaggle.</p>
              </div>
              <div className="flex flex-wrap gap-2 justify-center">
                {['Python', 'PyTorch', 'AG News', 'GPU T4'].map(tag => (
                  <span key={tag} className="px-3 py-1 rounded-full bg-[#20beff]/10 border border-[#20beff]/20 text-[10px] font-mono text-[#20beff] uppercase tracking-wider">{tag}</span>
                ))}
              </div>
              <a href="https://www.kaggle.com/code/ssimranjit302/textexp2" target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-2 bg-[#20beff] hover:bg-[#18a8e6] text-white px-6 py-3 rounded-lg font-bold text-sm uppercase tracking-wide transition-all hover:shadow-[0_0_25px_rgba(32,190,255,0.3)] hover:scale-105 active:scale-95">
                <ExternalLink size={16} /> Open in Kaggle
              </a>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* ── Live Metric Cards ─────────────────────────────── */}
      {(isTraining || hasResults) && plotData && (() => {
        const tl = plotData.trainLoss?.slice(-1)[0]
        const vl = plotData.valLoss?.slice(-1)[0]
        const ta = plotData.trainAcc?.slice(-1)[0]
        const va = plotData.valAcc?.slice(-1)[0]
        const sr = plotData.srOverall?.slice(-1)[0]
        const cards = [
          { label: 'Train Loss',  value: tl != null ? tl.toFixed(4) : '—',       color: '#ff716c', sub: 'CE loss' },
          { label: 'Val Loss',    value: vl != null ? vl.toFixed(4) : '—',       color: '#ffb74d', sub: 'CE loss' },
          { label: 'Train Acc',   value: ta != null ? ta.toFixed(2) + '%' : '—', color: '#81ecff', sub: 'top-1'   },
          { label: 'Val Acc',     value: va != null ? va.toFixed(2) + '%' : '—', color: '#4caf50', sub: 'top-1'   },
          { label: 'Stable Rank', value: sr != null ? sr.toFixed(4) : '—',       color: '#a78bfa', sub: 'avg SR'  },
        ]
        return (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-5">
            {cards.map(({ label, value, color, sub }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: i * 0.06 }}
                className="glass-panel rounded-xl p-3.5 flex flex-col gap-1 border border-outline/5"
                style={{ boxShadow: `0 0 16px ${color}10` }}
              >
                <span className="text-[9px] uppercase tracking-widest text-on-surface-variant/50">{label}</span>
                <span className="text-lg font-mono font-bold leading-none" style={{ color }}>
                  {isTraining ? (
                    <motion.span
                      animate={{ opacity: [1, 0.5, 1] }}
                      transition={{ duration: 1.2, repeat: Infinity }}
                    >{value}</motion.span>
                  ) : value}
                </span>
                <span className="text-[9px] text-on-surface-variant/30 font-mono">{sub}</span>
              </motion.div>
            ))}
          </div>
        )
      })()}

      {/* 4-plot grid */}
      {plotData ? (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-5">
          <DualLineChart
            data1={plotData.trainLoss} data2={plotData.valLoss}
            label1="Train Loss" label2="Val Loss"
            color1="#ff716c" color2="#ffb74d"
            title="Loss" yLabel="Loss" animDelay={0.1}
          />
          <DualLineChart
            data1={plotData.trainAcc} data2={plotData.valAcc}
            label1="Train Acc" label2="Val Acc"
            color1="#81ecff" color2="#4caf50"
            title="Accuracy" yLabel="Accuracy (%)" animDelay={0.2}
          />
          <SRPerLayerChart
            srPerLayer={plotData.srPerLayer}
            srSteps={plotData.srSteps}
            title="Stable Rank / Layer"
            animDelay={0.3}
          />
          <SROverallChart
            steps={plotData.srSteps}
            values={plotData.srOverall}
            title="Stable Rank / Steps"
            animDelay={0.4}
          />
        </section>
      ) : (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-5">
          {['Loss', 'Accuracy', 'Stable Rank / Layer', 'Stable Rank / Steps'].map(label => (
            <PlaceholderCard key={label} label={label} />
          ))}
        </section>
      )}

      {/* Hyperparams panel */}
      {(hasResults || isTraining) && hyperparams && (
        <div className="glass-panel p-4 rounded-xl mb-2">
          <h3 className="text-[10px] uppercase font-bold text-outline tracking-widest mb-3">Hyperparameters</h3>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {[
              { label: 'Embeddings', value: hyperparams.n_embd, color: 'text-primary' },
              { label: 'Heads', value: hyperparams.n_head, color: 'text-tertiary' },
              { label: 'Layers', value: hyperparams.n_layer, color: 'text-secondary' },
              { label: 'Learn Rate', value: hyperparams.learning_rate, color: 'text-[#ffb74d]' },
              { label: 'Epochs', value: hyperparams.epochs, color: 'text-[#a78bfa]' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex flex-col">
                <span className="text-[9px] uppercase tracking-wider text-on-surface-variant/50 mb-0.5">{label}</span>
                <span className={`text-sm font-mono font-bold ${color}`}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dataset info */}
      <div className="glass-panel p-3 rounded-xl border border-outline/5">
        <div className="flex items-center gap-3">
          <span className="text-[9px] uppercase tracking-widest text-outline font-bold">Dataset</span>
          <span className="text-xs font-mono text-on-surface-variant">AG News · 4 classes (World / Sports / Business / Sci-Tech) · 120K train · 7.6K test</span>
        </div>
      </div>

      {status === 'failed' && (
        <div className="glass-panel p-4 rounded-xl mb-2 border border-error/30 mt-2">
          <p className="text-sm text-error font-body">Training failed. Check backend logs.</p>
        </div>
      )}
      {status === 'cancelled' && (
        <div className="glass-panel p-4 rounded-xl mb-2 border border-yellow-500/30 mt-2">
          <p className="text-sm text-yellow-400 font-body">Training stopped. Click "TEXT EXP2 : TRAIN" to restart.</p>
        </div>
      )}
    </motion.div>
  )
}

// ── Text Experiment 3 Block ──────────────────────────────────

function TextExp3Block({ metrics, loading, status, progress, onTrain, onCancel, hyperparams }) {
  const hasResults = status === 'completed' && metrics
  const isTraining = status === 'training'
  const [notebookOpen, setNotebookOpen] = useState(false)

  const plotData = isTraining && progress
    ? { trainLoss: progress.trainLoss, valLoss: progress.valLoss,
        trainAcc: progress.trainAcc, valAcc: progress.valAcc,
        srPerLayer: progress.srPerLayer, srOverall: progress.srOverall, srSteps: progress.srSteps }
    : hasResults ? metrics : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="col-span-12"
    >
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-baseline gap-4">
          <h2 className="text-xl font-headline font-bold text-primary uppercase tracking-wider">
            Text Experiment 3
          </h2>
          <p className="text-sm text-on-surface-variant font-body">
            {isTraining && progress
              ? `Training — Epoch ${progress.currentEpoch} / ${progress.totalEpochs}`
              : 'Differential Attention with RMSNorm, SwiGLU FFN, lambda-based attention and dropout(0.1)'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setNotebookOpen(true)}
            className="bg-[#20beff]/10 hover:bg-[#20beff]/20 text-[#20beff] border border-[#20beff]/20 px-4 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_20px_rgba(32,190,255,0.15)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap"
          >
            <svg viewBox="0 0 24 24" className="w-3.5 h-3.5" fill="#20beff">
              <path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.281.18.046.149.034.233-.07.352l-6.871 7.058 7.261 7.669c.093.117.117.2.07.282z"/>
            </svg>
            NOTEBOOK
          </button>
          {loading ? (
            <button onClick={onCancel} className="bg-error hover:bg-red-600 text-white px-6 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_25px_rgba(255,113,108,0.3)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap">
              <Loader2 size={14} className="animate-spin" /> STOP TRAINING
            </button>
          ) : (
            <button onClick={onTrain} className="bg-primary hover:bg-primary-dim text-on-primary px-6 py-2.5 font-black rounded-lg transition-all hover:shadow-[0_0_25px_rgba(129,236,255,0.3)] active:scale-95 uppercase tracking-widest text-xs flex items-center gap-2 whitespace-nowrap">
              TEXT EXP3 : TRAIN
            </button>
          )}
        </div>
      </div>

      {/* Notebook Modal */}
      {notebookOpen && (
        <motion.div
          initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={() => setNotebookOpen(false)}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.25 }}
            className="glass-panel rounded-2xl border border-outline/10 shadow-2xl w-full max-w-lg mx-4 overflow-hidden"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between p-4 border-b border-outline/10">
              <h2 className="text-sm font-headline font-bold text-on-surface uppercase tracking-wider">Text Exp 2 — Kaggle Notebook</h2>
              <button onClick={() => setNotebookOpen(false)} className="p-2 text-on-surface-variant/60 hover:text-on-surface-variant hover:bg-surface-container rounded-lg transition-all">
                <X size={18} />
              </button>
            </div>
            <div className="p-6 flex flex-col items-center gap-5">
              <div className="w-16 h-16 rounded-2xl bg-[#20beff]/10 border border-[#20beff]/20 flex items-center justify-center">
                <svg viewBox="0 0 24 24" className="w-8 h-8" fill="#20beff"><path d="M18.825 23.859c-.022.092-.117.141-.281.141h-3.139c-.187 0-.351-.082-.492-.248l-5.178-6.589-1.448 1.374v5.111c0 .235-.117.352-.351.352H5.505c-.236 0-.354-.117-.354-.352V.353c0-.233.118-.353.354-.353h2.431c.234 0 .351.12.351.353v14.343l6.203-6.272c.165-.165.33-.246.495-.246h3.239c.144 0 .236.06.281.18.046.149.034.233-.07.352l-6.871 7.058 7.261 7.669c.093.117.117.2.07.282z"/></svg>
              </div>
              <div className="text-center">
                <p className="text-on-surface text-base font-headline font-bold mb-1">Kaggle Notebook</p>
                <p className="text-on-surface-variant/60 text-xs font-body max-w-xs">View training code for Text Experiment 3 on Kaggle.</p>
              </div>
              <div className="flex flex-wrap gap-2 justify-center">
                {['Python', 'PyTorch', 'AG News', 'GPU T4'].map(tag => (
                  <span key={tag} className="px-3 py-1 rounded-full bg-[#20beff]/10 border border-[#20beff]/20 text-[10px] font-mono text-[#20beff] uppercase tracking-wider">{tag}</span>
                ))}
              </div>
              <a href="https://www.kaggle.com/code/ssimranjit302/textexp3" target="_blank" rel="noopener noreferrer"
                className="flex items-center gap-2 bg-[#20beff] hover:bg-[#18a8e6] text-white px-6 py-3 rounded-lg font-bold text-sm uppercase tracking-wide transition-all hover:shadow-[0_0_25px_rgba(32,190,255,0.3)] hover:scale-105 active:scale-95">
                <ExternalLink size={16} /> Open in Kaggle
              </a>
            </div>
          </motion.div>
        </motion.div>
      )}

      {/* ── Live Metric Cards ─────────────────────────────── */}
      {(isTraining || hasResults) && plotData && (() => {
        const tl = plotData.trainLoss?.slice(-1)[0]
        const vl = plotData.valLoss?.slice(-1)[0]
        const ta = plotData.trainAcc?.slice(-1)[0]
        const va = plotData.valAcc?.slice(-1)[0]
        const sr = plotData.srOverall?.slice(-1)[0]
        const cards = [
          { label: 'Train Loss',  value: tl != null ? tl.toFixed(4) : '—',       color: '#ff716c', sub: 'CE loss' },
          { label: 'Val Loss',    value: vl != null ? vl.toFixed(4) : '—',       color: '#ffb74d', sub: 'CE loss' },
          { label: 'Train Acc',   value: ta != null ? ta.toFixed(2) + '%' : '—', color: '#81ecff', sub: 'top-1'   },
          { label: 'Val Acc',     value: va != null ? va.toFixed(2) + '%' : '—', color: '#4caf50', sub: 'top-1'   },
          { label: 'Stable Rank', value: sr != null ? sr.toFixed(4) : '—',       color: '#a78bfa', sub: 'avg SR'  },
        ]
        return (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-5">
            {cards.map(({ label, value, color, sub }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, delay: i * 0.06 }}
                className="glass-panel rounded-xl p-3.5 flex flex-col gap-1 border border-outline/5"
                style={{ boxShadow: `0 0 16px ${color}10` }}
              >
                <span className="text-[9px] uppercase tracking-widest text-on-surface-variant/50">{label}</span>
                <span className="text-lg font-mono font-bold leading-none" style={{ color }}>
                  {isTraining ? (
                    <motion.span
                      animate={{ opacity: [1, 0.5, 1] }}
                      transition={{ duration: 1.2, repeat: Infinity }}
                    >{value}</motion.span>
                  ) : value}
                </span>
                <span className="text-[9px] text-on-surface-variant/30 font-mono">{sub}</span>
              </motion.div>
            ))}
          </div>
        )
      })()}

      {/* 4-plot grid */}
      {plotData ? (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-5">
          <DualLineChart
            data1={plotData.trainLoss} data2={plotData.valLoss}
            label1="Train Loss" label2="Val Loss"
            color1="#ff716c" color2="#ffb74d"
            title="Loss" yLabel="Loss" animDelay={0.1}
          />
          <DualLineChart
            data1={plotData.trainAcc} data2={plotData.valAcc}
            label1="Train Acc" label2="Val Acc"
            color1="#81ecff" color2="#4caf50"
            title="Accuracy" yLabel="Accuracy (%)" animDelay={0.2}
          />
          <SRPerLayerChart
            srPerLayer={plotData.srPerLayer}
            srSteps={plotData.srSteps}
            title="Stable Rank / Layer"
            animDelay={0.3}
          />
          <SROverallChart
            steps={plotData.srSteps}
            values={plotData.srOverall}
            title="Stable Rank / Steps"
            animDelay={0.4}
          />
        </section>
      ) : (
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-5">
          {['Loss', 'Accuracy', 'Stable Rank / Layer', 'Stable Rank / Steps'].map(label => (
            <PlaceholderCard key={label} label={label} />
          ))}
        </section>
      )}

      {/* Hyperparams panel */}
      {(hasResults || isTraining) && hyperparams && (
        <div className="glass-panel p-4 rounded-xl mb-2">
          <h3 className="text-[10px] uppercase font-bold text-outline tracking-widest mb-3">Hyperparameters</h3>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {[
              { label: 'Embeddings', value: hyperparams.n_embd, color: 'text-primary' },
              { label: 'Heads', value: hyperparams.n_head, color: 'text-tertiary' },
              { label: 'Layers', value: hyperparams.n_layer, color: 'text-secondary' },
              { label: 'Learn Rate', value: hyperparams.learning_rate, color: 'text-[#ffb74d]' },
              { label: 'Epochs', value: hyperparams.epochs, color: 'text-[#a78bfa]' },
            ].map(({ label, value, color }) => (
              <div key={label} className="flex flex-col">
                <span className="text-[9px] uppercase tracking-wider text-on-surface-variant/50 mb-0.5">{label}</span>
                <span className={`text-sm font-mono font-bold ${color}`}>{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dataset info */}
      <div className="glass-panel p-3 rounded-xl border border-outline/5">
        <div className="flex items-center gap-3">
          <span className="text-[9px] uppercase tracking-widest text-outline font-bold">Dataset</span>
          <span className="text-xs font-mono text-on-surface-variant">AG News · 4 classes (World / Sports / Business / Sci-Tech) · 120K train · 7.6K test</span>
        </div>
      </div>

      {status === 'failed' && (
        <div className="glass-panel p-4 rounded-xl mb-2 border border-error/30 mt-2">
          <p className="text-sm text-error font-body">Training failed. Check backend logs.</p>
        </div>
      )}
      {status === 'cancelled' && (
        <div className="glass-panel p-4 rounded-xl mb-2 border border-yellow-500/30 mt-2">
          <p className="text-sm text-yellow-400 font-body">Training stopped. Click "TEXT EXP3 : TRAIN" to restart.</p>
        </div>
      )}
    </motion.div>
  )
}

// ── Main export ──────────────────────────────────────────────

export default function TextMetricsSection({
  textExp1Metrics, textExp1Loading, textExp1Status, textExp1Progress,
  onTrainTextExp1, onCancelTextExp1, textExp1Hyperparams,
  textExp2Metrics, textExp2Loading, textExp2Status, textExp2Progress,
  onTrainTextExp2, onCancelTextExp2, textExp2Hyperparams,
  textExp3Metrics, textExp3Loading, textExp3Status, textExp3Progress,
  onTrainTextExp3, onCancelTextExp3, textExp3Hyperparams}) {
  return (
    <div className="col-span-12 space-y-8">
      <TextExp1Block
        metrics={textExp1Metrics}
        loading={textExp1Loading}
        status={textExp1Status}
        progress={textExp1Progress}
        onTrain={onTrainTextExp1}
        onCancel={onCancelTextExp1}
        hyperparams={textExp1Hyperparams}
      />
      <TextExp2Block
        metrics={textExp2Metrics}
        loading={textExp2Loading}
        status={textExp2Status}
        progress={textExp2Progress}
        onTrain={onTrainTextExp2}
        onCancel={onCancelTextExp2}
        hyperparams={textExp2Hyperparams}
      />
      <TextExp3Block
        metrics={textExp3Metrics}
        loading={textExp3Loading}
        status={textExp3Status}
        progress={textExp3Progress}
        onTrain={onTrainTextExp3}
        onCancel={onCancelTextExp3}
        hyperparams={textExp3Hyperparams}
      />
    </div>
  )
}
