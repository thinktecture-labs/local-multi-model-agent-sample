import { useEffect, useRef } from 'react'
import type { ExecutionStep } from '../../types/api.ts'
import { formatMs, modelColor, shortModel, WF_ACTION_LABEL } from '../../utils/format.ts'

interface LatencyWaterfallProps {
  steps: ExecutionStep[]
  totalMs: number
}

export function LatencyWaterfall({ steps, totalMs }: LatencyWaterfallProps) {
  const barsRef = useRef<(HTMLDivElement | null)[]>([])

  const markers = [0, 25, 50, 75, 100]

  const uniqueModels = Array.from(new Set(steps.map(s => shortModel(s.model)).filter(Boolean)))

  let cumulative = 0
  const offsets = steps.map(s => {
    const offset = cumulative
    cumulative += s.duration_ms
    return offset
  })

  // Use the larger of totalMs or cumulative step duration as the scale
  // (voice transcription can exceed the reported agent execution time)
  const scaleMs = Math.max(totalMs, cumulative)

  useEffect(() => {
    barsRef.current.forEach((el, i) => {
      if (!el) return
      const widthPct = scaleMs > 0 ? (steps[i].duration_ms / scaleMs) * 100 : 0
      const leftPct = scaleMs > 0 ? (offsets[i] / scaleMs) * 100 : 0
      requestAnimationFrame(() => {
        el.style.left = `${leftPct}%`
        el.style.width = `${widthPct}%`
      })
    })
  }, [steps, scaleMs, offsets])

  return (
    <div className="waterfall">
      <div className="timeline-lbl">Latency Waterfall</div>
      <div className="wf-axis">
        {markers.map(pct => (
          <span className="wf-marker" key={pct} style={{ left: `${pct}%` }}>
            {formatMs((pct / 100) * scaleMs)}
          </span>
        ))}
      </div>

      {steps.map((step, i) => (
        <div className="wf-row" key={i}>
          <span className="wf-label">{WF_ACTION_LABEL[step.action] ?? step.action}</span>
          <div className="wf-track">
            <div
              ref={el => { barsRef.current[i] = el }}
              className="wf-bar"
              style={{
                left: 0,
                width: 0,
                background: modelColor(step.model),
              }}
            />
          </div>
          <span className="wf-dur">{formatMs(step.duration_ms)}</span>
        </div>
      ))}

      <div className="timeline-legend">
        {uniqueModels.map(m => (
          <span className="legend-item" key={m}>
            <span
              className="legend-dot"
              style={{ background: modelColor(m) }}
            />
            {m}
          </span>
        ))}
      </div>

      <div className="wf-total">
        {formatMs(scaleMs)} total
      </div>
    </div>
  )
}
