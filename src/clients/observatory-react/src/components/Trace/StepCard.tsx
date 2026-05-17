import { useEffect, useRef } from 'react'
import type { ExecutionStep } from '../../types/api.ts'
import { formatMs, modelKey, modelColor, shortModel, ACTION_LABEL } from '../../utils/format.ts'
import { StepDetail } from './StepDetail.tsx'

interface StepCardProps {
  step: ExecutionStep
  index: number
  totalMs: number
}

export function StepCard({ step, index, totalMs }: StepCardProps) {
  const barRef = useRef<HTMLDivElement>(null)
  const widthPct = totalMs > 0 ? (step.duration_ms / totalMs) * 100 : 0

  useEffect(() => {
    const el = barRef.current
    if (!el) return
    requestAnimationFrame(() => {
      el.style.width = `${widthPct}%`
    })
  }, [widthPct])

  return (
    <div className="step-card">
      <div className="step-head">
        <span className="step-num">{index + 1}</span>
        <span className="step-action">{ACTION_LABEL[step.action] ?? step.action}</span>
        <span className="step-model-tag" data-m={modelKey(step.model)}>
          {shortModel(step.model)}
        </span>
        <span className="step-dur">{formatMs(step.duration_ms)}</span>
      </div>
      <div className="bar-track">
        <div
          ref={barRef}
          className="bar-fill"
          style={{ width: 0, background: modelColor(step.model) }}
          data-w={widthPct}
        />
      </div>
      <StepDetail step={step} />
    </div>
  )
}
