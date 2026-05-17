import { useState } from 'react'
import { useAppState } from '../../state/AppContext.tsx'
import { formatMs, intentLabel } from '../../utils/format.ts'
import { StepCard } from './StepCard.tsx'
import { LatencyWaterfall } from './LatencyWaterfall.tsx'
import type { QueryResult } from '../../types/api.ts'

type TracePath = 'multi-models' | 'qwen' | 'cloud'
const PATH_LABELS: Record<TracePath, string> = {
  'multi-models': 'Multi-Models',
  'qwen': 'MoE',
  'cloud': 'Cloud',
}

function TraceContent({ result }: { result: QueryResult }) {
  const totalMs = result.execution_time_ms
  const steps = result.steps ?? []

  return (
    <div className="trace-body">
      <div className="intent-row">
        <span className="intent-lbl">Intent</span>
        <span className={`intent-val ${result.intent}`}>{intentLabel(result.intent)}</span>
      </div>
      <div className="step-list">
        {steps.map((step, i) => (
          <StepCard key={i} step={step} index={i} totalMs={totalMs} />
        ))}
      </div>
      {steps.length > 1 && (
        <LatencyWaterfall steps={steps} totalMs={totalMs} />
      )}
    </div>
  )
}

export function TracePanel() {
  const { exchanges, activeIdx, loading } = useAppState()
  const [activePath, setActivePath] = useState<TracePath>('multi-models')

  const exchange = activeIdx >= 0 ? exchanges[activeIdx] : null
  const threePathResult = exchange?.threePathResult
  const result = exchange?.result ?? null

  if (loading) {
    return (
      <>
        <div className="trace-header">
          <span className="trace-title">Pipeline Trace</span>
          <span className="trace-total">{'\u2014'}</span>
        </div>
        <div className="trace-body">
          <div className="trace-loading">
            <div className="pipeline-viz">
              <div className="pl-node g3 active">classify</div>
              <div className="pl-arr">{'\u2192'}</div>
              <div className="pl-node fn">resolve</div>
              <div className="pl-arr">{'\u2192'}</div>
              <div className="pl-node emb">search</div>
              <div className="pl-arr">{'\u2192'}</div>
              <div className="pl-node g3">synthesize</div>
            </div>
            <div>Processing&hellip;</div>
          </div>
        </div>
      </>
    )
  }

  if (!exchange || !result) {
    return (
      <>
        <div className="trace-header">
          <span className="trace-title">Pipeline Trace</span>
          <span className="trace-total">{'\u2014'}</span>
        </div>
        <div className="trace-body">
          <div className="trace-empty">
            <div className="icon">{'\u2B21'}</div>
            <div>Select a query to see<br/>the agent's decision trace</div>
          </div>
        </div>
      </>
    )
  }

  // Three-path mode: show tab bar
  if (threePathResult) {
    const pathResult =
      activePath === 'qwen' ? threePathResult.qwen :
      activePath === 'cloud' ? threePathResult.cloud :
      threePathResult.multi_models

    return (
      <>
        <div className="trace-header">
          <span className="trace-title">Pipeline Trace</span>
          <span className="trace-total">
            {pathResult ? formatMs(pathResult.execution_time_ms) : '\u2014'}
          </span>
        </div>
        <div className="trace-path-tabs">
          {(['multi-models', 'qwen', 'cloud'] as TracePath[]).map((path) => {
            const pathData =
              path === 'qwen' ? threePathResult.qwen :
              path === 'cloud' ? threePathResult.cloud :
              threePathResult.multi_models
            return (
              <button
                key={path}
                className={`trace-path-tab${activePath === path ? ' active' : ''}${!pathData ? ' disabled' : ''}`}
                onClick={() => pathData && setActivePath(path)}
                disabled={!pathData}
              >
                {PATH_LABELS[path]}
              </button>
            )
          })}
        </div>
        {pathResult ? (
          <TraceContent result={pathResult} />
        ) : (
          <div className="trace-body">
            <div className="trace-empty">
              <div>Path not available</div>
            </div>
          </div>
        )}
      </>
    )
  }

  // Standard single-path trace
  return (
    <>
      <div className="trace-header">
        <span className="trace-title">Pipeline Trace</span>
        <span className="trace-total">{formatMs(result.execution_time_ms)}</span>
      </div>
      <TraceContent result={result} />
    </>
  )
}
