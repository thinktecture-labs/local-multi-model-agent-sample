import { useRef, useEffect } from 'react'
import type { ThreePathResult, QueryResult } from '../../types/api.ts'
import { formatMs, formatResponse } from '../../utils/format.ts'

function PathColumn({
  label,
  result,
  streamingText,
  colorVar,
  onSelect,
}: {
  label: string
  result: QueryResult | null
  streamingText?: string
  colorVar: string
  onSelect?: () => void
}) {
  const bodyRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom as content changes (streaming tokens or final result)
  useEffect(() => {
    if (bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight
    }
  }, [streamingText, result])

  // Waiting for first token — show typing dots
  if (!result && streamingText != null && streamingText.length === 0) {
    return (
      <div className="three-path-col" onClick={onSelect}>
        <div className="three-path-header">
          <span className="three-path-label" style={{ color: colorVar }}>{label}</span>
        </div>
        <div className="three-path-body">
          <div className="typing-dots"><span /><span /><span /></div>
        </div>
      </div>
    )
  }

  // Stream never started or backend errored — genuinely not available
  if (!result && !streamingText) {
    return (
      <div className="three-path-col unavailable" onClick={onSelect}>
        <div className="three-path-header">
          <span className="three-path-label" style={{ color: colorVar }}>{label}</span>
        </div>
        <div className="three-path-body">Not available</div>
      </div>
    )
  }

  // Streaming in progress — show partial text, auto-scrolling
  if (!result && streamingText) {
    return (
      <div className="three-path-col" onClick={onSelect}>
        <div className="three-path-header">
          <span className="three-path-label" style={{ color: colorVar }}>{label}</span>
        </div>
        <div className="three-path-body" ref={bodyRef}>
          <span dangerouslySetInnerHTML={{ __html: formatResponse(streamingText) }} />
        </div>
      </div>
    )
  }

  if (!result) return null

  const cost = result.cloud_cost
  const models = result.models_used?.length
    ? result.models_used.join(', ')
    : result.steps?.[0]?.model ?? '--'

  return (
    <div className="three-path-col" onClick={onSelect}>
      <div className="three-path-header">
        <span className="three-path-label" style={{ color: colorVar }}>{label}</span>
        <span className="three-path-meta">
          {formatMs(result.execution_time_ms)}
          {cost != null ? ` · $${cost.toFixed(4)}` : ' · $0.00'}
        </span>
      </div>
      <div className="three-path-models">{models}</div>
      <div
        className="three-path-body"
        ref={bodyRef}
        dangerouslySetInnerHTML={{ __html: formatResponse(result.response) }}
      />
    </div>
  )
}

export function ThreePathView({
  threePathResult,
  threePathStreaming,
  onSelectPath,
}: {
  threePathResult: ThreePathResult
  threePathStreaming?: Partial<Record<'multi_models' | 'qwen' | 'cloud', string>>
  onSelectPath?: (path: 'multi-models' | 'qwen' | 'cloud') => void
}) {
  return (
    <div className="three-path-split">
      <PathColumn
        label="MULTI-MODELS"
        result={threePathResult.multi_models}
        streamingText={threePathStreaming?.multi_models}
        colorVar="var(--c-gemma)"
        onSelect={() => onSelectPath?.('multi-models')}
      />
      <PathColumn
        label="MoE"
        result={threePathResult.qwen}
        streamingText={threePathStreaming?.qwen}
        colorVar="var(--c-qwen)"
        onSelect={() => onSelectPath?.('qwen')}
      />
      <PathColumn
        label="CLOUD"
        result={threePathResult.cloud}
        streamingText={threePathStreaming?.cloud}
        colorVar="var(--c-cloud)"
        onSelect={() => onSelectPath?.('cloud')}
      />
    </div>
  )
}
