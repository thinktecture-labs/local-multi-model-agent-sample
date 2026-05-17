import { useState, useCallback } from 'react'
import type { Exchange } from '../../types/state.ts'
import { formatMs, formatResponse } from '../../utils/format.ts'
import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'
import { escalateQueryStream } from '../../api/client.ts'
import { IntentBadge } from './IntentBadge.tsx'
import { CompareView } from './CompareView.tsx'
import { ThreePathView } from './ThreePathView.tsx'
import { EscalationBanner } from './EscalationBanner.tsx'
import { SourcePanel } from './SourcePanel.tsx'

export function ExchangeCard({
  exchange,
  idx,
  isActive,
  onClick,
}: {
  exchange: Exchange
  idx: number
  isActive: boolean
  onClick: () => void
}) {
  const { query, images, imageDataUrls, result, compareResult, threePathResult, threePathStreaming } = exchange
  const state = useAppState()
  const dispatch = useAppDispatch()
  const [isEscalating, setIsEscalating] = useState(false)

  // Check if confidence_assessment step recommends escalation
  const shouldEscalate = result?.steps.some(
    s => s.action === 'confidence_assessment' && s.details?.should_escalate === true,
  ) ?? false
  const showEscalationBanner = shouldEscalate && !result?.escalated && state.routingMode === 'hybrid' && state.networkMode === 'online'

  const handleEscalate = useCallback(async () => {
    setIsEscalating(true)
    // Clear any previous streaming text and mark as escalating
    dispatch({
      type: 'UPDATE_EXCHANGE',
      idx,
      updates: { streamingCloudText: '' },
    })
    try {
      await escalateQueryStream(query, {
        onToken: (text) => {
          dispatch({ type: 'APPEND_CLOUD_TOKEN', idx, text })
        },
        onDone: (esc) => {
          dispatch({
            type: 'UPDATE_EXCHANGE',
            idx,
            updates: {
              streamingCloudText: undefined,
              result: {
                ...result!,
                escalated: true,
                cloud_response: esc.cloud_response,
                cloud_model: esc.cloud_model,
                cloud_latency_ms: esc.cloud_latency_ms,
                cloud_cost: esc.cloud_cost,
                steps: [
                  ...result!.steps,
                  {
                    action: 'cloud_escalation' as const,
                    model: esc.cloud_model,
                    duration_ms: esc.cloud_latency_ms,
                    details: {
                      reason: 'Local confidence below threshold — escalated to cloud with local RAG context',
                      response: esc.cloud_response,
                      tokens: esc.cloud_tokens,
                      cost: esc.cloud_cost,
                      bytes_sent: esc.cloud_bytes_sent,
                    },
                    tokens_used: esc.cloud_tokens,
                  },
                ],
              },
            },
          })
          dispatch({ type: 'UPDATE_CLOUD_COST', cost: esc.cloud_cost, bytes: esc.cloud_bytes_sent })
        },
        onError: () => {
          dispatch({
            type: 'UPDATE_EXCHANGE',
            idx,
            updates: { streamingCloudText: undefined },
          })
        },
      })
    } catch {
      dispatch({
        type: 'UPDATE_EXCHANGE',
        idx,
        updates: { streamingCloudText: undefined },
      })
    } finally {
      setIsEscalating(false)
    }
  }, [query, idx, result, dispatch])

  return (
    <div className={isActive ? 'exchange active' : 'exchange'} onClick={onClick}>
      {/* Header row */}
      <div className="exchange-header">
        <span className="exchange-query">{query}</span>
        {imageDataUrls.length > 0 && (
          <span className="exchange-images">
            {imageDataUrls.map((url, i) => (
              <img
                key={i}
                className="exchange-thumb"
                src={url}
                alt={`Attached image ${i + 1}`}
              />
            ))}
          </span>
        )}
        {!imageDataUrls.length && images.length > 0 && (
          <span className="exchange-image-count">{images.length} image(s)</span>
        )}
        <span className="exchange-badges">
          {result && <IntentBadge intent={result.intent} />}
          {result?.escalated && result.cloud_model && (
            <span className="intent-badge cloud_escalation">CLOUD</span>
          )}
          {result && (
            <span className="time-badge">{formatMs(result.execution_time_ms)}</span>
          )}
        </span>
      </div>

      {/* Response area */}
      <div className="exchange-response">
        {threePathResult ? (
          <ThreePathView threePathResult={threePathResult} threePathStreaming={threePathStreaming} />
        ) : !result || (!result.response && exchange.streamingText == null) ? (
          <div className="typing-dots">
            <span />
            <span />
            <span />
          </div>
        ) : compareResult ? (
          <CompareView compareResult={compareResult} />
        ) : exchange.streamingText != null ? (
          <div className="exchange-text streaming">
            <span dangerouslySetInnerHTML={{ __html: formatResponse(exchange.streamingText) }} />
            <span className="cursor-blink" />
          </div>
        ) : (
          <div
            className="exchange-text"
            dangerouslySetInnerHTML={{ __html: formatResponse(result.response) }}
          />
        )}
      </div>

      {/* HITL escalation banner */}
      {showEscalationBanner && (
        <EscalationBanner exchangeIdx={idx} onEscalate={handleEscalate} isEscalating={isEscalating} />
      )}

      {/* Cloud escalation response — streaming or complete */}
      {exchange.streamingCloudText != null && (
        <div className="cloud-answer">
          <div className="cloud-answer-header">
            <span className="cloud-answer-label">Cloud answer</span>
            <span className="cloud-answer-meta">streaming&hellip;</span>
          </div>
          <div className="cloud-answer-body streaming">
            <span dangerouslySetInnerHTML={{ __html: formatResponse(exchange.streamingCloudText) }} />
            <span className="cursor-blink" />
          </div>
        </div>
      )}
      {result?.escalated && result.cloud_response && exchange.streamingCloudText == null && (
        <div className="cloud-answer">
          <div className="cloud-answer-header">
            <span className="cloud-answer-label">Cloud answer</span>
            <span className="cloud-answer-meta">
              {result.cloud_model} · {formatMs(result.cloud_latency_ms ?? 0)}
              {result.cloud_cost != null && ` · $${result.cloud_cost.toFixed(4)}`}
            </span>
          </div>
          <div
            className="cloud-answer-body"
            dangerouslySetInnerHTML={{ __html: formatResponse(result.cloud_response) }}
          />
        </div>
      )}

      {/* Source panel */}
      {result && !compareResult && !threePathResult && <SourcePanel result={result} />}
    </div>
  )
}
