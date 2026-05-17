import { useState, useEffect, useRef } from 'react'
import type { ExecutionStep } from '../../types/api.ts'

interface StepDetailProps {
  step: ExecutionStep
}

export function StepDetail({ step }: StepDetailProps) {
  const d = step.details

  switch (step.action) {
    case 'classify_intent':
      return null

    case 'rewrite_query':
      return (
        <div className="step-detail">
          <div className="kv">
            <span className="kv-k">{'original \u2192'}</span>
            <span className="kv-v">{String(d.original ?? d.original_query ?? '')}</span>
          </div>
          <div className="kv">
            <span className="kv-k">{'rewritten \u2192'}</span>
            <span className="kv-v">{String(d.rewritten ?? d.rewritten_query ?? '')}</span>
          </div>
        </div>
      )

    case 'vector_search':
    case 'document_search':
      return <VectorSearchDetail details={d} />

    case 'synthesize_response':
      return (
        <div className="step-detail">
          <div className="kv">
            <span className="kv-k">{'context \u2192'}</span>
            <span className="kv-v">{String(d.context_docs ?? (d.documents as unknown[])?.length ?? d.num_documents ?? 0)} documents</span>
          </div>
          {!!(d.response || d.answer) && (
            <div className="answer-box">{String(d.response ?? d.answer ?? '')}</div>
          )}
        </div>
      )

    case 'select_tool': {
      const args = d.arguments && typeof d.arguments === 'object'
        ? Object.entries(d.arguments as Record<string, unknown>)
        : []
      return (
        <div className="step-detail">
          <div className="kv">
            <span className="kv-k">{'tool \u2192'}</span>
            <span className="kv-v">{String(d.tool ?? d.tool_name ?? '')}</span>
          </div>
          {args.map(([k, v]) => (
            <div className="kv" key={k}>
              <span className="kv-k">{`${k} \u2192`}</span>
              <span className="kv-v">{String(v)}</span>
            </div>
          ))}
        </div>
      )
    }

    case 'execute_tool':
      return <ExecuteToolDetail details={d} />

    case 'format_response':
      return (
        <div className="step-detail">
          {d.response
            ? <div className="answer-box">{String(d.response)}</div>
            : <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Formatted result for human reading</span>}
        </div>
      )

    case 'direct_response':
      return (
        <div className="step-detail">
          {d.response
            ? <div className="answer-box">{String(d.response)}</div>
            : <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>Answered directly &mdash; no tools required</span>}
        </div>
      )

    case 'confidence_assessment':
      return <ConfidenceDetail details={d} />

    case 'cloud_inference': {
      const calls = (d.calls ?? []) as { tool: string; arguments: Record<string, unknown> }[]
      return (
        <div className="step-detail">
          {calls.length > 0 ? (
            calls.map((c, i) => (
              <div key={i}>
                <div className="kv">
                  <span className="kv-k">{'tool \u2192'}</span>
                  <span className="kv-v">{c.tool}</span>
                </div>
                {Object.entries(c.arguments).map(([k, v]) => (
                  <div className="kv" key={k}>
                    <span className="kv-k">{k} {'\u2192'}</span>
                    <span className="kv-v mono">{String(v)}</span>
                  </div>
                ))}
              </div>
            ))
          ) : d.cost != null ? (
            <div className="kv">
              <span className="kv-k">{'cost \u2192'}</span>
              <span className="kv-v">${Number(d.cost).toFixed(6)}</span>
            </div>
          ) : null}
        </div>
      )
    }

    case 'cloud_escalation':
      return (
        <div className="step-detail">
          {d.reason != null && String(d.reason).length > 0 && (
            <div className="kv">
              <span className="kv-k">{'reason \u2192'}</span>
              <span className="kv-v">{String(d.reason)}</span>
            </div>
          )}
          {d.tokens != null && (
            <div className="kv">
              <span className="kv-k">{'tokens \u2192'}</span>
              <span className="kv-v">{Number(d.tokens).toLocaleString('en-US')}</span>
            </div>
          )}
          {d.cost != null && (
            <div className="kv">
              <span className="kv-k">{'cost \u2192'}</span>
              <span className="kv-v">${Number(d.cost).toFixed(6)}</span>
            </div>
          )}
          {d.bytes_sent != null && (
            <div className="kv">
              <span className="kv-k">{'bytes sent \u2192'}</span>
              <span className="kv-v">{Number(d.bytes_sent).toLocaleString('en-US')}</span>
            </div>
          )}
          {!!d.response && (
            <div className="answer-box">{String(d.response)}</div>
          )}
        </div>
      )

    case 'decompose_query': {
      const planSteps = (d.steps ?? d.sub_queries ?? []) as string[]
      return (
        <div className="step-detail">
          <div className="kv">
            <span className="kv-k">{'plan \u2192'}</span>
            <span className="kv-v">{planSteps.length} step{planSteps.length !== 1 ? 's' : ''}</span>
          </div>
          {planSteps.map((s, i) => (
            <div className="kv" key={i}>
              <span className="kv-k">{i + 1}.</span>
              <span className="kv-v">{s}</span>
            </div>
          ))}
        </div>
      )
    }

    case 'concretize_step':
      return (
        <div className="step-detail">
          <div className="kv">
            <span className="kv-k">{'original \u2192'}</span>
            <span className="kv-v">{String(d.original ?? '')}</span>
          </div>
          <div className="kv">
            <span className="kv-k">{'concrete \u2192'}</span>
            <span className="kv-v">{String(d.concrete ?? d.concretized ?? '')}</span>
          </div>
        </div>
      )

    case 'extract_data':
      return <ExtractionDetail details={d} />

    case 'voice_transcribe':
      return (
        <div className="step-detail">
          <div className="answer-box">{String(d.text ?? d.transcription ?? '')}</div>
          {d.language != null && (
            <span className="lang-badge">{String(d.language)}</span>
          )}
        </div>
      )

    case 'voice_synthesize':
      return (
        <div className="step-detail">
          {d.language != null && (
            <div className="kv">
              <span className="kv-k">{'language \u2192'}</span>
              <span className="kv-v">{String(d.language)}</span>
            </div>
          )}
        </div>
      )

    default:
      return null
  }
}

function VectorSearchDetail({ details: d }: { details: Record<string, unknown> }) {
  const docs = (d.documents ?? d.results ?? []) as Array<Record<string, unknown>>
  const [openDocs, setOpenDocs] = useState<Record<number, boolean>>({})
  const barsRef = useRef<(HTMLDivElement | null)[]>([])

  const toggleDoc = (i: number) => {
    setOpenDocs(prev => ({ ...prev, [i]: !prev[i] }))
  }

  useEffect(() => {
    barsRef.current.forEach((el, i) => {
      if (!el) return
      const score = Number((docs[i] as Record<string, unknown>)?.score ?? 0)
      const pct = Math.round(score * 100)
      requestAnimationFrame(() => {
        el.style.width = `${pct}%`
      })
    })
  }, [docs])

  if (!docs.length) {
    return (
      <div className="step-detail" style={{ color: 'var(--text-muted)' }}>
        No documents retrieved
      </div>
    )
  }

  return (
    <div className="step-detail">
      <div className="kv">
        <span className="kv-k">{'query \u2192'}</span>
        <span className="kv-v">{String(d.query ?? '')}</span>
      </div>
      <div className="doc-list">
        {docs.map((doc, i) => {
          const meta = doc.metadata as Record<string, unknown> | undefined
          const title = String(meta?.title ?? doc.title ?? doc.id ?? doc.source ?? 'Document')
          const score = Number(doc.score ?? 0)
          const pct = Math.round(score * 100)
          const content = String(doc.content ?? doc.text ?? '')
          return (
            <div className={`doc-item${openDocs[i] ? ' open' : ''}`} key={i}>
              <div className="doc-row" onClick={() => toggleDoc(i)}>
                <span className="doc-expand">{'\u25B6'}</span>
                <span className="doc-title">{title}</span>
                {doc.score != null && (
                  <>
                    <div className="doc-score-track">
                      <div
                        ref={el => { barsRef.current[i] = el }}
                        className="doc-score-fill"
                        data-pct={pct}
                        style={{ width: 0 }}
                      />
                    </div>
                    <span className="doc-score-val">{score.toFixed(2)}</span>
                  </>
                )}
              </div>
              {openDocs[i] && (
                <div className="doc-content">
                  {content.slice(0, 500)}{content.length > 500 ? '\u2026' : ''}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

function ExecuteToolDetail({ details: d }: { details: Record<string, unknown> }) {
  const toolName = String(d.tool ?? d.tool_name ?? '').toLowerCase()
  const result = d.result as Record<string, unknown> | undefined

  if (!result) return null

  if (toolName === 'calculator' || (result as Record<string, unknown>)?.result != null) {
    return (
      <div className="step-detail">
        <div className="calc-result">
          {'= '}{String((result as Record<string, unknown>)?.result ?? result)}
        </div>
      </div>
    )
  }

  if (result?.columns && result?.rows) {
    const headers = (result.columns ?? []) as string[]
    const rawRows = (result.rows ?? []) as unknown[]

    return (
      <div className="step-detail">
        {headers.length > 0 ? (
          <div className="sql-wrap">
            <table className="sql-table">
              <thead>
                <tr>
                  {headers.map((h, i) => <th key={i}>{h}</th>)}
                </tr>
              </thead>
              <tbody>
                {rawRows.map((row, ri) => {
                  const cells = Array.isArray(row)
                    ? row
                    : headers.map(h => (row as Record<string, unknown>)[h])
                  return (
                    <tr key={ri}>
                      {cells.map((cell, ci) => (
                        <td key={ci}>{String(cell ?? '')}</td>
                      ))}
                    </tr>
                  )
                })}
              </tbody>
            </table>
            <div className="sql-more">{rawRows.length} rows</div>
          </div>
        ) : (
          <pre className="json-fallback">{JSON.stringify(result, null, 2)}</pre>
        )}
      </div>
    )
  }

  return (
    <div className="step-detail">
      <div className="kv-v">{JSON.stringify(result, null, 2)}</div>
    </div>
  )
}

function ExtractionDetail({ details: d }: { details: Record<string, unknown> }) {
  const [showRaw, setShowRaw] = useState(false)
  const extracted = d.extracted as Record<string, unknown> | undefined
  const rawOutput = String(d.raw_output ?? '')
  const stored = Boolean(d.stored)

  if (!extracted) {
    return (
      <div className="step-detail">
        <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
          Extraction failed: {String(d.error ?? 'unknown error')}
        </span>
      </div>
    )
  }

  // Display fields (exclude metadata)
  const displayFields = Object.entries(extracted).filter(
    ([k]) => !['source_document', 'extracted_at'].includes(k)
  )

  return (
    <div className="step-detail">
      {stored && (
        <div className="extraction-badge">
          {'\u2713'} Stored in competitors table
        </div>
      )}
      <table className="extraction-table">
        <tbody>
          {displayFields.map(([k, v]) => (
            <tr key={k}>
              <td className="extraction-field">{k}</td>
              <td className={`extraction-value${v === null ? ' null' : ''}`}>
                {v === null ? 'null' : typeof v === 'number'
                  ? v.toLocaleString('en-US')
                  : String(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <button
        className="raw-toggle"
        onClick={() => setShowRaw(prev => !prev)}
      >
        {showRaw ? '\u25BC' : '\u25B6'} Raw LLM output
      </button>
      {showRaw && (
        <pre className="extraction-raw">{rawOutput}</pre>
      )}
    </div>
  )
}

function ConfidenceDetail({ details: d }: { details: Record<string, unknown> }) {
  const score = Number(d.score ?? d.confidence ?? 0)
  const pct = Math.round(score * 100)
  const threshold = Number(d.threshold ?? 0.7)
  const factors = (d.factors ?? []) as string[]
  const barRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (barRef.current) {
      requestAnimationFrame(() => {
        barRef.current!.style.width = `${pct}%`
      })
    }
  }, [pct])

  return (
    <div className="step-detail">
      <div className="confidence-score">{pct}%</div>
      <div className="score-bar">
        <div ref={barRef} className="score-fill" style={{ width: 0 }} />
      </div>
      <div className="threshold-msg">
        {score >= threshold
          ? `Above threshold (${Math.round(threshold * 100)}%)`
          : `Below threshold (${Math.round(threshold * 100)}%) -- may escalate`}
      </div>
      {factors.length > 0 && (
        <ul className="factors-list">
          {factors.map((f, i) => <li key={i}>{f}</li>)}
        </ul>
      )}
    </div>
  )
}
