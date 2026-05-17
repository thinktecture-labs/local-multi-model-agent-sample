import { useState } from 'react'
import type { QueryResult, ExecutionStep } from '../../types/api.ts'

interface SourceDoc {
  title?: string
  content?: string
  source?: string
}

// Pair each select_tool step with its following execute_tool step.
// Multi-step traces produce multiple pairs (e.g., SQL lookup → calculator).
function pairToolSteps(steps: ExecutionStep[]): Array<{ select: ExecutionStep; execute: ExecutionStep }> {
  const pairs: Array<{ select: ExecutionStep; execute: ExecutionStep }> = []
  let pending: ExecutionStep | null = null
  for (const s of steps) {
    if (s.action === 'select_tool') {
      pending = s
    } else if (s.action === 'execute_tool' && pending) {
      pairs.push({ select: pending, execute: s })
      pending = null
    }
  }
  return pairs
}

function renderToolPair(select: ExecutionStep, execute: ExecutionStep, index: number, totalPairs: number): React.ReactNode {
  const toolName = (select.details.tool as string | undefined) ?? ''
  const stepLabel = totalPairs > 1 ? `Step ${index + 1} · ` : ''

  if (toolName === 'calculator' || toolName.includes('calc')) {
    // Calculator's tool data is nested: details.result = { expression, result: <number> }
    const rawResult = execute.details.result
    const isResultObject = typeof rawResult === 'object' && rawResult !== null
    const innerResult = isResultObject
      ? (rawResult as { result?: unknown }).result
      : rawResult
    const expression = (select.details.arguments as Record<string, unknown> | undefined)?.expression
      ?? (isResultObject ? (rawResult as { expression?: unknown }).expression : undefined)
    return (
      <div key={index} className="source-tool">
        <span className="source-tool-label">{stepLabel}Calculator</span>
        <code>{String(expression ?? '')} = {String(innerResult ?? '')}</code>
      </div>
    )
  }

  if (toolName === 'sql_query' || toolName.includes('sql')) {
    const rawResult = execute.details.result as Record<string, unknown> | undefined
    const columns = (rawResult?.columns ?? execute.details.columns ?? []) as string[]
    const rawRows = (rawResult?.rows ?? execute.details.rows ?? []) as unknown[]
    return (
      <div key={index} className="source-tool">
        <span className="source-tool-label">{stepLabel}SQL Query</span>
        {columns.length > 0 && (
          <table className="source-sql-table">
            <thead>
              <tr>
                {columns.map((col, i) => (
                  <th key={i}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rawRows.map((row, ri) => {
                const cells = Array.isArray(row)
                  ? row
                  : columns.map(col => (row as Record<string, unknown>)[col])
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
        )}
      </div>
    )
  }

  return (
    <div key={index} className="source-tool">
      <span className="source-tool-label">{stepLabel}{toolName || 'Tool'}</span>
      <code>{JSON.stringify(execute.details.result ?? execute.details, null, 2)}</code>
    </div>
  )
}

export function SourcePanel({ result }: { result: QueryResult }) {
  const [open, setOpen] = useState(false)

  const intent = result.intent

  // RAG: find vector search step
  const vectorStep = result.steps.find((s) => s.action === 'vector_search')

  // tool_use: pair every select_tool with its execute_tool (multi-step support)
  const toolPairs = pairToolSteps(result.steps)

  let content: React.ReactNode = null

  if (intent === 'rag_query' && vectorStep) {
    const docs = (vectorStep.details.documents ?? []) as SourceDoc[]
    content = (
      <ul className="source-doc-list">
        {docs.map((doc, i) => (
          <li key={i} className="source-doc">
            <strong>{doc.title ?? doc.source ?? `Document ${i + 1}`}</strong>
            <span className="source-snippet">
              {(doc.content ?? '').slice(0, 100)}
              {(doc.content ?? '').length > 100 ? '...' : ''}
            </span>
          </li>
        ))}
      </ul>
    )
  } else if (intent === 'tool_use' && toolPairs.length > 0) {
    content = (
      <div className="source-tool-list">
        {toolPairs.map((pair, i) => renderToolPair(pair.select, pair.execute, i, toolPairs.length))}
      </div>
    )
  } else if (intent === 'direct_answer') {
    content = <em className="source-none">Model knowledge — no external source</em>
  } else if (intent === 'image_query') {
    content = <em className="source-none">Image analysis — visual input</em>
  } else {
    content = <em className="source-none">No source details available</em>
  }

  return (
    <div className={`source-panel ${open ? 'open' : ''}`}>
      <button className="source-toggle" onClick={() => setOpen(!open)}>
        <span className={`source-arrow ${open ? 'open' : ''}`}>&#9654;</span>
        Sources
      </button>
      {open && <div className="source-content">{content}</div>}
    </div>
  )
}
