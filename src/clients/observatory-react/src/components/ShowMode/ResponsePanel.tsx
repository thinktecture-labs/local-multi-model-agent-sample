import type { ExecutionStep } from '../../types/api.ts'
import { formatMs, shortModel } from '../../utils/format.ts'
import { SmartCard } from './SmartCard.tsx'

interface ResponsePanelProps {
  query: string | null
  response: string | null
  steps: ExecutionStep[]
  visible: boolean
  /** Hero centred layout for the canned Jarvis greeting — no SmartCard,
   *  no pipeline, just the exchange centred and enlarged. */
  greetingMode?: boolean
}

function buildPipelineLabel(s: ExecutionStep): string {
  const model = shortModel(s.model)
  const ms = formatMs(s.duration_ms)

  // Show tool name for execute_tool steps
  if (s.action === 'execute_tool') {
    const tool = (s.details?.tool as string) || (s.details?.name as string) || ''
    return tool ? `${tool} ${ms}` : `${model} ${ms}`
  }

  return `${model} ${ms}`
}

export function ResponsePanel({ query, response, steps, visible, greetingMode = false }: ResponsePanelProps) {
  if (!visible || (!query && !response)) return null

  const pipeline = steps
    .filter(s => s.duration_ms > 0)
    .map(buildPipelineLabel)
    .join('  \u2192  ')

  const totalMs = steps.reduce((sum, s) => sum + s.duration_ms, 0)

  if (greetingMode) {
    // Hero layout \u2014 centred Q + answer, larger type, no SmartCard/pipeline.
    return (
      <div className="show-response greeting">
        {query && <div className="show-greeting-q">{query}</div>}
        {response && <div className="show-greeting-a">{response}</div>}
      </div>
    )
  }

  return (
    <div className={`show-response${response ? ' has-response' : ''}`}>
      {query && (
        <div className="show-query">
          <span className="show-q-label">Q</span>
          <span className="show-q-text">{query}</span>
        </div>
      )}

      {response && (
        <>
          <SmartCard response={response} steps={steps} />

          {pipeline && (
            <div className="show-pipeline">
              <span className="show-pipeline-chain">{pipeline}</span>
              <span className="show-pipeline-total">{formatMs(totalMs)} total</span>
            </div>
          )}
        </>
      )}
    </div>
  )
}
