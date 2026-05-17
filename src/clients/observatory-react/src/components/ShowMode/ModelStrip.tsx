import type { ExecutionStep, StepAction } from '../../types/api.ts'
import { formatMs } from '../../utils/format.ts'

interface ModelNode {
  id: string
  label: string
  role: string
  color: string
  matchActions: StepAction[]
  matchModels: RegExp
}

const MODELS: ModelNode[] = [
  { id: 'whisper',   label: 'Whisper',   role: 'STT',       color: '#f97316', matchActions: ['voice_transcribe'],           matchModels: /whisper/i },
  { id: 'logreg',    label: 'LogReg',    role: 'Intent',    color: '#8b5cf6', matchActions: ['classify_intent'],            matchModels: /logreg/i },
  { id: 'qwen',      label: 'Qwen 4B',   role: 'Tools',     color: '#ff584f', matchActions: ['select_tool', 'execute_tool', 'decompose_query', 'concretize_step'], matchModels: /qwen|toolcalling/i },
  { id: 'gemma3',    label: 'Gemma3',    role: 'Synthesis',  color: '#0170B9', matchActions: ['synthesize_response', 'direct_response', 'rewrite_query', 'format_response'], matchModels: /gemma3|gemma/i },
  { id: 'embedding', label: 'Embedding', role: 'Search',     color: '#34A853', matchActions: ['vector_search', 'document_search'], matchModels: /embedding/i },
]

interface ModelStripProps {
  steps: ExecutionStep[]
  activeAction: StepAction | null
  /** When true, all five nodes render as "online" — used during the canned
   *  Jarvis greeting where there are no real agent steps but we want the
   *  strip to visually echo "All five models online, sir." */
  greetingMode?: boolean
}

export function ModelStrip({ steps, activeAction, greetingMode = false }: ModelStripProps) {
  return (
    <div className={`show-model-strip${greetingMode ? ' greeting' : ''}`}>
      {MODELS.map((model, idx) => {
        // Match by action name OR by model string in the step
        const matchedStep = steps.find(s =>
          model.matchActions.includes(s.action) || model.matchModels.test(s.model || '')
        )
        const isActive = activeAction !== null && (
          model.matchActions.includes(activeAction)
        )
        const isDone = !!matchedStep || greetingMode

        // Sum durations if multiple steps match this model
        const totalMs = steps
          .filter(s => model.matchActions.includes(s.action) || model.matchModels.test(s.model || ''))
          .reduce((sum, s) => sum + s.duration_ms, 0)

        const lit = isDone || isActive

        return (
          <div
            key={model.id}
            className={`show-model-node${isActive ? ' active' : ''}${isDone ? ' done' : ''}${greetingMode ? ' greeting' : ''}`}
            style={{
              '--node-color': model.color,
              '--node-index': idx,
              borderColor: lit ? model.color : undefined,
            } as React.CSSProperties}
          >
            <div className="show-model-dot" style={{ background: lit ? model.color : undefined }} />
            <span className="show-model-label">{model.label}</span>
            <span className="show-model-role">{model.role}</span>
            {isDone && !greetingMode && totalMs > 0 && (
              <span className="show-model-ms">{formatMs(totalMs)}</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
