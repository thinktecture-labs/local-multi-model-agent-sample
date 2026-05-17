import type { HealthStatus } from '../../types/api.ts'

const PILLS: Array<{ key: string; label: string; model: string; title: string }> = [
  { key: 'LOGREG',    label: 'LogReg intent',       model: 'logreg',     title: 'Router — primary intent classifier, deterministic LogReg on embeddinggemma vectors (~10 ms, no LLM call). Handles ~93% of traffic.' },
  { key: 'INFERENCE', label: 'gemma3 1B',          model: 'inference',  title: 'Thinker — direct answers, query decomposition, tool-result synthesis, intent-classification fallback when LogReg model is absent (gemma3-ft 1B)' },
  { key: 'FUNCTION',  label: 'qwen3.5 4B FT',      model: 'function',   title: 'Doer — tool selection, argument extraction (Qwen3.5-4B fine-tuned)' },
  { key: 'EMBEDDING', label: 'embeddinggemma 308M', model: 'embedding',  title: 'Librarian — semantic search, document retrieval (embeddinggemma 308M)' },
  { key: 'VISION',    label: 'gemma3 4B-v',         model: 'vision',     title: 'Eye — image understanding, RAG synthesis, data extraction (gemma3-4B vision)' },
  { key: 'OCR',       label: 'GLM-OCR 0.9B',        model: 'ocr',        title: 'Reader — PDF text + table extraction (GLM-OCR, upload-time only)' },
  { key: 'WHISPER',   label: 'whisper STT',          model: 'whisper',    title: 'Ear — speech-to-text transcription (whisper.cpp)' },
  { key: 'PIPER',     label: 'piper TTS',            model: 'piper',      title: 'Voice — text-to-speech synthesis (piper TTS)' },
  { key: 'QWEN',      label: 'Qwen 3.5 MoE',         model: 'qwen',       title: 'Qwen 3.5 35B-A3B — single Mixture-of-Experts model comparison backend' },
  { key: 'CLOUD',     label: 'GPT-5.4',              model: 'cloud',      title: 'Cloud — GPT-5.4 for hybrid routing escalation' },
]

export function HealthPills({ health }: { health: HealthStatus | null }) {
  const models = health?.models ?? {}

  // PIPER isn't reported separately — infer from WHISPER
  const effective = { ...models }
  if (effective['WHISPER'] && !('PIPER' in effective)) {
    effective['PIPER'] = true
  }

  return (
    <div className="health-bar">
      {PILLS.map(({ key, label, model, title }) => {
        const isOn = !!effective[key]
        return (
          <div
            key={key}
            className={`model-pill${isOn ? ' online' : ' offline'}`}
            data-model={model}
            title={title}
          >
            <div className="dot"></div><span>{label}</span>
          </div>
        )
      })}
      <div className="doc-count">
        {health?.document_count != null ? `${health.document_count} chunks` : '\u2014 docs'}
      </div>
    </div>
  )
}
