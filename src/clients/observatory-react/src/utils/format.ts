import type { Intent } from '../types/api.ts'

export function formatMs(ms: number): string {
  if (ms == null) return '--'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(2)} s`
}

export function escapeHtml(s: unknown): string {
  if (s == null) return ''
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

export function intentLabel(intent: Intent | string): string {
  const map: Record<string, string> = {
    rag_query: 'RAG Query',
    tool_use: 'Tool Use',
    direct_answer: 'Direct Answer',
    image_query: 'Image Query',
    voice: 'Voice',
    document_chat: 'Document Chat',
  }
  return map[intent] ?? intent
}

export function intentBadgeLabel(intent: Intent | string): string {
  const map: Record<string, string> = {
    rag_query: 'RAG',
    tool_use: 'TOOL',
    direct_answer: 'DIRECT',
    image_query: 'IMAGE',
    voice: 'VOICE',
    document_chat: 'DOC',
  }
  return map[intent] ?? intent
}

export function shortModel(model: string): string {
  if (!model || model === 'local_execution') return 'local'
  return model.replace(/-ft$/, '').replace(/-merged$/, '')
}

export function modelKey(model: string): string {
  if (!model) return 'local'
  const m = model.toLowerCase()
  if (m.startsWith('embeddinggemma')) return 'embeddinggemma'
  if (m.startsWith('gemma3') || m.startsWith('gemma-3')) return 'gemma3'
  if (m === 'whisper') return 'whisper'
  if (m === 'piper') return 'piper'
  if (m === 'heuristic') return 'local'
  if (m.startsWith('qwen')) return 'qwen'
  if (m.startsWith('gpt-') || m.startsWith('claude')) return 'cloud'
  return 'local'
}

export function modelColor(model: string): string {
  const k = modelKey(model)
  const map: Record<string, string> = {
    gemma3: 'var(--c-gemma3)',
    embeddinggemma: 'var(--c-embedding)',
    local: 'var(--c-local)',
    qwen: 'var(--c-qwen)',
    whisper: '#e67e22',
    piper: '#9b59b6',
    cloud: 'var(--c-cloud)',
  }
  return map[k] ?? 'var(--c-local)'
}

export const ACTION_LABEL: Record<string, string> = {
  classify_intent: 'Classify Intent',
  rewrite_query: 'Rewrite Query',
  vector_search: 'Semantic Search',
  synthesize_response: 'Synthesize Response',
  select_tool: 'Select Tool',
  execute_tool: 'Execute Tool',
  format_response: 'Format Response',
  direct_response: 'Direct Response',
  analyse_image: 'Analyse Image',
  confidence_assessment: 'Confidence',
  cloud_escalation: 'Cloud Escalation',
  cloud_inference: 'Cloud Inference',
  decompose_query: 'Plan Steps',
  concretize_step: 'Concretize Step',
  voice_transcribe: 'Voice \u2192 Text',
  voice_synthesize: 'Text \u2192 Voice',
}

export const WF_ACTION_LABEL: Record<string, string> = {
  classify_intent: 'Classify',
  rewrite_query: 'Rewrite',
  vector_search: 'Search',
  synthesize_response: 'Synthesize',
  select_tool: 'Select tool',
  execute_tool: 'Execute',
  format_response: 'Format',
  direct_response: 'Direct',
  analyse_image: 'Vision',
  decompose_query: 'Plan',
  concretize_step: 'Concretize',
  confidence_assessment: 'Confidence',
  cloud_escalation: 'Escalation',
  cloud_inference: 'Cloud',
  voice_transcribe: 'Transcribe',
  voice_synthesize: 'TTS',
}

export const SUGGESTIONS = [
  "What's the pricing for the Enterprise plan?",
  "Calculate 23 deals \u00D7 $52,400 average deal size",
  "What integrations does the platform support?",
  "Show top 3 customers by revenue",
  "What are the support SLAs?",
  "What's 15% of $45,000?",
  "Which product tier generates the most revenue?",
  "How does our MRR break down by industry?",
  "What is Nextera's uptime SLA on the Enterprise plan, and how does it compare to OpenAI's API availability guarantees?",
  "Why did Meridian choose Azure over AWS and GCP?",
]

export const CLOUD_PRICING = {
  input_per_1m: 3.0,
  output_per_1m: 15.0,
}

export function formatResponse(text: string): string {
  if (!text) return ''

  const lines = text.split('\n')
  const parts: string[] = []
  let inList = false

  for (const line of lines) {
    // Headings: ### Title, ## Title, # Title
    const headingMatch = line.match(/^(#{1,4})\s+(.*)/)
    if (headingMatch) {
      if (inList) { parts.push('</ul>'); inList = false }
      const level = Math.min(headingMatch[1].length + 2, 6) // ### → h5, ## → h4
      parts.push(`<h${level}>${fmtInline(headingMatch[2])}</h${level}>`)
      continue
    }

    const bulletMatch = line.match(/^(\s*)[*\-]\s+(.*)/)
    if (bulletMatch) {
      const content = fmtInline(bulletMatch[2])
      if (!inList) {
        inList = true
        parts.push('<ul>')
      }
      parts.push(`<li>${content}</li>`)
    } else {
      if (inList) {
        parts.push('</ul>')
        inList = false
      }
      const trimmed = line.trim()
      if (!trimmed) {
        if (parts.length > 0 && parts[parts.length - 1] !== '</p><p>') {
          parts.push('</p><p>')
        }
      } else {
        parts.push(fmtInline(trimmed) + '<br>')
      }
    }
  }
  if (inList) parts.push('</ul>')

  let html = parts.join('')
  html = html.replace(/<br>(<\/p>|<ul>|$)/g, '$1')
  html = html.replace(/^<\/p><p>/, '')
  return `<p>${html}</p>`.replace(/<p><\/p>/g, '')
}

function fmtInline(s: string): string {
  return escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
}

export function formatBytes(b: number): string {
  if (b === 0) return '0 bytes'
  if (b >= 1024) return `${(b / 1024).toFixed(1)} KB`
  return `${b} B`
}
