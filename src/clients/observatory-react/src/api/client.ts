import type {
  QueryResult,
  CompareResult,
  EscalateResult,
  ExecutionStep,
  HealthStatus,
  EvalResult,
  EvalResults,
  DemoMode,
  ThreePathResult,
} from '../types/api.ts'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''
const JSON_HEADERS = { 'Content-Type': 'application/json' }

export async function queryAgent(
  query: string,
  images?: string[],
  backend: DemoMode = 'multi-models',
): Promise<QueryResult> {
  const body: Record<string, unknown> = { query }
  if (images?.length) body.images = images
  if (backend !== 'multi-models' && backend !== 'all') body.backend = backend
  const r = await fetch(`${API_BASE}/query`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`Query failed: ${r.status}`)
  return r.json()
}

export async function queryCompareAll(
  query: string,
  images?: string[],
): Promise<ThreePathResult> {
  const body: Record<string, unknown> = { query }
  if (images?.length) body.images = images
  const r = await fetch(`${API_BASE}/query/compare-all`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`Compare-all failed: ${r.status}`)
  return r.json()
}

export async function compareQuery(
  query: string,
  images?: string[],
): Promise<CompareResult> {
  const body: Record<string, unknown> = { query }
  if (images?.length) body.images = images
  const r = await fetch(`${API_BASE}/compare`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error(`Compare failed: ${r.status}`)
  return r.json()
}

export async function escalateQuery(query: string): Promise<EscalateResult> {
  const r = await fetch(`${API_BASE}/escalate`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ query }),
  })
  if (!r.ok) throw new Error(`Escalate failed: ${r.status}`)
  return r.json()
}

export async function escalateQueryStream(
  query: string,
  callbacks: {
    onToken: (text: string) => void
    onDone: (meta: EscalateResult) => void
    onError: (msg: string) => void
  },
): Promise<void> {
  const resp = await fetch(`${API_BASE}/escalate/stream`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ query }),
  })
  if (!resp.ok) {
    callbacks.onError(`Escalate failed: ${resp.status}`)
    return
  }

  const reader = resp.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const parts = buffer.split('\n\n')
    buffer = parts.pop()!
    for (const part of parts) {
      const lines = part.split('\n')
      let evType = ''
      let evData = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) evType = line.slice(7)
        if (line.startsWith('data: ')) evData = line.slice(6)
      }
      if (!evType || !evData) continue
      try {
        const parsed = JSON.parse(evData)
        if (evType === 'token') callbacks.onToken(parsed.text)
        else if (evType === 'done') callbacks.onDone(parsed as EscalateResult)
        else if (evType === 'error') callbacks.onError(parsed.message)
      } catch { /* ignore parse errors */ }
    }
  }
}

export async function queryAgentStream(
  query: string,
  callbacks: {
    onStep: (step: ExecutionStep) => void
    onToken: (text: string) => void
    onDone: (meta: { intent: string; execution_time_ms: number; total_tokens: number; prompt_tokens: number; completion_tokens: number; models_used: string[]; cloud_cost?: number | null }) => void
    onError: (msg: string) => void
  },
  images?: string[],
  backend: DemoMode = 'multi-models',
  documentId?: string | null,
): Promise<void> {
  const body: Record<string, unknown> = { query }
  if (images?.length) body.images = images
  if (backend !== 'multi-models' && backend !== 'all') body.backend = backend
  if (documentId) body.document_id = documentId

  const resp = await fetch(`${API_BASE}/query/stream`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  })
  if (!resp.ok) {
    callbacks.onError(`Query stream failed: ${resp.status}`)
    return
  }

  const reader = resp.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const parts = buffer.split('\n\n')
    buffer = parts.pop()!
    for (const part of parts) {
      const lines = part.split('\n')
      let evType = ''
      let evData = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) evType = line.slice(7)
        if (line.startsWith('data: ')) evData = line.slice(6)
      }
      if (!evType || !evData) continue
      try {
        const parsed = JSON.parse(evData)
        if (evType === 'step') callbacks.onStep(parsed as ExecutionStep)
        else if (evType === 'token') callbacks.onToken(parsed.text)
        else if (evType === 'done') callbacks.onDone(parsed)
        else if (evType === 'error') callbacks.onError(parsed.message)
      } catch { /* ignore parse errors */ }
    }
  }
}

export async function swapModels(
  mode: 'base' | 'finetuned',
): Promise<{ status: string; detail?: string; message?: string }> {
  const r = await fetch(`${API_BASE}/models/swap`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ mode }),
  })
  return r.json()
}

export async function getModelMode(): Promise<{ mode: string }> {
  const r = await fetch(`${API_BASE}/models/mode`)
  return r.json()
}

export async function fetchHealth(): Promise<HealthStatus> {
  const r = await fetch(`${API_BASE}/health`)
  if (!r.ok) throw new Error(`Health check failed: ${r.status}`)
  return r.json()
}

export async function runEval(model = 'gemma3'): Promise<EvalResult> {
  const r = await fetch(`${API_BASE}/eval`, {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify({ model }),
  })
  if (!r.ok) throw new Error(`Eval failed: ${r.status}`)
  return r.json()
}

export async function getEvalResults(): Promise<EvalResults> {
  const r = await fetch(`${API_BASE}/eval/results`)
  return r.json()
}

export async function resetEval(): Promise<void> {
  await fetch(`${API_BASE}/eval/reset`, { method: 'POST' })
}

export async function setNetworkMode(): Promise<{ network_mode: string }> {
  const r = await fetch(`${API_BASE}/network-mode`, { method: 'POST' })
  return r.json()
}

export async function setRoutingMode(): Promise<{ routing_mode: string }> {
  const r = await fetch(`${API_BASE}/routing-mode`, { method: 'POST' })
  return r.json()
}

export async function uploadDocument(
  file: File,
  onEvent: (event: { stage: string; message?: string; detail: Record<string, unknown> }) => void,
): Promise<void> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE}/upload-document`, {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: `HTTP ${response.status}` }))
    throw new Error(err.detail || `HTTP ${response.status}`)
  }

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const lines = buffer.split('\n')
    buffer = lines.pop()!
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          onEvent(JSON.parse(line.slice(6)))
        } catch { /* ignore parse errors */ }
      }
    }
  }
}

export interface ExtractionResult {
  success: boolean
  extracted: Record<string, unknown> | null
  raw_output: string | null
  stored: boolean
  error: string | null
  execution_time_ms: number
}

export async function extractData(documentId: string): Promise<ExtractionResult> {
  const res = await fetch(`${API_BASE}/extract`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ document_id: documentId }),
  })
  return res.json()
}

export async function getCompetitors(): Promise<{ competitors: Record<string, unknown>[]; count: number }> {
  const res = await fetch(`${API_BASE}/competitors`)
  return res.json()
}

export async function voiceChat(
  blob: Blob,
  onEvent: (type: string, data: unknown) => void,
): Promise<void> {
  const form = new FormData()
  form.append('file', blob, 'recording.webm')

  const resp = await fetch(`${API_BASE}/voice/chat`, { method: 'POST', body: form })
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

  const reader = resp.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const parts = buffer.split('\n\n')
    buffer = parts.pop()!
    for (const part of parts) {
      const lines = part.split('\n')
      let evType = ''
      let evData = ''
      for (const line of lines) {
        if (line.startsWith('event: ')) evType = line.slice(7)
        if (line.startsWith('data: ')) evData = line.slice(6)
      }
      if (evType && evData) {
        try {
          onEvent(evType, JSON.parse(evData))
        } catch { /* ignore parse errors */ }
      }
    }
  }
}
