export type Intent = 'rag_query' | 'tool_use' | 'direct_answer' | 'image_query' | 'voice' | 'document_chat'

export type StepAction =
  | 'classify_intent'
  | 'rewrite_query'
  | 'vector_search'
  | 'document_search'
  | 'synthesize_response'
  | 'select_tool'
  | 'execute_tool'
  | 'format_response'
  | 'direct_response'
  | 'analyse_image'
  | 'confidence_assessment'
  | 'cloud_escalation'
  | 'cloud_inference'
  | 'decompose_query'
  | 'concretize_step'
  | 'voice_transcribe'
  | 'voice_synthesize'
  | 'extract_data'

export interface ExecutionStep {
  action: StepAction
  model: string
  duration_ms: number
  details: Record<string, unknown>
  tokens_used: number
  prompt_tokens?: number
  completion_tokens?: number
}

export interface QueryResult {
  request_id?: string
  intent: Intent
  response: string
  execution_time_ms: number
  steps: ExecutionStep[]
  models_used: string[]
  total_tokens: number
  prompt_tokens?: number
  completion_tokens?: number
  escalated?: boolean
  cloud_response?: string
  cloud_model?: string
  cloud_latency_ms?: number
  cloud_cost?: number
  confidence?: number
}

export interface CompareResult {
  local_response: string
  local_latency_ms: number
  cloud_response?: string
  cloud_model?: string
  cloud_latency_ms?: number
  cloud_cost?: number
  estimated_cloud_cost?: number
  steps: ExecutionStep[]
  intent: Intent
  execution_time_ms: number
  total_tokens: number
  prompt_tokens?: number
  completion_tokens?: number
}

export type DemoMode = 'multi-models' | 'qwen' | 'cloud' | 'all'

export interface ThreePathResult {
  multi_models: QueryResult
  qwen: QueryResult | null
  cloud: QueryResult | null
}

export interface EscalateResult {
  cloud_response: string
  cloud_model: string
  cloud_latency_ms: number
  cloud_cost: number
  cloud_tokens: number
  cloud_bytes_sent: number
}

export interface HealthStatus {
  models: Record<string, boolean>
  document_count?: number
  interaction_count?: number
}

export interface GpuStats {
  available: boolean
  name: string
  backend: string
  vram_used_mb: number
  vram_total_mb: number
  utilization_pct: number
  temperature_c: number
}

export interface EnergyStats {
  backend: string
  sample_count: number
  gpu_power_now_w: number
  system_power_now_w: number
  total_wh: number
  total_queries: number
  wh_per_query: number
  co2_local_g: number
  co2_cloud_g: number
  electricity_cost_local: number
  estimated_cloud_wh: number
}

export interface EvalResult {
  overall_accuracy: number
  per_class: Record<string, { accuracy?: number; correct: number; n: number }>
  overall_correct: number
  n: number
}

export interface EvalResults {
  before: EvalResult | null
  after: EvalResult | null
}

export interface DocumentUploadEvent {
  stage: 'parsing' | 'chunking' | 'embedding' | 'indexed'
  message?: string
  detail: Record<string, unknown>
}

export interface VoiceTranscriptionEvent {
  text: string
  language: string
  duration_ms: number
}

export interface VoiceResponseEvent {
  text: string
  intent?: string
  duration_ms: number
}

export interface VoiceAudioEvent {
  url: string
  duration_ms: number
}

export interface StatsMessage {
  gpu: GpuStats
  energy: EnergyStats
}
