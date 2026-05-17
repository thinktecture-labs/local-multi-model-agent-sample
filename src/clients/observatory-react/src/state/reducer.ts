import type { AppState, Exchange } from '../types/state.ts'
import type { EvalResult, ExecutionStep, DemoMode } from '../types/api.ts'

export type Action =
  | { type: 'ADD_EXCHANGE'; exchange: Exchange }
  | { type: 'UPDATE_EXCHANGE'; idx: number; updates: Partial<Exchange> }
  | { type: 'APPEND_CLOUD_TOKEN'; idx: number; text: string }
  | { type: 'APPEND_COLUMN_TOKEN'; idx: number; key: 'multi_models' | 'qwen' | 'cloud'; text: string }
  | { type: 'APPEND_STEP'; idx: number; step: ExecutionStep }
  | { type: 'APPEND_TOKEN'; idx: number; text: string }
  | { type: 'FINALIZE_STREAM'; idx: number; meta: { intent: string; execution_time_ms: number; total_tokens: number; prompt_tokens: number; completion_tokens: number; models_used: string[]; cloud_cost?: number | null } }
  | { type: 'SET_ACTIVE_IDX'; idx: number }
  | { type: 'SET_LOADING'; loading: boolean }
  | { type: 'ADD_PENDING_IMAGE'; base64: string; dataUrl: string }
  | { type: 'REMOVE_PENDING_IMAGE'; idx: number }
  | { type: 'CLEAR_IMAGES' }
  | { type: 'UPDATE_TOKENS'; prompt: number; completion: number; total: number }
  | { type: 'SET_COMPARE_MODE'; enabled: boolean }
  | { type: 'SET_DEMO_MODE'; mode: DemoMode }
  | { type: 'SET_EVAL_LOADING'; loading: boolean }
  | { type: 'SET_EVAL_RESULT'; data: EvalResult }
  | { type: 'RESET_EVAL' }
  | { type: 'UPDATE_CLOUD_COST'; cost: number; bytes: number }
  | { type: 'CLEAR_HISTORY'; resetEval?: boolean }
  | { type: 'SET_THEME'; theme: 'dark' | 'light' }
  | { type: 'SET_TRACE_COLLAPSED'; collapsed: boolean }
  | { type: 'SET_MODEL_MODE'; mode: 'base' | 'finetuned' }
  | { type: 'SET_MODEL_SWAPPING'; swapping: boolean }
  | { type: 'SET_NETWORK_MODE'; mode: 'online' | 'offline' }
  | { type: 'SET_ROUTING_MODE'; mode: 'local' | 'hybrid' }
  | { type: 'SET_ACTIVE_DOCUMENT'; documentId: string; documentName: string }
  | { type: 'CLEAR_ACTIVE_DOCUMENT' }
  | { type: 'CLEAR_COLUMN_STREAMING'; idx: number; key: 'multi_models' | 'qwen' | 'cloud' }

export function reducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'ADD_EXCHANGE':
      return {
        ...state,
        exchanges: [...state.exchanges, action.exchange],
      }

    case 'UPDATE_EXCHANGE': {
      const exchanges = state.exchanges.map((ex, i) =>
        i === action.idx ? { ...ex, ...action.updates } : ex,
      )
      return { ...state, exchanges }
    }

    case 'APPEND_CLOUD_TOKEN': {
      const exchanges = state.exchanges.map((ex, i) =>
        i === action.idx
          ? { ...ex, streamingCloudText: (ex.streamingCloudText ?? '') + action.text }
          : ex,
      )
      return { ...state, exchanges }
    }

    case 'APPEND_COLUMN_TOKEN': {
      const exchanges = state.exchanges.map((ex, i) =>
        i === action.idx
          ? {
              ...ex,
              threePathStreaming: {
                ...ex.threePathStreaming,
                [action.key]: (ex.threePathStreaming?.[action.key] ?? '') + action.text,
              },
            }
          : ex,
      )
      return { ...state, exchanges }
    }

    case 'CLEAR_COLUMN_STREAMING': {
      const exchanges = state.exchanges.map((ex, i) =>
        i === action.idx
          ? {
              ...ex,
              threePathStreaming: {
                ...ex.threePathStreaming,
                [action.key]: undefined,
              },
            }
          : ex,
      )
      return { ...state, exchanges }
    }

    case 'APPEND_STEP': {
      const exchanges = state.exchanges.map((ex, i) => {
        if (i !== action.idx || !ex.result) return ex
        return { ...ex, result: { ...ex.result, steps: [...ex.result.steps, action.step] } }
      })
      return { ...state, exchanges }
    }

    case 'APPEND_TOKEN': {
      const exchanges = state.exchanges.map((ex, i) =>
        i === action.idx
          ? { ...ex, streamingText: (ex.streamingText ?? '') + action.text }
          : ex,
      )
      return { ...state, exchanges }
    }

    case 'FINALIZE_STREAM': {
      const exchanges = state.exchanges.map((ex, i) => {
        if (i !== action.idx) return ex
        return {
          ...ex,
          streamingText: undefined,
          result: {
            ...ex.result!,
            intent: action.meta.intent as never,
            response: ex.streamingText ?? ex.result?.response ?? '',
            execution_time_ms: action.meta.execution_time_ms,
            total_tokens: action.meta.total_tokens,
            prompt_tokens: action.meta.prompt_tokens,
            completion_tokens: action.meta.completion_tokens,
            models_used: action.meta.models_used,
            cloud_cost: action.meta.cloud_cost ?? undefined,
          },
        }
      })
      return { ...state, exchanges }
    }

    case 'SET_ACTIVE_IDX':
      return { ...state, activeIdx: action.idx }

    case 'SET_LOADING':
      return { ...state, loading: action.loading }

    case 'ADD_PENDING_IMAGE':
      return {
        ...state,
        pendingImages: [...state.pendingImages, action.base64],
        pendingImageDataUrls: [...state.pendingImageDataUrls, action.dataUrl],
      }

    case 'REMOVE_PENDING_IMAGE':
      return {
        ...state,
        pendingImages: state.pendingImages.filter((_, i) => i !== action.idx),
        pendingImageDataUrls: state.pendingImageDataUrls.filter((_, i) => i !== action.idx),
      }

    case 'CLEAR_IMAGES':
      return { ...state, pendingImages: [], pendingImageDataUrls: [] }

    case 'UPDATE_TOKENS':
      return {
        ...state,
        sessionTokens: {
          prompt: state.sessionTokens.prompt + action.prompt,
          completion: state.sessionTokens.completion + action.completion,
          total: state.sessionTokens.total + action.total,
        },
      }

    case 'SET_COMPARE_MODE':
      return { ...state, compareMode: action.enabled }

    case 'SET_DEMO_MODE':
      return { ...state, demoMode: action.mode }

    case 'SET_EVAL_LOADING':
      return { ...state, eval: { ...state.eval, loading: action.loading } }

    case 'SET_EVAL_RESULT': {
      if (!state.eval.before) {
        return { ...state, eval: { ...state.eval, before: action.data } }
      }
      return { ...state, eval: { ...state.eval, after: action.data } }
    }

    case 'RESET_EVAL':
      return { ...state, eval: { before: null, after: null, loading: false } }

    case 'UPDATE_CLOUD_COST':
      return {
        ...state,
        cloudCostActual: state.cloudCostActual + action.cost,
        cloudBytesSent: state.cloudBytesSent + action.bytes,
      }

    case 'CLEAR_HISTORY':
      return {
        ...state,
        exchanges: [],
        activeIdx: -1,
        sessionTokens: { prompt: 0, completion: 0, total: 0 },
        cloudCostActual: 0,
        cloudBytesSent: 0,
        activeDocumentId: null,
        activeDocumentName: null,
        ...(action.resetEval
          ? { eval: { before: null, after: null, loading: false } }
          : {}),
      }

    case 'SET_THEME':
      localStorage.setItem('ui-theme', action.theme)
      return { ...state, theme: action.theme }

    case 'SET_TRACE_COLLAPSED':
      localStorage.setItem('trace-collapsed', String(action.collapsed))
      return { ...state, traceCollapsed: action.collapsed }

    case 'SET_MODEL_MODE':
      return { ...state, modelMode: action.mode }

    case 'SET_MODEL_SWAPPING':
      return { ...state, modelSwapping: action.swapping }

    case 'SET_NETWORK_MODE':
      return { ...state, networkMode: action.mode }

    case 'SET_ROUTING_MODE':
      return { ...state, routingMode: action.mode }

    case 'SET_ACTIVE_DOCUMENT':
      return { ...state, activeDocumentId: action.documentId, activeDocumentName: action.documentName }

    case 'CLEAR_ACTIVE_DOCUMENT':
      return { ...state, activeDocumentId: null, activeDocumentName: null }

    default:
      return state
  }
}
