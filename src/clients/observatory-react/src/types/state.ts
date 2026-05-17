import type { QueryResult, CompareResult, EvalResult, DemoMode, ThreePathResult } from './api.ts'

export interface Exchange {
  query: string
  images: string[]
  imageDataUrls: string[]
  result: QueryResult | null
  compareResult?: CompareResult
  threePathResult?: ThreePathResult
  threePathStreaming?: Partial<Record<'multi_models' | 'qwen' | 'cloud', string>>
  voiceMode?: boolean
  streamingText?: string
  streamingCloudText?: string
}

export interface SessionTokens {
  prompt: number
  completion: number
  total: number
}

export interface AppState {
  exchanges: Exchange[]
  activeIdx: number
  loading: boolean
  pendingImages: string[]
  pendingImageDataUrls: string[]
  sessionTokens: SessionTokens
  cloudCostActual: number
  cloudBytesSent: number
  compareMode: boolean
  demoMode: DemoMode
  eval: {
    before: EvalResult | null
    after: EvalResult | null
    loading: boolean
  }
  theme: 'dark' | 'light'
  traceCollapsed: boolean
  modelMode: 'base' | 'finetuned'
  modelSwapping: boolean
  networkMode: 'online' | 'offline'
  routingMode: 'local' | 'hybrid'
  /** When set, queries are scoped to this uploaded document (document chat mode). */
  activeDocumentId: string | null
  activeDocumentName: string | null
}

export const initialState: AppState = {
  exchanges: [],
  activeIdx: -1,
  loading: false,
  pendingImages: [],
  pendingImageDataUrls: [],
  sessionTokens: { prompt: 0, completion: 0, total: 0 },
  cloudCostActual: 0,
  cloudBytesSent: 0,
  compareMode: false,
  demoMode: 'multi-models',
  eval: { before: null, after: null, loading: false },
  theme: (typeof localStorage !== 'undefined'
    ? (localStorage.getItem('ui-theme') as 'dark' | 'light') ?? 'light'
    : 'light'),
  traceCollapsed: typeof localStorage !== 'undefined'
    ? localStorage.getItem('trace-collapsed') !== 'false'
    : true,
  modelMode: 'base',
  modelSwapping: false,
  networkMode: 'online',
  routingMode: 'local',
  activeDocumentId: null,
  activeDocumentName: null,
}
