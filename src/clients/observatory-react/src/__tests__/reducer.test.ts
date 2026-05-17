import { reducer } from '../state/reducer.ts'
import { initialState } from '../types/state.ts'
import type { AppState, Exchange } from '../types/state.ts'
import type { EvalResult } from '../types/api.ts'

function makeExchange(overrides: Partial<Exchange> = {}): Exchange {
  return {
    query: 'test query',
    images: [],
    imageDataUrls: [],
    result: null,
    ...overrides,
  }
}

function makeEvalResult(overrides: Partial<EvalResult> = {}): EvalResult {
  return {
    overall_accuracy: 0.9,
    per_class: {},
    overall_correct: 9,
    n: 10,
    ...overrides,
  }
}

describe('reducer', () => {
  let state: AppState

  beforeEach(() => {
    state = { ...initialState, exchanges: [], pendingImages: [], pendingImageDataUrls: [] }
    localStorage.clear()
  })

  describe('ADD_EXCHANGE', () => {
    it('adds an exchange to the array', () => {
      const exchange = makeExchange()
      const next = reducer(state, { type: 'ADD_EXCHANGE', exchange })
      expect(next.exchanges).toHaveLength(1)
      expect(next.exchanges[0]).toBe(exchange)
    })

    it('appends to existing exchanges', () => {
      const first = makeExchange({ query: 'first' })
      const second = makeExchange({ query: 'second' })
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange: first })
      next = reducer(next, { type: 'ADD_EXCHANGE', exchange: second })
      expect(next.exchanges).toHaveLength(2)
      expect(next.exchanges[1].query).toBe('second')
    })
  })

  describe('UPDATE_EXCHANGE', () => {
    it('updates a specific exchange by index', () => {
      const exchange = makeExchange()
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange })
      next = reducer(next, {
        type: 'UPDATE_EXCHANGE',
        idx: 0,
        updates: { query: 'updated query' },
      })
      expect(next.exchanges[0].query).toBe('updated query')
    })

    it('does not modify other exchanges', () => {
      const first = makeExchange({ query: 'first' })
      const second = makeExchange({ query: 'second' })
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange: first })
      next = reducer(next, { type: 'ADD_EXCHANGE', exchange: second })
      next = reducer(next, { type: 'UPDATE_EXCHANGE', idx: 0, updates: { query: 'changed' } })
      expect(next.exchanges[1].query).toBe('second')
    })
  })

  describe('SET_ACTIVE_IDX', () => {
    it('sets the active index', () => {
      const next = reducer(state, { type: 'SET_ACTIVE_IDX', idx: 3 })
      expect(next.activeIdx).toBe(3)
    })
  })

  describe('SET_LOADING', () => {
    it('sets loading to true', () => {
      const next = reducer(state, { type: 'SET_LOADING', loading: true })
      expect(next.loading).toBe(true)
    })

    it('sets loading to false', () => {
      const loaded = reducer(state, { type: 'SET_LOADING', loading: true })
      const next = reducer(loaded, { type: 'SET_LOADING', loading: false })
      expect(next.loading).toBe(false)
    })
  })

  describe('ADD_PENDING_IMAGE', () => {
    it('adds image to both arrays', () => {
      const next = reducer(state, {
        type: 'ADD_PENDING_IMAGE',
        base64: 'abc123',
        dataUrl: 'data:image/png;base64,abc123',
      })
      expect(next.pendingImages).toEqual(['abc123'])
      expect(next.pendingImageDataUrls).toEqual(['data:image/png;base64,abc123'])
    })

    it('appends multiple images', () => {
      let next = reducer(state, { type: 'ADD_PENDING_IMAGE', base64: 'a', dataUrl: 'da' })
      next = reducer(next, { type: 'ADD_PENDING_IMAGE', base64: 'b', dataUrl: 'db' })
      expect(next.pendingImages).toEqual(['a', 'b'])
      expect(next.pendingImageDataUrls).toEqual(['da', 'db'])
    })
  })

  describe('REMOVE_PENDING_IMAGE', () => {
    it('removes by index from both arrays', () => {
      let next = reducer(state, { type: 'ADD_PENDING_IMAGE', base64: 'a', dataUrl: 'da' })
      next = reducer(next, { type: 'ADD_PENDING_IMAGE', base64: 'b', dataUrl: 'db' })
      next = reducer(next, { type: 'REMOVE_PENDING_IMAGE', idx: 0 })
      expect(next.pendingImages).toEqual(['b'])
      expect(next.pendingImageDataUrls).toEqual(['db'])
    })
  })

  describe('CLEAR_IMAGES', () => {
    it('empties both arrays', () => {
      let next = reducer(state, { type: 'ADD_PENDING_IMAGE', base64: 'a', dataUrl: 'da' })
      next = reducer(next, { type: 'ADD_PENDING_IMAGE', base64: 'b', dataUrl: 'db' })
      next = reducer(next, { type: 'CLEAR_IMAGES' })
      expect(next.pendingImages).toEqual([])
      expect(next.pendingImageDataUrls).toEqual([])
    })
  })

  describe('UPDATE_TOKENS', () => {
    it('accumulates tokens', () => {
      let next = reducer(state, { type: 'UPDATE_TOKENS', prompt: 10, completion: 5, total: 15 })
      expect(next.sessionTokens).toEqual({ prompt: 10, completion: 5, total: 15 })
      next = reducer(next, { type: 'UPDATE_TOKENS', prompt: 20, completion: 10, total: 30 })
      expect(next.sessionTokens).toEqual({ prompt: 30, completion: 15, total: 45 })
    })
  })

  describe('SET_COMPARE_MODE', () => {
    it('toggles compare mode', () => {
      const next = reducer(state, { type: 'SET_COMPARE_MODE', enabled: true })
      expect(next.compareMode).toBe(true)
      const off = reducer(next, { type: 'SET_COMPARE_MODE', enabled: false })
      expect(off.compareMode).toBe(false)
    })
  })

  describe('SET_EVAL_LOADING', () => {
    it('sets eval loading', () => {
      const next = reducer(state, { type: 'SET_EVAL_LOADING', loading: true })
      expect(next.eval.loading).toBe(true)
    })
  })

  describe('SET_EVAL_RESULT', () => {
    it('first call sets before', () => {
      const evalResult = makeEvalResult()
      const next = reducer(state, { type: 'SET_EVAL_RESULT', data: evalResult })
      expect(next.eval.before).toBe(evalResult)
      expect(next.eval.after).toBeNull()
    })

    it('second call sets after', () => {
      const first = makeEvalResult({ overall_accuracy: 0.8 })
      const second = makeEvalResult({ overall_accuracy: 0.95 })
      let next = reducer(state, { type: 'SET_EVAL_RESULT', data: first })
      next = reducer(next, { type: 'SET_EVAL_RESULT', data: second })
      expect(next.eval.before).toBe(first)
      expect(next.eval.after).toBe(second)
    })
  })

  describe('RESET_EVAL', () => {
    it('clears eval state', () => {
      let next = reducer(state, { type: 'SET_EVAL_RESULT', data: makeEvalResult() })
      next = reducer(next, { type: 'SET_EVAL_LOADING', loading: true })
      next = reducer(next, { type: 'RESET_EVAL' })
      expect(next.eval).toEqual({ before: null, after: null, loading: false })
    })
  })

  describe('UPDATE_CLOUD_COST', () => {
    it('accumulates cost and bytes', () => {
      let next = reducer(state, { type: 'UPDATE_CLOUD_COST', cost: 0.01, bytes: 100 })
      expect(next.cloudCostActual).toBe(0.01)
      expect(next.cloudBytesSent).toBe(100)
      next = reducer(next, { type: 'UPDATE_CLOUD_COST', cost: 0.02, bytes: 200 })
      expect(next.cloudCostActual).toBeCloseTo(0.03)
      expect(next.cloudBytesSent).toBe(300)
    })
  })

  describe('CLEAR_HISTORY', () => {
    it('resets exchanges, tokens, costs', () => {
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange: makeExchange() })
      next = reducer(next, { type: 'UPDATE_TOKENS', prompt: 10, completion: 5, total: 15 })
      next = reducer(next, { type: 'UPDATE_CLOUD_COST', cost: 0.01, bytes: 100 })
      next = reducer(next, { type: 'SET_ACTIVE_IDX', idx: 0 })
      next = reducer(next, { type: 'CLEAR_HISTORY' })
      expect(next.exchanges).toEqual([])
      expect(next.activeIdx).toBe(-1)
      expect(next.sessionTokens).toEqual({ prompt: 0, completion: 0, total: 0 })
      expect(next.cloudCostActual).toBe(0)
      expect(next.cloudBytesSent).toBe(0)
    })

    it('optionally resets eval when resetEval is true', () => {
      let next = reducer(state, { type: 'SET_EVAL_RESULT', data: makeEvalResult() })
      next = reducer(next, { type: 'CLEAR_HISTORY', resetEval: true })
      expect(next.eval).toEqual({ before: null, after: null, loading: false })
    })

    it('preserves eval when resetEval is not set', () => {
      const evalResult = makeEvalResult()
      let next = reducer(state, { type: 'SET_EVAL_RESULT', data: evalResult })
      next = reducer(next, { type: 'CLEAR_HISTORY' })
      expect(next.eval.before).toBe(evalResult)
    })
  })

  describe('SET_THEME', () => {
    it('changes theme and writes to localStorage', () => {
      const next = reducer(state, { type: 'SET_THEME', theme: 'dark' })
      expect(next.theme).toBe('dark')
      expect(localStorage.getItem('ui-theme')).toBe('dark')
    })

    it('switches back to light', () => {
      let next = reducer(state, { type: 'SET_THEME', theme: 'dark' })
      next = reducer(next, { type: 'SET_THEME', theme: 'light' })
      expect(next.theme).toBe('light')
      expect(localStorage.getItem('ui-theme')).toBe('light')
    })
  })

  describe('SET_TRACE_COLLAPSED', () => {
    it('changes collapsed state and writes to localStorage', () => {
      const next = reducer(state, { type: 'SET_TRACE_COLLAPSED', collapsed: false })
      expect(next.traceCollapsed).toBe(false)
      expect(localStorage.getItem('trace-collapsed')).toBe('false')
    })
  })

  describe('SET_MODEL_MODE', () => {
    it('sets model mode', () => {
      const next = reducer(state, { type: 'SET_MODEL_MODE', mode: 'finetuned' })
      expect(next.modelMode).toBe('finetuned')
    })
  })

  describe('SET_MODEL_SWAPPING', () => {
    it('sets swapping flag', () => {
      const next = reducer(state, { type: 'SET_MODEL_SWAPPING', swapping: true })
      expect(next.modelSwapping).toBe(true)
    })
  })

  describe('SET_NETWORK_MODE', () => {
    it('sets network mode', () => {
      const next = reducer(state, { type: 'SET_NETWORK_MODE', mode: 'offline' })
      expect(next.networkMode).toBe('offline')
    })
  })

  describe('SET_ROUTING_MODE', () => {
    it('sets routing mode', () => {
      const next = reducer(state, { type: 'SET_ROUTING_MODE', mode: 'hybrid' })
      expect(next.routingMode).toBe('hybrid')
    })
  })

  describe('immutability', () => {
    it('does not mutate the original state', () => {
      const original = { ...initialState, exchanges: [], pendingImages: [], pendingImageDataUrls: [] }
      const frozen = JSON.parse(JSON.stringify(original))

      reducer(original, { type: 'ADD_EXCHANGE', exchange: makeExchange() })
      reducer(original, { type: 'SET_LOADING', loading: true })
      reducer(original, { type: 'SET_THEME', theme: 'dark' })
      reducer(original, { type: 'UPDATE_TOKENS', prompt: 10, completion: 5, total: 15 })

      expect(original.exchanges).toEqual(frozen.exchanges)
      expect(original.loading).toBe(frozen.loading)
      expect(original.sessionTokens).toEqual(frozen.sessionTokens)
    })
  })

  describe('APPEND_CLOUD_TOKEN', () => {
    it('appends text to streamingCloudText on the target exchange', () => {
      const exchange = makeExchange()
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange })
      next = reducer(next, { type: 'APPEND_CLOUD_TOKEN', idx: 0, text: 'Hello' })
      expect(next.exchanges[0].streamingCloudText).toBe('Hello')
    })

    it('accumulates tokens across multiple dispatches', () => {
      const exchange = makeExchange()
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange })
      next = reducer(next, { type: 'APPEND_CLOUD_TOKEN', idx: 0, text: 'Hello' })
      next = reducer(next, { type: 'APPEND_CLOUD_TOKEN', idx: 0, text: ' world' })
      expect(next.exchanges[0].streamingCloudText).toBe('Hello world')
    })

    it('does not modify other exchanges', () => {
      const first = makeExchange({ query: 'first' })
      const second = makeExchange({ query: 'second' })
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange: first })
      next = reducer(next, { type: 'ADD_EXCHANGE', exchange: second })
      next = reducer(next, { type: 'APPEND_CLOUD_TOKEN', idx: 0, text: 'token' })
      expect(next.exchanges[1].streamingCloudText).toBeUndefined()
    })

    it('initializes from undefined streamingCloudText', () => {
      const exchange = makeExchange()
      expect(exchange.streamingCloudText).toBeUndefined()
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange })
      next = reducer(next, { type: 'APPEND_CLOUD_TOKEN', idx: 0, text: 'first' })
      expect(next.exchanges[0].streamingCloudText).toBe('first')
    })

    it('can be cleared via UPDATE_EXCHANGE', () => {
      const exchange = makeExchange()
      let next = reducer(state, { type: 'ADD_EXCHANGE', exchange })
      next = reducer(next, { type: 'APPEND_CLOUD_TOKEN', idx: 0, text: 'partial' })
      next = reducer(next, { type: 'UPDATE_EXCHANGE', idx: 0, updates: { streamingCloudText: undefined } })
      expect(next.exchanges[0].streamingCloudText).toBeUndefined()
    })
  })

  describe('unknown action', () => {
    it('returns the same state for unknown action types', () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const next = reducer(state, { type: 'UNKNOWN_ACTION' } as any)
      expect(next).toBe(state)
    })
  })
})
