import {
  queryAgent,
  compareQuery,
  escalateQuery,
  swapModels,
  getModelMode,
  fetchHealth,
  runEval,
  setNetworkMode,
  setRoutingMode,
} from '../api/client.ts'

function mockFetch(data: unknown, ok = true, status = 200) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: () => Promise.resolve(data),
  })
}

describe('API client', () => {
  const originalFetch = globalThis.fetch

  afterEach(() => {
    globalThis.fetch = originalFetch
  })

  describe('queryAgent', () => {
    it('sends POST /query with correct body', async () => {
      const result = { intent: 'direct_answer', response: 'hello', steps: [], total_tokens: 10 }
      globalThis.fetch = mockFetch(result)

      const data = await queryAgent('test question')

      expect(globalThis.fetch).toHaveBeenCalledWith('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'test question' }),
      })
      expect(data).toEqual(result)
    })

    it('includes images in body when provided', async () => {
      globalThis.fetch = mockFetch({})

      await queryAgent('test', ['img1', 'img2'])

      expect(globalThis.fetch).toHaveBeenCalledWith('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'test', images: ['img1', 'img2'] }),
      })
    })

    it('does not include images when array is empty', async () => {
      globalThis.fetch = mockFetch({})

      await queryAgent('test', [])

      expect(globalThis.fetch).toHaveBeenCalledWith('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'test' }),
      })
    })

    it('throws on non-ok response', async () => {
      globalThis.fetch = mockFetch({}, false, 500)
      await expect(queryAgent('test')).rejects.toThrow('Query failed: 500')
    })
  })

  describe('compareQuery', () => {
    it('sends POST /compare', async () => {
      const result = { local_response: 'local', cloud_response: 'cloud' }
      globalThis.fetch = mockFetch(result)

      const data = await compareQuery('test')

      expect(globalThis.fetch).toHaveBeenCalledWith('/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'test' }),
      })
      expect(data).toEqual(result)
    })

    it('throws on non-ok response', async () => {
      globalThis.fetch = mockFetch({}, false, 503)
      await expect(compareQuery('test')).rejects.toThrow('Compare failed: 503')
    })
  })

  describe('escalateQuery', () => {
    it('sends POST /escalate', async () => {
      const result = { cloud_response: 'escalated' }
      globalThis.fetch = mockFetch(result)

      const data = await escalateQuery('hard question')

      expect(globalThis.fetch).toHaveBeenCalledWith('/escalate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'hard question' }),
      })
      expect(data).toEqual(result)
    })

    it('throws on non-ok response', async () => {
      globalThis.fetch = mockFetch({}, false, 400)
      await expect(escalateQuery('test')).rejects.toThrow('Escalate failed: 400')
    })
  })

  describe('swapModels', () => {
    it('sends POST /models/swap with mode', async () => {
      const result = { status: 'ok' }
      globalThis.fetch = mockFetch(result)

      const data = await swapModels('finetuned')

      expect(globalThis.fetch).toHaveBeenCalledWith('/models/swap', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'finetuned' }),
      })
      expect(data).toEqual(result)
    })
  })

  describe('getModelMode', () => {
    it('sends GET /models/mode', async () => {
      const result = { mode: 'base' }
      globalThis.fetch = mockFetch(result)

      const data = await getModelMode()

      expect(globalThis.fetch).toHaveBeenCalledWith('/models/mode')
      expect(data).toEqual(result)
    })
  })

  describe('fetchHealth', () => {
    it('sends GET /health and returns data', async () => {
      const result = { models: { gemma3: true }, document_count: 5 }
      globalThis.fetch = mockFetch(result)

      const data = await fetchHealth()

      expect(globalThis.fetch).toHaveBeenCalledWith('/health')
      expect(data).toEqual(result)
    })

    it('throws on non-ok response', async () => {
      globalThis.fetch = mockFetch({}, false, 503)
      await expect(fetchHealth()).rejects.toThrow('Health check failed: 503')
    })
  })

  describe('runEval', () => {
    it('sends POST /eval with default model', async () => {
      const result = { overall_accuracy: 0.9, per_class: {}, overall_correct: 9, n: 10 }
      globalThis.fetch = mockFetch(result)

      const data = await runEval()

      expect(globalThis.fetch).toHaveBeenCalledWith('/eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'gemma3' }),
      })
      expect(data).toEqual(result)
    })

    it('sends POST /eval with custom model', async () => {
      globalThis.fetch = mockFetch({})

      await runEval('qwen')

      expect(globalThis.fetch).toHaveBeenCalledWith('/eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'qwen' }),
      })
    })

    it('throws on non-ok response', async () => {
      globalThis.fetch = mockFetch({}, false, 500)
      await expect(runEval()).rejects.toThrow('Eval failed: 500')
    })
  })

  describe('setNetworkMode', () => {
    it('sends POST /network-mode', async () => {
      const result = { network_mode: 'offline' }
      globalThis.fetch = mockFetch(result)

      const data = await setNetworkMode()

      expect(globalThis.fetch).toHaveBeenCalledWith('/network-mode', { method: 'POST' })
      expect(data).toEqual(result)
    })
  })

  describe('setRoutingMode', () => {
    it('sends POST /routing-mode', async () => {
      const result = { routing_mode: 'hybrid' }
      globalThis.fetch = mockFetch(result)

      const data = await setRoutingMode()

      expect(globalThis.fetch).toHaveBeenCalledWith('/routing-mode', { method: 'POST' })
      expect(data).toEqual(result)
    })
  })
})
