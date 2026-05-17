import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { AppProvider } from '../../state/AppContext.tsx'
import App from '../../App.tsx'

// Mock WebSocket to prevent connection attempts
class MockWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  close() {}
}

const healthData = {
  models: {
    INFERENCE: true,
    FUNCTION: true,
    EMBEDDING: true,
    VISION: true,
  },
  document_count: 10,
  interaction_count: 5,
}

const queryResult = {
  intent: 'rag_query',
  response: 'We have a variety of products including electronics and software.',
  execution_time_ms: 342,
  steps: [
    {
      action: 'classify_intent',
      model: 'gemma3-1B',
      duration_ms: 45,
      details: {},
      tokens_used: 12,
    },
    {
      action: 'vector_search',
      model: 'embeddinggemma-308M',
      duration_ms: 120,
      details: {
        query: 'What products do we have?',
        documents: [{ title: 'Products Overview', score: 0.92, content: 'Electronics and software' }],
      },
      tokens_used: 0,
    },
    {
      action: 'synthesize_response',
      model: 'gemma3-4B',
      duration_ms: 177,
      details: { response: 'We have electronics and software.' },
      tokens_used: 80,
    },
  ],
  models_used: ['gemma3-1B', 'embeddinggemma-308M', 'gemma3-4B'],
  total_tokens: 92,
  prompt_tokens: 50,
  completion_tokens: 42,
}

const modelModeResult = { mode: 'base' }
const evalResults = { before: null, after: null }

describe('Integration: query flow', () => {
  const originalFetch = globalThis.fetch
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    // Mock scrollTo since jsdom does not implement it
    Element.prototype.scrollTo = vi.fn()

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    globalThis.WebSocket = MockWebSocket as any

    globalThis.fetch = vi.fn().mockImplementation((url: string, options?: RequestInit) => {
      const method = options?.method ?? 'GET'

      if (url === '/health' && method === 'GET') {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(healthData),
        })
      }

      if (url === '/models/mode' && method === 'GET') {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(modelModeResult),
        })
      }

      if (url === '/eval/results' && method === 'GET') {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(evalResults),
        })
      }

      if (url === '/query' && method === 'POST') {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: () => Promise.resolve(queryResult),
        })
      }

      if (url === '/query/stream' && method === 'POST') {
        const steps = queryResult.steps as Array<Record<string, unknown>>
        const events: string[] = []
        for (const step of steps) {
          events.push(`event: step\ndata: ${JSON.stringify(step)}\n\n`)
        }
        events.push(`event: token\ndata: ${JSON.stringify({ text: queryResult.response })}\n\n`)
        events.push(`event: done\ndata: ${JSON.stringify({
          intent: queryResult.intent,
          execution_time_ms: queryResult.execution_time_ms,
          total_tokens: queryResult.total_tokens,
          prompt_tokens: queryResult.prompt_tokens,
          completion_tokens: queryResult.completion_tokens,
          models_used: queryResult.models_used,
        })}\n\n`)
        const encoded = new TextEncoder().encode(events.join(''))
        const body = new ReadableStream({
          start(controller) { controller.enqueue(encoded); controller.close() },
        })
        return Promise.resolve({ ok: true, body })
      }

      // Fallback
      return Promise.resolve({
        ok: true,
        status: 200,
        json: () => Promise.resolve({}),
      })
    })
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
    globalThis.WebSocket = originalWebSocket
    vi.restoreAllMocks()
  })

  it('submitting a query shows the exchange with response text and trace steps', async () => {
    render(
      <AppProvider>
        <App />
      </AppProvider>,
    )

    // Wait for initial health poll to complete
    await waitFor(() => {
      expect(globalThis.fetch).toHaveBeenCalledWith('/health')
    })

    // Type a query
    const textarea = screen.getByPlaceholderText('Ask a question\u2026')
    fireEvent.change(textarea, { target: { value: 'What products do we have?' } })

    // Click send
    const sendButton = document.querySelector('.send-btn') as HTMLButtonElement
    fireEvent.click(sendButton)

    // Wait for the response to appear
    await waitFor(() => {
      expect(
        screen.getByText('We have a variety of products including electronics and software.'),
      ).toBeTruthy()
    })

    // Verify the exchange query is shown (may appear multiple times: suggestion chip + exchange)
    expect(screen.getAllByText('What products do we have?').length).toBeGreaterThan(0)

    // Verify trace steps appear (labels appear in both StepCard and LatencyWaterfall, so use getAllByText)
    await waitFor(() => {
      expect(screen.getAllByText('Classify Intent').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Semantic Search').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Synthesize Response').length).toBeGreaterThan(0)
    })
  })
})
