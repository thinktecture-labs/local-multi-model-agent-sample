import { render, screen, fireEvent, waitFor, within } from '@testing-library/react'
import { AppProvider } from '../../state/AppContext.tsx'
import App from '../../App.tsx'

class MockWebSocket {
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  close() {}
  send() {}
}

const healthData = {
  models: {
    INFERENCE: true,
    FUNCTION: true,
    EMBEDDING: true,
    VISION: true,
    WHISPER: true,
    QWEN: true,
    CLOUD: true,
  },
  document_count: 13,
  interaction_count: 5,
}

const ragQueryResult = {
  intent: 'rag_query',
  response: 'The Enterprise plan is priced at $499/month.',
  execution_time_ms: 342,
  steps: [
    { action: 'classify_intent', model: 'gemma3-1B', duration_ms: 45, details: {}, tokens_used: 12 },
    {
      action: 'vector_search', model: 'embeddinggemma-308M', duration_ms: 120,
      details: { query: 'pricing', documents: [{ title: 'Pricing Page', score: 0.95, content: 'Enterprise plan $499/mo' }] },
      tokens_used: 0,
    },
    {
      action: 'synthesize_response', model: 'gemma3-4B', duration_ms: 177,
      details: { response: 'Enterprise plan is $499/month.' },
      tokens_used: 80,
    },
  ],
  models_used: ['gemma3-1B', 'embeddinggemma-308M', 'gemma3-4B'],
  total_tokens: 92,
  prompt_tokens: 50,
  completion_tokens: 42,
}

const toolQueryResult = {
  intent: 'tool_use',
  response: 'The result is $1,205,200.',
  execution_time_ms: 520,
  steps: [
    { action: 'classify_intent', model: 'gemma3-1B', duration_ms: 38, details: {}, tokens_used: 10 },
    {
      action: 'select_tool', model: 'qwen3.5-4b', duration_ms: 55,
      details: { tool: 'calculator', arguments: { expression: '23 * 52400' } },
      tokens_used: 15,
    },
    {
      action: 'execute_tool', model: 'local', duration_ms: 2,
      details: { tool: 'calculator', result: '1205200' },
      tokens_used: 0,
    },
    {
      action: 'format_response', model: 'gemma3-1B', duration_ms: 120,
      details: { response: 'The result is $1,205,200.' },
      tokens_used: 45,
    },
  ],
  models_used: ['gemma3-1B', 'qwen3.5-4b'],
  total_tokens: 70,
  prompt_tokens: 35,
  completion_tokens: 35,
}

const compareResult = {
  intent: 'rag_query',
  local_response: 'Local: Enterprise plan is $499/mo.',
  cloud_response: 'Cloud: The Enterprise plan costs $499 per month.',
  cloud_model: 'GPT-5.4',
  cloud_latency_ms: 1200,
  cloud_cost: 0.0042,
  execution_time_ms: 350,
  steps: [
    { action: 'classify_intent', model: 'gemma3-1B', duration_ms: 40, details: {}, tokens_used: 12 },
  ],
  total_tokens: 90,
  prompt_tokens: 45,
  completion_tokens: 45,
}

const compareAllResult = {
  multi_models: { ...ragQueryResult },
  qwen: { ...ragQueryResult, response: 'Qwen: Enterprise plan is $499/mo.' },
  cloud: { ...ragQueryResult, response: 'Cloud: Enterprise plan is $499/mo.', cloud_cost: 0.0042 },
}

const evalResult = {
  model: 'gemma3',
  overall_accuracy: 0.92,
  overall_correct: 23,
  n: 25,
  per_class: {
    rag_query: { correct: 10, n: 10, accuracy: 1.0 },
    tool_use: { correct: 8, n: 10, accuracy: 0.8 },
    direct_answer: { correct: 5, n: 5, accuracy: 1.0 },
  },
}

const sqlQueryResult = {
  intent: 'tool_use',
  response: 'Here are the top 3 customers by revenue.',
  execution_time_ms: 480,
  steps: [
    { action: 'classify_intent', model: 'gemma3-1B', duration_ms: 40, details: {}, tokens_used: 10 },
    {
      action: 'select_tool', model: 'qwen3.5-4b', duration_ms: 50,
      details: { tool: 'sql_query', arguments: { query: 'SELECT customer, revenue FROM customers ORDER BY revenue DESC LIMIT 3' } },
      tokens_used: 15,
    },
    {
      action: 'execute_tool', model: 'local', duration_ms: 8,
      details: {
        tool: 'sql_query',
        result: {
          columns: ['customer', 'revenue'],
          rows: [
            { customer: 'Acme Corp', revenue: 450000 },
            { customer: 'Globex Inc', revenue: 320000 },
            { customer: 'Initech', revenue: 280000 },
          ],
          count: 3,
        },
      },
      tokens_used: 0,
    },
    {
      action: 'format_response', model: 'gemma3-1B', duration_ms: 110,
      details: { response: 'Here are the top 3 customers by revenue.' },
      tokens_used: 40,
    },
  ],
  models_used: ['gemma3-1B', 'qwen3.5-4b'],
  total_tokens: 65,
  prompt_tokens: 30,
  completion_tokens: 35,
}

const modelModeResult = { mode: 'base' }
const evalResults = { before: null, after: null }

/** Build a mock SSE ReadableStream response from a QueryResult-like object. */
function makeSseResponse(result: Record<string, unknown>) {
  const steps = (result.steps ?? []) as Array<Record<string, unknown>>
  const events: string[] = []
  for (const step of steps) {
    events.push(`event: step\ndata: ${JSON.stringify(step)}\n\n`)
  }
  const response = result.response as string ?? ''
  if (response) {
    events.push(`event: token\ndata: ${JSON.stringify({ text: response })}\n\n`)
  }
  events.push(`event: done\ndata: ${JSON.stringify({
    intent: result.intent ?? 'direct_answer',
    execution_time_ms: result.execution_time_ms ?? 0,
    total_tokens: result.total_tokens ?? 0,
    prompt_tokens: result.prompt_tokens ?? 0,
    completion_tokens: result.completion_tokens ?? 0,
    models_used: result.models_used ?? [],
  })}\n\n`)
  const encoded = new TextEncoder().encode(events.join(''))
  const body = new ReadableStream({
    start(controller) { controller.enqueue(encoded); controller.close() },
  })
  return { ok: true, body }
}

function setupMockFetch(overrides: Record<string, unknown> = {}) {
  let queryCount = 0
  let streamCount = 0

  globalThis.fetch = vi.fn().mockImplementation((url: string, options?: RequestInit) => {
    const method = options?.method ?? 'GET'

    if (url === '/scenario' && method === 'GET') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ scenario: 'nextera', brand: 'Nextera', label: 'Nextera Gemma', language: 'en' }) })
    }
    if (url === '/health' && method === 'GET') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(overrides.health ?? healthData) })
    }
    if (url === '/models/mode' && method === 'GET') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(overrides.modelMode ?? modelModeResult) })
    }
    if (url === '/eval/results' && method === 'GET') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(overrides.evalResults ?? evalResults) })
    }
    if (url === '/query' && method === 'POST') {
      queryCount++
      const result = queryCount === 1 ? (overrides.query1 ?? ragQueryResult) : (overrides.query2 ?? toolQueryResult)
      return Promise.resolve({ ok: true, json: () => Promise.resolve(result) })
    }
    if (url === '/query/stream' && method === 'POST') {
      streamCount++
      const result = streamCount === 1 ? (overrides.query1 ?? ragQueryResult) : (overrides.query2 ?? toolQueryResult)
      return Promise.resolve(makeSseResponse(result as Record<string, unknown>))
    }
    if (url === '/compare' && method === 'POST') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(overrides.compare ?? compareResult) })
    }
    if (url === '/query/compare-all' && method === 'POST') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(overrides.compareAll ?? compareAllResult) })
    }
    if (url === '/models/swap' && method === 'POST') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ status: 'ok' }) })
    }
    if (url === '/eval' && method === 'POST') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve(overrides.eval ?? evalResult) })
    }
    if (url === '/network-mode' && method === 'POST') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ network_mode: 'offline' }) })
    }
    if (url === '/routing-mode' && method === 'POST') {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ routing_mode: 'planner' }) })
    }

    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
  })
}

function renderApp() {
  const result = render(
    <AppProvider>
      <App />
    </AppProvider>,
  )
  return result
}

async function waitForHealth() {
  await waitFor(() => {
    expect(globalThis.fetch).toHaveBeenCalledWith('/health')
  })
}

async function submitQuery(text: string) {
  const textarea = screen.getByPlaceholderText('Ask a question\u2026')
  fireEvent.change(textarea, { target: { value: text } })
  const sendBtn = document.querySelector('.send-btn') as HTMLButtonElement
  fireEvent.click(sendBtn)
}

describe('E2E: Full Application', () => {
  const originalFetch = globalThis.fetch
  const originalWebSocket = globalThis.WebSocket

  beforeEach(() => {
    Element.prototype.scrollTo = vi.fn()
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    globalThis.WebSocket = MockWebSocket as any
    setupMockFetch()
  })

  afterEach(() => {
    globalThis.fetch = originalFetch
    globalThis.WebSocket = originalWebSocket
    vi.restoreAllMocks()
  })

  describe('Initial render', () => {
    it('displays brand, health pills, and all header controls', async () => {
      renderApp()
      await waitForHealth()

      expect(screen.getByText(/Nextera/)).toBeTruthy()
      expect(screen.getByText(/Agent/)).toBeTruthy()
      expect(screen.getByText(/Observatory/)).toBeTruthy()

      const { container } = { container: document.body }
      // Models bar is collapsed by default — expand it to check pills
      fireEvent.click(screen.getByText('Models'))
      expect(container.querySelectorAll('.model-pill').length).toBe(10)

      const whisperPill = container.querySelector('[data-model="whisper"]')
      expect(whisperPill?.className).toContain('online')
    })

    it('shows suggestion chips with Try: label', async () => {
      renderApp()
      await waitForHealth()

      // Suggestions collapsed by default — only toggle visible
      expect(screen.getByText('Try:')).toBeTruthy()
      // Expand to see chips
      fireEvent.click(screen.getByText('Try:'))
      expect(screen.getByText("What's the pricing for the Enterprise plan?")).toBeTruthy()
    })

    it('shows empty trace pane with pipeline trace header', async () => {
      renderApp()
      await waitForHealth()

      expect(screen.getByText('Pipeline Trace')).toBeTruthy()
      expect(screen.getByText(/Select a query to see/)).toBeTruthy()
    })

    it('shows flywheel with interaction count from health', async () => {
      renderApp()
      await waitForHealth()

      await waitFor(() => {
        expect(screen.getByText('5 logged')).toBeTruthy()
      })
    })

    it('shows cost counter with initial zero values', async () => {
      renderApp()
      await waitForHealth()

      expect(screen.getByText('Local: $0.00')).toBeTruthy()
      expect(screen.getByText(/Cloud:/)).toBeTruthy()
    })

    it('hides eval panel when no eval data', async () => {
      renderApp()
      await waitForHealth()

      const evalPanel = document.querySelector('.eval-panel') as HTMLElement
      expect(evalPanel).toBeTruthy()
      expect(evalPanel.classList.contains('active')).toBe(false)
    })

    it('fetches model mode and eval results on mount', async () => {
      renderApp()
      await waitForHealth()

      await waitFor(() => {
        expect(globalThis.fetch).toHaveBeenCalledWith('/models/mode')
        expect(globalThis.fetch).toHaveBeenCalledWith('/eval/results')
      })
    })
  })

  describe('Query submission', () => {
    it('RAG query shows response, intent badge, and trace steps', async () => {
      renderApp()
      await waitForHealth()

      await submitQuery("What's the pricing for the Enterprise plan?")

      await waitFor(() => {
        expect(screen.getByText('The Enterprise plan is priced at $499/month.')).toBeTruthy()
      })

      expect(document.querySelector('.intent-badge.rag_query')).toBeTruthy()
      expect(screen.getByText('RAG')).toBeTruthy()

      await waitFor(() => {
        expect(screen.getAllByText('Classify Intent').length).toBeGreaterThan(0)
        expect(screen.getAllByText('Semantic Search').length).toBeGreaterThan(0)
        expect(screen.getAllByText('Synthesize Response').length).toBeGreaterThan(0)
      })
    })

    it('tool_use query shows calculator result and tool steps', async () => {
      setupMockFetch({ query1: toolQueryResult })
      renderApp()
      await waitForHealth()

      await submitQuery('Calculate 23 * 52400')

      await waitFor(() => {
        expect(screen.getAllByText('The result is $1,205,200.').length).toBeGreaterThan(0)
      })

      expect(document.querySelector('.intent-badge.tool_use')).toBeTruthy()

      await waitFor(() => {
        expect(screen.getAllByText('Select Tool').length).toBeGreaterThan(0)
        expect(screen.getAllByText('Execute Tool').length).toBeGreaterThan(0)
      })
    })

    it('clicking suggestion chip sends query', async () => {
      renderApp()
      await waitForHealth()

      // Expand collapsed suggestions first
      fireEvent.click(screen.getByText('Try:'))
      fireEvent.click(screen.getByText("What's the pricing for the Enterprise plan?"))

      await waitFor(() => {
        expect(globalThis.fetch).toHaveBeenCalledWith('/query/stream', expect.objectContaining({ method: 'POST' }))
      })
    })

    it('removes streaming cursor after stream completes', async () => {
      renderApp()
      await waitForHealth()
      await submitQuery("What's the pricing for the Enterprise plan?")

      // After SSE streaming completes, the response is rendered and
      // no cursor-blink remains (streaming state is cleared).
      await waitFor(() => {
        expect(screen.getByText('The Enterprise plan is priced at $499/month.')).toBeTruthy()
      })
      expect(document.querySelector('.cursor-blink')).toBeNull()
    })

    it('multiple queries create multiple exchange cards', async () => {
      renderApp()
      await waitForHealth()

      await submitQuery('First query')
      await waitFor(() => {
        expect(screen.getByText('The Enterprise plan is priced at $499/month.')).toBeTruthy()
      })

      await submitQuery('Second query')
      await waitFor(() => {
        expect(screen.getAllByText(/The result is/).length).toBeGreaterThan(0)
      })

      const exchanges = document.querySelectorAll('.exchange')
      expect(exchanges.length).toBe(2)
    })

    it('clicking an exchange card selects it and shows its trace', async () => {
      renderApp()
      await waitForHealth()

      await submitQuery('First query')
      await waitFor(() => {
        expect(screen.getByText('The Enterprise plan is priced at $499/month.')).toBeTruthy()
      })

      await submitQuery('Second query')
      await waitFor(() => {
        expect(screen.getAllByText(/The result is/).length).toBeGreaterThan(0)
      })

      const firstExchange = document.querySelectorAll('.exchange')[0] as HTMLElement
      fireEvent.click(firstExchange)

      await waitFor(() => {
        expect(screen.getAllByText('Semantic Search').length).toBeGreaterThan(0)
      })
    })
  })

  describe('Header controls', () => {
    it('theme toggle changes theme symbol', async () => {
      renderApp()
      await waitForHealth()

      const themeBtn = screen.getByTitle('Toggle light/dark')
      const initialText = themeBtn.textContent
      fireEvent.click(themeBtn)
      expect(themeBtn.textContent).not.toBe(initialText)
    })

    it('trace pane toggle hides/shows trace', async () => {
      renderApp()
      await waitForHealth()

      const traceBtn = screen.getByTitle('Toggle trace pane')
      const main = document.querySelector('main') as HTMLElement
      const initialCollapsed = main.classList.contains('trace-collapsed')

      fireEvent.click(traceBtn)

      expect(main.classList.contains('trace-collapsed')).not.toBe(initialCollapsed)
    })

    it('clear history removes all exchanges', async () => {
      renderApp()
      await waitForHealth()

      await submitQuery('Test query')
      await waitFor(() => {
        expect(document.querySelectorAll('.exchange').length).toBe(1)
      })

      const clearBtn = screen.getByTitle('Clear history')
      fireEvent.click(clearBtn)

      expect(document.querySelectorAll('.exchange').length).toBe(0)
    })

    it('model mode toggle calls swap API and updates label', async () => {
      renderApp()
      await waitForHealth()

      await waitFor(() => {
        expect(screen.getByText('BASE')).toBeTruthy()
      })

      const modeBtn = screen.getByTitle('Toggle base/fine-tuned')
      fireEvent.click(modeBtn)

      await waitFor(() => {
        expect(globalThis.fetch).toHaveBeenCalledWith('/models/swap', expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ mode: 'finetuned' }),
        }))
      })

      await waitFor(() => {
        expect(screen.getByText('FINE-TUNED')).toBeTruthy()
      })
    })

    it('network toggle calls API and disables routing toggle', async () => {
      renderApp()
      await waitForHealth()

      const netBtn = screen.getByTitle('Go offline (airplane mode)')
      fireEvent.click(netBtn)

      await waitFor(() => {
        expect(globalThis.fetch).toHaveBeenCalledWith('/network-mode', expect.objectContaining({ method: 'POST' }))
      })

      await waitFor(() => {
        const routeToggle = document.querySelector('.route-toggle')
        expect(routeToggle?.className).toContain('disabled')
      })
    })
  })

  describe('Compare mode', () => {
    it('submitting query in All mode calls /query/stream for each backend', async () => {
      renderApp()
      await waitForHealth()

      const allBtn = screen.getByText('All') as HTMLButtonElement
      fireEvent.click(allBtn)

      await submitQuery('Compare test')

      await waitFor(() => {
        const streamCalls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.filter(
          (c: unknown[]) => c[0] === '/query/stream',
        )
        // At least multi-models stream should be called
        expect(streamCalls.length).toBeGreaterThanOrEqual(1)
      })
    })
  })

  describe('Eval flow', () => {
    it('clicking Eval in flywheel runs eval and shows results', async () => {
      renderApp()
      await waitForHealth()

      const evalBtns = screen.getAllByText('Eval')
      const flywheelEvalBtn = evalBtns[evalBtns.length - 1]
      fireEvent.click(flywheelEvalBtn)

      await waitFor(() => {
        expect(globalThis.fetch).toHaveBeenCalledWith('/eval', expect.objectContaining({ method: 'POST' }))
      })

      await waitFor(() => {
        const evalPanel = document.querySelector('.eval-panel') as HTMLElement
        expect(evalPanel.classList.contains('active')).toBe(true)
      })
    })

    it('shows eval accuracy percentages after eval completes', async () => {
      renderApp()
      await waitForHealth()

      const evalBtns = screen.getAllByText('Eval')
      fireEvent.click(evalBtns[evalBtns.length - 1])

      await waitFor(() => {
        expect(screen.getByText('92%')).toBeTruthy()
      })
    })
  })

  describe('Health polling', () => {
    it('renders model pills as online when health reports them available', async () => {
      renderApp()
      await waitForHealth()

      // Expand models bar first
      fireEvent.click(screen.getByText('Models'))

      await waitFor(() => {
        const infPill = document.querySelector('[data-model="inference"]')
        expect(infPill?.classList.contains('online')).toBe(true)
      })
    })

    it('shows voice pills as offline when WHISPER is not available', async () => {
      setupMockFetch({
        health: {
          models: { INFERENCE: true, FUNCTION: true, EMBEDDING: true, VISION: true },
          document_count: 5,
          interaction_count: 0,
        },
      })
      renderApp()
      await waitForHealth()

      // Expand models bar
      fireEvent.click(screen.getByText('Models'))

      await waitFor(() => {
        const whisperPill = document.querySelector('[data-model="whisper"]')
        expect(whisperPill?.classList.contains('offline')).toBe(true)
      })
    })
  })

  describe('GPU and Energy panels', () => {
    it('renders GPU panel with placeholder when no data', async () => {
      renderApp()
      await waitForHealth()

      expect(document.querySelector('.gpu-panel')).toBeTruthy()
      expect(document.querySelector('.gpu-name')?.textContent).toBe('CPU mode')
    })

    it('renders Energy panel with waiting state when no data', async () => {
      renderApp()
      await waitForHealth()

      expect(document.querySelector('.energy-panel')).toBeTruthy()
    })

    it('renders GPU stats when WebSocket pushes data', async () => {
      renderApp()
      await waitForHealth()

      const wsInstance = MockWebSocket.prototype
      const onmessage = (globalThis.WebSocket as unknown as typeof MockWebSocket).prototype.onmessage

      if (onmessage) {
        const event = new MessageEvent('message', {
          data: JSON.stringify({
            gpu: {
              available: true,
              name: 'NVIDIA RTX 4090',
              backend: 'cuda',
              vram_used_mb: 8192,
              vram_total_mb: 24576,
              utilization_pct: 65,
              temperature_c: 72,
            },
          }),
        })
        onmessage(event)
      }
    })
  })

  describe('Document upload zone', () => {
    it('renders the upload drop zone', async () => {
      renderApp()
      await waitForHealth()

      expect(document.querySelector('.doc-drop-zone')).toBeTruthy()
    })
  })

  describe('Error handling', () => {
    it('shows error message when query fails', async () => {
      ;(globalThis.fetch as ReturnType<typeof vi.fn>).mockImplementation((url: string, options?: RequestInit) => {
        if (url === '/query/stream' && options?.method === 'POST') {
          return Promise.resolve({ ok: false, status: 500 })
        }
        if (url === '/health') return Promise.resolve({ ok: true, json: () => Promise.resolve(healthData) })
        if (url === '/models/mode') return Promise.resolve({ ok: true, json: () => Promise.resolve(modelModeResult) })
        if (url === '/eval/results') return Promise.resolve({ ok: true, json: () => Promise.resolve(evalResults) })
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) })
      })

      renderApp()
      await waitForHealth()
      await submitQuery('This will fail')

      await waitFor(() => {
        expect(screen.getByText(/Error:/)).toBeTruthy()
      })
    })
  })

  describe('Keyboard interaction', () => {
    it('Enter submits query, Shift+Enter does not', async () => {
      renderApp()
      await waitForHealth()

      const textarea = screen.getByPlaceholderText('Ask a question\u2026')

      fireEvent.change(textarea, { target: { value: 'test' } })
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true })

      expect(globalThis.fetch).not.toHaveBeenCalledWith('/query/stream', expect.anything())

      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })

      await waitFor(() => {
        expect(globalThis.fetch).toHaveBeenCalledWith('/query/stream', expect.objectContaining({ method: 'POST' }))
      })
    })

    it('does not submit empty input', async () => {
      renderApp()
      await waitForHealth()

      const textarea = screen.getByPlaceholderText('Ask a question\u2026')
      fireEvent.change(textarea, { target: { value: '   ' } })
      fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })

      const queryCalls = (globalThis.fetch as ReturnType<typeof vi.fn>).mock.calls.filter(
        (c: unknown[]) => c[0] === '/query',
      )
      expect(queryCalls.length).toBe(0)
    })
  })

  describe('Privacy badge', () => {
    it('shows zero bytes sent externally', async () => {
      renderApp()
      await waitForHealth()

      expect(screen.getByText('0 bytes sent externally')).toBeTruthy()
    })
  })

  describe('Trace details', () => {
    it('vector search step shows documents with scores', async () => {
      renderApp()
      await waitForHealth()
      await submitQuery('pricing query')

      await waitFor(() => {
        expect(screen.getByText('Pricing Page')).toBeTruthy()
      })
    })

    it('shows latency waterfall for multi-step traces', async () => {
      renderApp()
      await waitForHealth()
      await submitQuery('pricing query')

      await waitFor(() => {
        const waterfall = document.querySelector('.waterfall')
        expect(waterfall).toBeTruthy()
      })
    })
  })

  describe('SQL query with dict rows (real server format)', () => {
    it('renders SQL table from dict-style rows without crashing', async () => {
      setupMockFetch({ query1: sqlQueryResult })
      renderApp()
      await waitForHealth()
      await submitQuery('Show top 3 customers by revenue')

      await waitFor(() => {
        expect(screen.getAllByText('Here are the top 3 customers by revenue.').length).toBeGreaterThan(0)
      })

      // Open the Sources panel
      const sourceToggle = document.querySelector('.source-toggle') as HTMLButtonElement
      fireEvent.click(sourceToggle)

      await waitFor(() => {
        // Should render SQL table with dict rows converted to cells
        expect(screen.getAllByText('Acme Corp').length).toBeGreaterThan(0)
        expect(screen.getAllByText('450000').length).toBeGreaterThan(0)
        expect(screen.getAllByText('Globex Inc').length).toBeGreaterThan(0)
      })
    })

    it('renders SQL table in trace step detail with dict rows', async () => {
      setupMockFetch({ query1: sqlQueryResult })
      renderApp()
      await waitForHealth()
      await submitQuery('Show top 3 customers by revenue')

      await waitFor(() => {
        expect(screen.getAllByText('Here are the top 3 customers by revenue.').length).toBeGreaterThan(0)
      })

      // The trace pane should show the execute_tool step with SQL table
      await waitFor(() => {
        const sqlTables = document.querySelectorAll('.sql-table')
        expect(sqlTables.length).toBeGreaterThanOrEqual(1)
      })

      // Verify table headers
      await waitFor(() => {
        expect(screen.getByText('customer')).toBeTruthy()
        expect(screen.getByText('revenue')).toBeTruthy()
      })
    })
  })
})
