import { render, screen, fireEvent } from '@testing-library/react'
import { AppProvider } from '../../state/AppContext.tsx'
import { IntentBadge } from '../../components/Conversation/IntentBadge.tsx'
import { ExchangeCard } from '../../components/Conversation/ExchangeCard.tsx'
import { SuggestionChips } from '../../components/Input/SuggestionChips.tsx'
import { QueryInput } from '../../components/Input/QueryInput.tsx'
import { SUGGESTIONS } from '../../utils/format.ts'
import type { Exchange } from '../../types/state.ts'
import type { Intent } from '../../types/api.ts'

function renderWithProvider(ui: React.ReactNode) {
  return render(<AppProvider>{ui}</AppProvider>)
}

describe('IntentBadge', () => {
  const intents: Array<{ intent: Intent; badge: string; className: string }> = [
    { intent: 'rag_query', badge: 'RAG', className: 'rag_query' },
    { intent: 'tool_use', badge: 'TOOL', className: 'tool_use' },
    { intent: 'direct_answer', badge: 'DIRECT', className: 'direct_answer' },
    { intent: 'image_query', badge: 'IMAGE', className: 'image_query' },
    { intent: 'voice', badge: 'VOICE', className: 'voice' },
  ]

  it.each(intents)('renders correct class and label for $intent', ({ intent, badge, className }) => {
    const { container } = render(<IntentBadge intent={intent} />)
    const span = container.querySelector('.intent-badge')
    expect(span?.textContent).toBe(badge)
    expect(span?.className).toContain(className)
  })
})

describe('ExchangeCard', () => {
  function makeExchange(overrides: Partial<Exchange> = {}): Exchange {
    return {
      query: 'What is our refund policy?',
      images: [],
      imageDataUrls: [],
      result: null,
      ...overrides,
    }
  }

  it('renders the query text', () => {
    const exchange = makeExchange()
    renderWithProvider(
      <ExchangeCard exchange={exchange} idx={0} isActive={false} onClick={() => {}} />,
    )
    expect(screen.getByText('What is our refund policy?')).toBeTruthy()
  })

  it('renders intent badge when result exists', () => {
    const exchange = makeExchange({
      result: {
        intent: 'rag_query',
        response: 'Refund within 30 days',
        execution_time_ms: 150,
        steps: [],
        models_used: ['gemma3-1B'],
        total_tokens: 50,
      },
    })
    renderWithProvider(
      <ExchangeCard exchange={exchange} idx={0} isActive={false} onClick={() => {}} />,
    )
    expect(screen.getByText('RAG')).toBeTruthy()
  })

  it('renders response text when result exists', () => {
    const exchange = makeExchange({
      result: {
        intent: 'direct_answer',
        response: 'The answer is 42',
        execution_time_ms: 100,
        steps: [],
        models_used: [],
        total_tokens: 20,
      },
    })
    renderWithProvider(
      <ExchangeCard exchange={exchange} idx={0} isActive={false} onClick={() => {}} />,
    )
    expect(screen.getByText('The answer is 42')).toBeTruthy()
  })

  it('renders typing dots when loading (no result)', () => {
    const exchange = makeExchange({ result: null })
    const { container } = renderWithProvider(
      <ExchangeCard exchange={exchange} idx={0} isActive={false} onClick={() => {}} />,
    )
    expect(container.querySelector('.typing-dots')).toBeTruthy()
  })

  it('applies active class when isActive is true', () => {
    const exchange = makeExchange()
    const { container } = renderWithProvider(
      <ExchangeCard exchange={exchange} idx={0} isActive={true} onClick={() => {}} />,
    )
    expect(container.querySelector('.exchange.active')).toBeTruthy()
  })
})

describe('SuggestionChips', () => {
  it('renders collapsed by default with Try: toggle', () => {
    const onSend = vi.fn()
    render(<SuggestionChips onSend={onSend} />)
    expect(screen.getByText('Try:')).toBeTruthy()
    // Chips not visible when collapsed
    expect(screen.queryByText(SUGGESTIONS[0])).toBeNull()
  })

  it('renders all suggestions when expanded', () => {
    const onSend = vi.fn()
    render(<SuggestionChips onSend={onSend} />)
    fireEvent.click(screen.getByText('Try:'))
    for (const text of SUGGESTIONS) {
      expect(screen.getByText(text)).toBeTruthy()
    }
  })

  it('clicking a chip calls onSend with its text', () => {
    const onSend = vi.fn()
    render(<SuggestionChips onSend={onSend} />)
    fireEvent.click(screen.getByText('Try:'))
    fireEvent.click(screen.getByText(SUGGESTIONS[0]))
    expect(onSend).toHaveBeenCalledWith(SUGGESTIONS[0])
  })
})

describe('QueryInput', () => {
  it('typing and pressing Enter calls onSend', () => {
    const onSend = vi.fn()
    renderWithProvider(
      <QueryInput onSend={onSend} onImageFile={() => {}} voiceAvailable={false} />,
    )
    const textarea = screen.getByPlaceholderText('Ask a question\u2026')

    fireEvent.change(textarea, { target: { value: 'hello world' } })
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })

    expect(onSend).toHaveBeenCalledWith('hello world')
  })

  it('Shift+Enter does not call onSend', () => {
    const onSend = vi.fn()
    renderWithProvider(
      <QueryInput onSend={onSend} onImageFile={() => {}} voiceAvailable={false} />,
    )
    const textarea = screen.getByPlaceholderText('Ask a question\u2026')

    fireEvent.change(textarea, { target: { value: 'hello world' } })
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: true })

    // onSend should not be called directly from keyDown with shift
    // (it might be called from suggestion chips though, so filter)
    const keyDownCalls = onSend.mock.calls.filter(
      (call: unknown[]) => call[0] === 'hello world',
    )
    expect(keyDownCalls).toHaveLength(0)
  })

  it('does not send empty input', () => {
    const onSend = vi.fn()
    renderWithProvider(
      <QueryInput onSend={onSend} onImageFile={() => {}} voiceAvailable={false} />,
    )
    const textarea = screen.getByPlaceholderText('Ask a question\u2026')

    fireEvent.change(textarea, { target: { value: '   ' } })
    fireEvent.keyDown(textarea, { key: 'Enter', shiftKey: false })

    expect(onSend).not.toHaveBeenCalled()
  })
})
