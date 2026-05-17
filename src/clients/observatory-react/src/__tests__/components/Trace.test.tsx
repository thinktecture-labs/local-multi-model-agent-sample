import { render, screen } from '@testing-library/react'
import { AppProvider } from '../../state/AppContext.tsx'
import { StepCard } from '../../components/Trace/StepCard.tsx'
import { StepDetail } from '../../components/Trace/StepDetail.tsx'
import { TracePanel } from '../../components/Trace/TracePanel.tsx'
import type { ExecutionStep } from '../../types/api.ts'

function renderWithProvider(ui: React.ReactNode) {
  return render(<AppProvider>{ui}</AppProvider>)
}

function makeStep(overrides: Partial<ExecutionStep> = {}): ExecutionStep {
  return {
    action: 'classify_intent',
    model: 'gemma3-1B',
    duration_ms: 50,
    details: {},
    tokens_used: 10,
    ...overrides,
  }
}

describe('StepCard', () => {
  it('renders step number', () => {
    const step = makeStep()
    const { container } = render(<StepCard step={step} index={0} totalMs={100} />)
    expect(container.querySelector('.step-num')?.textContent).toBe('1')
  })

  it('renders action label', () => {
    const step = makeStep({ action: 'vector_search' })
    render(<StepCard step={step} index={0} totalMs={100} />)
    expect(screen.getByText('Semantic Search')).toBeTruthy()
  })

  it('renders model tag', () => {
    const step = makeStep({ model: 'embeddinggemma' })
    const { container } = render(<StepCard step={step} index={0} totalMs={100} />)
    const modelTag = container.querySelector('.step-model-tag')
    expect(modelTag?.textContent).toBe('embeddinggemma')
  })

  it('renders duration', () => {
    const step = makeStep({ duration_ms: 250 })
    render(<StepCard step={step} index={0} totalMs={500} />)
    expect(screen.getByText('250 ms')).toBeTruthy()
  })
})

describe('StepDetail', () => {
  it('returns null for classify_intent', () => {
    const step = makeStep({ action: 'classify_intent' })
    const { container } = render(<StepDetail step={step} />)
    expect(container.innerHTML).toBe('')
  })

  it('renders doc list for vector_search', () => {
    const step = makeStep({
      action: 'vector_search',
      details: {
        query: 'test',
        documents: [
          { title: 'Doc A', score: 0.95, content: 'Some content' },
          { title: 'Doc B', score: 0.8, content: 'Other content' },
        ],
      },
    })
    render(<StepDetail step={step} />)
    expect(screen.getByText('Doc A')).toBeTruthy()
    expect(screen.getByText('Doc B')).toBeTruthy()
  })

  it('renders calculator result for execute_tool with calculator', () => {
    const step = makeStep({
      action: 'execute_tool',
      details: {
        tool: 'calculator',
        result: '37.4985',
      },
    })
    const { container } = render(<StepDetail step={step} />)
    const calcResult = container.querySelector('.calc-result')
    expect(calcResult?.textContent).toContain('37.4985')
  })

  it('renders SQL table for execute_tool with sql', () => {
    const step = makeStep({
      action: 'execute_tool',
      details: {
        tool: 'sql_query',
        result: {
          columns: ['name', 'count'],
          rows: [['Alice', 5], ['Bob', 3]],
        },
      },
    })
    render(<StepDetail step={step} />)
    expect(screen.getByText('name')).toBeTruthy()
    expect(screen.getByText('count')).toBeTruthy()
    expect(screen.getByText('Alice')).toBeTruthy()
    expect(screen.getByText('Bob')).toBeTruthy()
  })
})

describe('TracePanel', () => {
  it('shows empty state when no selection', () => {
    renderWithProvider(<TracePanel />)
    expect(screen.getByText(/Select a query to see/)).toBeTruthy()
  })
})
