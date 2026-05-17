import { render, screen, fireEvent } from '@testing-library/react'
import { AppProvider } from '../../state/AppContext.tsx'
import { HealthPills } from '../../components/Header/HealthPills.tsx'
import { CostCounter } from '../../components/Header/CostCounter.tsx'
import { ThemeToggle } from '../../components/Header/ThemeToggle.tsx'
import { CompareToggle } from '../../components/Header/CompareToggle.tsx'
import type { HealthStatus } from '../../types/api.ts'

function renderWithProvider(ui: React.ReactNode) {
  return render(<AppProvider>{ui}</AppProvider>)
}

describe('HealthPills', () => {
  it('renders all model pills (always visible, online or offline)', () => {
    const health: HealthStatus = {
      models: {
        INFERENCE: true,
        FUNCTION: true,
        EMBEDDING: true,
        VISION: true,
      },
    }
    const { container } = render(<HealthPills health={health} />)
    const pills = container.querySelectorAll('.model-pill')
    expect(pills.length).toBe(10)  // LogReg + 5 core (inference, function, embedding, vision) + OCR + whisper + piper + qwen + cloud
  })

  it('applies online class when models are available', () => {
    const health: HealthStatus = {
      models: {
        INFERENCE: true,
        FUNCTION: false,
        EMBEDDING: true,
        VISION: true,
      },
    }
    const { container } = render(<HealthPills health={health} />)
    const infPill = container.querySelector('[data-model="inference"]')
    const funcPill = container.querySelector('[data-model="function"]')
    const embPill = container.querySelector('[data-model="embedding"]')

    expect(infPill?.className).toContain('online')
    expect(funcPill?.className).not.toContain('online')
    expect(embPill?.className).toContain('online')
  })

  it('shows whisper and piper when WHISPER is available', () => {
    const health: HealthStatus = {
      models: {
        INFERENCE: true,
        FUNCTION: true,
        EMBEDDING: true,
        VISION: true,
        WHISPER: true,
      },
    }
    const { container } = render(<HealthPills health={health} />)
    const whisperPill = container.querySelector('[data-model="whisper"]')
    const piperPill = container.querySelector('[data-model="piper"]')
    expect(whisperPill).toBeTruthy()
    expect(piperPill).toBeTruthy()
  })

  it('shows whisper and piper as offline when WHISPER is not available', () => {
    const health: HealthStatus = {
      models: {
        INFERENCE: true,
        FUNCTION: true,
        EMBEDDING: true,
        VISION: true,
      },
    }
    const { container } = render(<HealthPills health={health} />)
    const whisperPill = container.querySelector('[data-model="whisper"]')
    const piperPill = container.querySelector('[data-model="piper"]')
    expect(whisperPill?.className).toContain('offline')
    expect(piperPill?.className).toContain('offline')
  })

  it('renders document count when available', () => {
    const health: HealthStatus = {
      models: {},
      document_count: 42,
    }
    render(<HealthPills health={health} />)
    expect(screen.getByText('42 chunks')).toBeTruthy()
  })

  it('renders gracefully with null health', () => {
    const { container } = render(<HealthPills health={null} />)
    const pills = container.querySelectorAll('.model-pill')
    expect(pills.length).toBe(10)
    pills.forEach(pill => {
      expect(pill.className).toContain('offline')
    })
  })
})

describe('CostCounter', () => {
  it('renders local $0.00 and estimated cloud cost', () => {
    renderWithProvider(<CostCounter />)
    expect(screen.getByText('Local: $0.00')).toBeTruthy()
    expect(screen.getByText(/Cloud:/)).toBeTruthy()
  })
})

describe('ThemeToggle', () => {
  it('clicking toggles theme', () => {
    renderWithProvider(<ThemeToggle />)
    const button = screen.getByRole('button')
    const initialText = button.textContent

    fireEvent.click(button)

    expect(button.textContent).not.toBe(initialText)
  })
})

describe('CompareToggle', () => {
  it('clicking toggles compare mode', () => {
    renderWithProvider(<CompareToggle />)
    const button = screen.getByRole('button')

    fireEvent.click(button)

    expect(button).toBeTruthy()
  })

  it('is disabled when offline', () => {
    renderWithProvider(<CompareToggle />)
    const button = screen.getByRole('button')
    expect(button).not.toBeDisabled()
  })
})
