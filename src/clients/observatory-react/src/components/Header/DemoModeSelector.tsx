import type { HealthStatus, DemoMode } from '../../types/api.ts'
import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'

const MODES: { value: DemoMode; label: string; title?: string; needsQwen?: boolean; needsCloud?: boolean }[] = [
  { value: 'multi-models', label: 'Multi-Models', title: '5 specialized dense models (1B-4B)' },
  { value: 'qwen', label: 'MoE', title: 'Qwen 3.5 35B-A3B MoE — single Mixture-of-Experts model', needsQwen: true },
  { value: 'cloud', label: 'Cloud', title: 'GPT-5.4 via OpenAI API', needsCloud: true },
  { value: 'all', label: 'All', title: 'All three backends in parallel' },
]

export function DemoModeSelector({ health }: { health: HealthStatus | null }) {
  const { demoMode, networkMode } = useAppState()
  const dispatch = useAppDispatch()

  const qwenAvailable = !!health?.models?.['QWEN']
  const cloudAvailable = !!health?.models?.['CLOUD'] && networkMode !== 'offline'

  const isDisabled = (mode: typeof MODES[number]) => {
    if (mode.needsQwen && !qwenAvailable) return true
    if (mode.needsCloud && !cloudAvailable) return true
    if (mode.value === 'all' && !qwenAvailable && !cloudAvailable) return true
    return false
  }

  return (
    <div className="demo-mode-selector">
      {MODES.map((mode) => {
        const disabled = isDisabled(mode)
        return (
          <button
            key={mode.value}
            className={`demo-mode-btn${demoMode === mode.value ? ' active' : ''}${disabled ? ' disabled' : ''}`}
            onClick={() => !disabled && dispatch({ type: 'SET_DEMO_MODE', mode: mode.value })}
            disabled={disabled}
            title={disabled ? `${mode.label} not available` : (mode.title ?? mode.label)}
          >
            {mode.label}
          </button>
        )
      })}
    </div>
  )
}
