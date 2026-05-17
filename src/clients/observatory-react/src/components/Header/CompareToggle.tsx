import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'

export function CompareToggle() {
  const { compareMode, networkMode } = useAppState()
  const dispatch = useAppDispatch()

  const disabled = networkMode === 'offline'

  const handleToggle = () => {
    if (disabled) return
    dispatch({ type: 'SET_COMPARE_MODE', enabled: !compareMode })
  }

  return (
    <button
      className={`compare-btn${compareMode ? ' active' : ''}`}
      onClick={handleToggle}
      disabled={disabled}
      title={disabled ? 'Compare requires online mode' : 'Compare local vs cloud LLM'}
    >
      {'\u21C4'}
    </button>
  )
}
