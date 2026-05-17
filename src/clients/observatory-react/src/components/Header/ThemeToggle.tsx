import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'

export function ThemeToggle() {
  const { theme } = useAppState()
  const dispatch = useAppDispatch()

  const handleToggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    dispatch({ type: 'SET_THEME', theme: next })
  }

  return (
    <button
      className="theme-btn"
      onClick={handleToggle}
      title="Toggle light/dark"
    >
      {theme === 'light' ? '\u263E' : '\u2600\uFE0E'}
    </button>
  )
}
