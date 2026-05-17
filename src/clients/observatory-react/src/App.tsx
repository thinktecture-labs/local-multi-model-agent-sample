import { useEffect, useState, useCallback } from 'react'
import { useAppState, useAppDispatch } from './state/AppContext.tsx'
import { Header } from './components/Header/Header.tsx'
import { ConversationPane } from './components/Conversation/ConversationPane.tsx'
import { TracePaneContainer } from './components/Panels/TracePaneContainer.tsx'
import { ShowScreen } from './components/ShowMode/ShowScreen.tsx'
import { useHealthPolling } from './hooks/useHealthPolling.ts'
import { useStatsWebSocket } from './hooks/useStatsWebSocket.ts'
import { getModelMode, getEvalResults } from './api/client.ts'

export default function App() {
  const { theme, traceCollapsed } = useAppState()
  const dispatch = useAppDispatch()
  const health = useHealthPolling()
  const { gpu, energy } = useStatsWebSocket()
  const [showMode, setShowMode] = useState(false)
  const exitShowMode = useCallback(() => setShowMode(false), [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    getModelMode()
      .then(d => {
        dispatch({
          type: 'SET_MODEL_MODE',
          mode: d.mode === 'finetuned' ? 'finetuned' : 'base',
        })
      })
      .catch(() => {})
  }, [dispatch])

  useEffect(() => {
    getEvalResults()
      .then(d => {
        if (d.before) dispatch({ type: 'SET_EVAL_RESULT', data: d.before })
        if (d.after) dispatch({ type: 'SET_EVAL_RESULT', data: d.after })
      })
      .catch(() => {})
  }, [dispatch])

  // Cmd+Shift+P toggles Show Mode
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.shiftKey && e.key === 'p') {
        e.preventDefault()
        setShowMode(prev => !prev)
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [])

  if (showMode) return <ShowScreen onExit={exitShowMode} />

  return (
    <div className="app">
      <Header health={health} onShowMode={() => setShowMode(true)} />
      <main className={traceCollapsed ? 'trace-collapsed' : ''}>
        <ConversationPane health={health} />
        <TracePaneContainer gpu={gpu} energy={energy} health={health} />
      </main>
    </div>
  )
}
