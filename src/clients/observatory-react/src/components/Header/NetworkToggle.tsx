import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'
import { setNetworkMode } from '../../api/client.ts'

export function NetworkToggle() {
  const { networkMode } = useAppState()
  const dispatch = useAppDispatch()

  const isOffline = networkMode === 'offline'

  const handleToggle = async () => {
    try {
      const res = await setNetworkMode()
      const newMode = res.network_mode as 'online' | 'offline'
      dispatch({ type: 'SET_NETWORK_MODE', mode: newMode })
      if (newMode === 'offline') {
        dispatch({ type: 'SET_COMPARE_MODE', enabled: false })
      }
    } catch (err) {
      console.error('Network mode toggle failed:', err)
    }
  }

  return (
    <button
      className={`net-btn${isOffline ? ' offline' : ''}`}
      onClick={handleToggle}
      title={isOffline ? 'Go online' : 'Go offline (airplane mode)'}
    >
      {'\u2708\uFE0E'}
    </button>
  )
}
