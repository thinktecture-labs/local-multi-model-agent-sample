import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'
import { setRoutingMode } from '../../api/client.ts'

export function RoutingToggle() {
  const { routingMode, networkMode } = useAppState()
  const dispatch = useAppDispatch()

  const isHybrid = routingMode === 'hybrid'
  const isOffline = networkMode === 'offline'

  const handleToggle = async () => {
    if (isOffline) return
    try {
      const res = await setRoutingMode()
      const newMode = res.routing_mode as 'local' | 'hybrid'
      dispatch({ type: 'SET_ROUTING_MODE', mode: newMode })
    } catch (err) {
      console.error('Routing mode toggle failed:', err)
    }
  }

  return (
    <div className={`route-toggle${isOffline ? ' disabled' : ''}`}>
      <span className="route-label">{isHybrid && !isOffline ? 'HYBRID' : 'LOCAL ONLY'}</span>
      <button
        className={`route-switch${isHybrid && !isOffline ? ' active' : ''}`}
        onClick={handleToggle}
        disabled={isOffline}
        title={isOffline ? 'Offline — cloud escalation unavailable' : 'Toggle local-only / hybrid routing'}
      >
        <div className="route-knob"></div>
      </button>
    </div>
  )
}
