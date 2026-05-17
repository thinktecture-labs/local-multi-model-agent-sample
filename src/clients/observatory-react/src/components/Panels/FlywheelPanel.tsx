import { useAppDispatch } from '../../state/AppContext.tsx'
import { runEval } from '../../api/client.ts'

interface FlywheelPanelProps {
  interactionCount: number
}

export function FlywheelPanel({ interactionCount }: FlywheelPanelProps) {
  const dispatch = useAppDispatch()

  const handleEval = async () => {
    dispatch({ type: 'SET_EVAL_LOADING', loading: true })
    try {
      const result = await runEval()
      dispatch({ type: 'SET_EVAL_RESULT', data: result })
    } catch (err) {
      console.error('Eval failed:', err)
    } finally {
      dispatch({ type: 'SET_EVAL_LOADING', loading: false })
    }
  }

  return (
    <div className="flywheel-panel">
      <div className="fw-row">
        <div className="flywheel-flow">
          <span className="fw-step fw-active">Use</span>
          <span className="fw-arrow">{'\u2192'}</span>
          <span className={`fw-step${interactionCount > 0 ? ' fw-active' : ''}`} id="fw-log-step">Log</span>
          <span className="fw-arrow">{'\u2192'}</span>
          <span className="fw-step">Train</span>
          <span className="fw-arrow">{'\u2192'}</span>
          <span className="fw-step">Deploy</span>
        </div>
        <div className="fw-count">{interactionCount} logged</div>
        <button className="fw-eval-btn" onClick={handleEval}>Eval</button>
      </div>
    </div>
  )
}
