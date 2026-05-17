import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'
import { runEval } from '../../api/client.ts'
import type { EvalResult } from '../../types/api.ts'

export function EvalPanel() {
  const { eval: evalState } = useAppState()
  const dispatch = useAppDispatch()

  const hasData = evalState.before != null || evalState.after != null

  const handleRunEval = async () => {
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
    <div className={`eval-panel${hasData ? ' active' : ''}`}>
      <div className="eval-compare">
        <EvalColumn label="Before" data={evalState.before} />
        <EvalColumn label="After" data={evalState.after} />
      </div>

      <button
        className="fw-eval-btn"
        onClick={handleRunEval}
        disabled={evalState.loading}
      >
        {evalState.loading ? 'Running...' : 'Run Eval'}
      </button>
    </div>
  )
}

function EvalColumn({ label, data }: { label: string; data: EvalResult | null }) {
  if (!data) {
    return (
      <div className="eval-col">
        <div className="eval-col-title">{label}</div>
        <div className="eval-pct">--</div>
      </div>
    )
  }

  const overallPct = Math.round(data.overall_accuracy * 100)
  const classes = Object.entries(data.per_class)

  return (
    <div className="eval-col">
      <div className="eval-col-title">{label}</div>
      <div className="eval-pct">{overallPct}%</div>
      <div className="eval-sub">{data.overall_correct} / {data.n}</div>

      {classes.map(([cls, info]) => {
        const acc = info.accuracy ?? (info.n > 0 ? info.correct / info.n : 0)
        const accPct = Math.round(acc * 100)

        return (
          <div className="eval-bar-row" key={cls}>
            <span className="eval-bar-label">{cls}</span>
            <div className="eval-bar-track">
              <div className="eval-bar-fill" style={{ width: `${accPct}%` }}></div>
            </div>
            <span className="eval-bar-val">{accPct}%</span>
          </div>
        )
      })}
    </div>
  )
}
