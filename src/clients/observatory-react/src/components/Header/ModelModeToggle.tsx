import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'
import { swapModels } from '../../api/client.ts'

export function ModelModeToggle() {
  const { modelMode, modelSwapping } = useAppState()
  const dispatch = useAppDispatch()

  const isFinetuned = modelMode === 'finetuned'

  const handleToggle = async () => {
    if (modelSwapping) return
    const newMode = isFinetuned ? 'base' : 'finetuned'
    dispatch({ type: 'SET_MODEL_SWAPPING', swapping: true })
    try {
      await swapModels(newMode)
      dispatch({ type: 'SET_MODEL_MODE', mode: newMode })
      dispatch({ type: 'CLEAR_HISTORY' })
    } catch (err) {
      console.error('Model swap failed:', err)
    } finally {
      dispatch({ type: 'SET_MODEL_SWAPPING', swapping: false })
    }
  }

  return (
    <div className={`mode-toggle${modelSwapping ? ' swapping' : ''}`}>
      <span className="mode-label">{isFinetuned ? 'FINE-TUNED' : 'BASE'}</span>
      <button
        className={`mode-switch${isFinetuned ? ' active' : ''}`}
        onClick={handleToggle}
        disabled={modelSwapping}
        title={`Toggle base/fine-tuned`}
      >
        <div className="mode-knob"></div>
      </button>
    </div>
  )
}
