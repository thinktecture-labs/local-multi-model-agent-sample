export function EscalationBanner({
  exchangeIdx: _exchangeIdx,
  onEscalate,
  isEscalating,
}: {
  exchangeIdx: number
  onEscalate: () => void
  isEscalating: boolean
}) {
  return (
    <div className="escalation-banner">
      <span className="escalation-label">BELOW THRESHOLD</span>
      <span className="escalation-text">
        Low confidence score — consider escalating to cloud model for better accuracy.
      </span>
      <button
        className="escalation-btn"
        onClick={onEscalate}
        disabled={isEscalating}
      >
        {isEscalating ? 'Escalating...' : 'Escalate to Cloud?'}
      </button>
    </div>
  )
}
