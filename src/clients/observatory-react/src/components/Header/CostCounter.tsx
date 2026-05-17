import { useAppState } from '../../state/AppContext.tsx'
import { CLOUD_PRICING } from '../../utils/format.ts'

export function CostCounter() {
  const { sessionTokens, cloudCostActual } = useAppState()

  const estimatedCloud =
    (sessionTokens.prompt / 1_000_000) * CLOUD_PRICING.input_per_1m +
    (sessionTokens.completion / 1_000_000) * CLOUD_PRICING.output_per_1m

  const cloudCost = cloudCostActual > 0 ? cloudCostActual : estimatedCloud

  const cloudDisplay = cloudCost === 0 && sessionTokens.total === 0
    ? '$0.00'
    : `$${cloudCost.toFixed(4)}`

  return (
    <div className="cost-counter">
      <span className="cost-local" title="Local inference cost">
        Local: $0.00
      </span>
      <span className="cost-sep">|</span>
      <span className="cost-cloud" title={cloudCostActual > 0 ? 'Actual cloud cost' : 'Estimated cloud cost'}>
        Cloud: {cloudDisplay}{cloudCostActual === 0 && sessionTokens.total > 0 ? '*' : ''}
      </span>
    </div>
  )
}
