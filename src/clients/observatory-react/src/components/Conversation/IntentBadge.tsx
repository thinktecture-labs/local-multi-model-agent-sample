import type { Intent } from '../../types/api.ts'
import { intentBadgeLabel } from '../../utils/format.ts'

export function IntentBadge({ intent }: { intent: Intent | string }) {
  return <span className={`intent-badge ${intent}`}>{intentBadgeLabel(intent)}</span>
}
