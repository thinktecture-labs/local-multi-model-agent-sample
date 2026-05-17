import type { CompareResult } from '../../types/api.ts'
import { formatMs, formatResponse } from '../../utils/format.ts'

export function CompareView({ compareResult }: { compareResult: CompareResult }) {
  return (
    <div className="compare-split">
      <div className="compare-col compare-local">
        <div className="compare-header">
          <span className="compare-label">LOCAL</span>
          <span className="compare-meta">
            {formatMs(compareResult.local_latency_ms)} &middot; $0.00
          </span>
        </div>
        <div
          className="compare-body"
          dangerouslySetInnerHTML={{ __html: formatResponse(compareResult.local_response) }}
        />
      </div>
      <div className="compare-col compare-cloud">
        <div className="compare-header">
          <span className="compare-label">CLOUD</span>
          <span className="compare-meta">
            {compareResult.cloud_latency_ms != null
              ? formatMs(compareResult.cloud_latency_ms)
              : '--'}
            {compareResult.cloud_cost != null || compareResult.estimated_cloud_cost != null
              ? ` · $${(compareResult.cloud_cost ?? compareResult.estimated_cloud_cost ?? 0).toFixed(4)}`
              : ''}
          </span>
        </div>
        <div
          className="compare-body"
          dangerouslySetInnerHTML={{
            __html: formatResponse(
              compareResult.cloud_response ?? `Estimated cost: $${(compareResult.estimated_cloud_cost ?? 0).toFixed(4)}`,
            ),
          }}
        />
      </div>
    </div>
  )
}
