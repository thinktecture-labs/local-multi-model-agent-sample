import { useAppState } from '../../state/AppContext.tsx'
import { formatBytes } from '../../utils/format.ts'

export function PrivacyBadge() {
  const { cloudBytesSent, networkMode } = useAppState()

  const isOffline = networkMode === 'offline'

  return (
    <div className={`privacy-badge${isOffline ? ' offline' : ''}`} title="Zero data exfiltration">
      <span className="privacy-icon">{'\uD83D\uDD12'}</span>
      <span className="privacy-text">
        {isOffline ? 'Air-gapped' : cloudBytesSent === 0 ? '0 bytes sent externally' : `${formatBytes(cloudBytesSent)} sent externally`}
      </span>
    </div>
  )
}
