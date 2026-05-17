import type { GpuStats } from '../../types/api.ts'

interface GpuPanelProps {
  gpu: GpuStats | null
}

export function GpuPanel({ gpu }: GpuPanelProps) {
  const g = gpu
  const vramPct = g?.available && g.vram_total_mb > 0
    ? (g.vram_used_mb / g.vram_total_mb) * 100
    : 0
  const utilPct = g?.available ? g.utilization_pct : 0

  const gpuName = g?.available ? `${g.name} (${g.backend.toUpperCase()})` : 'CPU mode'
  const vramVal = g?.available && g.vram_total_mb > 0
    ? `${(g.vram_used_mb / 1024).toFixed(1)} / ${(g.vram_total_mb / 1024).toFixed(1)} GB`
    : 'N/A'
  const utilVal = g?.available && g.utilization_pct > 0 ? `${g.utilization_pct}%` : 'N/A'
  const tempVal = g?.available && g.temperature_c > 0 ? `${g.temperature_c} \u00B0C` : 'N/A'

  return (
    <div className="gpu-panel">
      <div className="gpu-header">
        <span className="trace-title" style={{ fontSize: '11px' }}>GPU</span>
        <span className="gpu-name">{gpuName}</span>
      </div>
      <div className="gpu-stats">
        <div className="gpu-stat">
          <span className="gpu-stat-label">VRAM</span>
          <div className="gpu-bar-track">
            <div className="gpu-bar-fill" style={{ width: `${vramPct}%` }}></div>
          </div>
          <span className="gpu-stat-val">{vramVal}</span>
        </div>
        <div className="gpu-stat">
          <span className="gpu-stat-label">Util</span>
          <div className="gpu-bar-track">
            <div className="gpu-bar-fill util" style={{ width: `${utilPct}%` }}></div>
          </div>
          <span className="gpu-stat-val">{utilVal}</span>
        </div>
        <div className="gpu-stat">
          <span className="gpu-stat-label">Temp</span>
          <span className="gpu-stat-val">{tempVal}</span>
        </div>
      </div>
    </div>
  )
}
