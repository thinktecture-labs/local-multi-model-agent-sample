import type { EnergyStats } from '../../types/api.ts'

interface EnergyPanelProps {
  energy: EnergyStats | null
}

export function EnergyPanel({ energy }: EnergyPanelProps) {
  const e = energy
  const isCuda = e?.backend?.toLowerCase().includes('cuda')
  const maxPower = isCuda ? 300 : 40
  const powerPct = e ? Math.min((e.gpu_power_now_w / maxPower) * 100, 100) : 0

  const backendLabel = e
    ? e.backend === 'cuda' ? 'CUDA' : e.backend === 'metal' ? 'Metal' : 'CPU'
    : '\u2014'
  const headerRight = e && e.sample_count > 0
    ? `${backendLabel} \u2022 ${e.sample_count} samples`
    : 'waiting\u2026'

  const powerVal = e
    ? e.system_power_now_w > 0
      ? `${e.gpu_power_now_w.toFixed(1)} W (sys ${e.system_power_now_w.toFixed(0)} W)`
      : `${e.gpu_power_now_w.toFixed(1)} W`
    : '\u2014 W'

  const sessionVal = e
    ? e.total_wh < 0.01
      ? `${(e.total_wh * 1000).toFixed(1)} mWh`
      : `${e.total_wh.toFixed(3)} Wh`
    : '0.000 Wh'

  const perQueryVal = e
    ? e.total_queries > 0
      ? e.wh_per_query < 0.01
        ? `${(e.wh_per_query * 1000).toFixed(1)} mWh`
        : `${e.wh_per_query.toFixed(3)} Wh`
      : '\u2014'
    : '\u2014'

  const co2LocalVal = e
    ? e.co2_local_g < 0.01
      ? `${(e.co2_local_g * 1000).toFixed(1)} mg`
      : `${e.co2_local_g.toFixed(2)} g`
    : '0.00 g'

  const co2CloudVal = e
    ? e.co2_cloud_g < 0.01
      ? `~${(e.co2_cloud_g * 1000).toFixed(1)} mg`
      : `~${e.co2_cloud_g.toFixed(2)} g`
    : '~0.00 g'

  const costLocalVal = e
    ? e.electricity_cost_local < 0.01
      ? `\u20AC${e.electricity_cost_local.toFixed(6)}`
      : `\u20AC${e.electricity_cost_local.toFixed(4)}`
    : '\u20AC0.000000'

  const cloudEstVal = e
    ? e.estimated_cloud_wh < 0.01
      ? `~${(e.estimated_cloud_wh * 1000).toFixed(1)} mWh`
      : `~${e.estimated_cloud_wh.toFixed(3)} Wh`
    : '~0.000 Wh'

  return (
    <div className="energy-panel">
      <div className="energy-header">
        <span className="trace-title" style={{ fontSize: '11px' }}>Energy</span>
        <span className="gpu-name">{headerRight}</span>
      </div>
      <div className="energy-power-bar">
        <span className="energy-power-label">Power</span>
        <div className="energy-power-track">
          <div className="energy-power-fill" style={{ width: `${powerPct}%` }}></div>
        </div>
        <span className="energy-power-val">{powerVal}</span>
      </div>
      <div className="energy-grid">
        <div className="energy-row">
          <span className="energy-label">Session</span>
          <span className="energy-val local">{sessionVal}</span>
        </div>
        <div className="energy-row">
          <span className="energy-label">Per query</span>
          <span className="energy-val local">{perQueryVal}</span>
        </div>
        <div className="energy-divider"></div>
        <div className="energy-row">
          <span className="energy-label">CO2 local</span>
          <span className="energy-val local">{co2LocalVal}</span>
        </div>
        <div className="energy-row">
          <span className="energy-label">CO2 cloud</span>
          <span className="energy-val cloud">{co2CloudVal}</span>
        </div>
        <div className="energy-row">
          <span className="energy-label">Cost local</span>
          <span className="energy-val local">{costLocalVal}</span>
        </div>
        <div className="energy-row">
          <span className="energy-label">Cloud est.</span>
          <span className="energy-val cloud">{cloudEstVal}</span>
        </div>
      </div>
    </div>
  )
}
