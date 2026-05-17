import type { GpuStats, EnergyStats, HealthStatus } from '../../types/api.ts'
import { TracePanel } from '../Trace/TracePanel.tsx'
import { GpuPanel } from './GpuPanel.tsx'
import { EnergyPanel } from './EnergyPanel.tsx'
import { EvalPanel } from './EvalPanel.tsx'
import { FlywheelPanel } from './FlywheelPanel.tsx'

interface TracePaneContainerProps {
  gpu: GpuStats | null
  energy: EnergyStats | null
  health: HealthStatus | null
}

export function TracePaneContainer({ gpu, energy, health }: TracePaneContainerProps) {
  const interactionCount = health?.interaction_count ?? 0

  return (
    <aside className="trace-pane">
      <TracePanel />
      <GpuPanel gpu={gpu} />
      <EnergyPanel energy={energy} />
      <EvalPanel />
      <FlywheelPanel interactionCount={interactionCount} />
    </aside>
  )
}
