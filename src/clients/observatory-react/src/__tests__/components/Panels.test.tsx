import { render, screen } from '@testing-library/react'
import { AppProvider } from '../../state/AppContext.tsx'
import { GpuPanel } from '../../components/Panels/GpuPanel.tsx'
import { EnergyPanel } from '../../components/Panels/EnergyPanel.tsx'
import { EvalPanel } from '../../components/Panels/EvalPanel.tsx'
import type { GpuStats, EnergyStats } from '../../types/api.ts'

function renderWithProvider(ui: React.ReactNode) {
  return render(<AppProvider>{ui}</AppProvider>)
}

describe('GpuPanel', () => {
  it('renders GPU name when available', () => {
    const gpu: GpuStats = {
      available: true,
      name: 'NVIDIA RTX 4090',
      backend: 'CUDA',
      vram_used_mb: 4096,
      vram_total_mb: 24576,
      utilization_pct: 45,
      temperature_c: 65,
    }
    render(<GpuPanel gpu={gpu} />)
    expect(screen.getByText('NVIDIA RTX 4090 (CUDA)')).toBeTruthy()
  })

  it('shows CPU mode when not available', () => {
    const { container } = render(<GpuPanel gpu={null} />)
    const gpuName = container.querySelector('.gpu-name')
    expect(gpuName?.textContent).toBe('CPU mode')
  })

  it('shows CPU mode when gpu.available is false', () => {
    const gpu: GpuStats = {
      available: false,
      name: '',
      backend: '',
      vram_used_mb: 0,
      vram_total_mb: 0,
      utilization_pct: 0,
      temperature_c: 0,
    }
    const { container } = render(<GpuPanel gpu={gpu} />)
    const gpuName = container.querySelector('.gpu-name')
    expect(gpuName?.textContent).toBe('CPU mode')
  })

  it('renders VRAM and utilization stats', () => {
    const gpu: GpuStats = {
      available: true,
      name: 'RTX 4090',
      backend: 'CUDA',
      vram_used_mb: 8192,
      vram_total_mb: 24576,
      utilization_pct: 72,
      temperature_c: 70,
    }
    render(<GpuPanel gpu={gpu} />)
    expect(screen.getByText('8.0 / 24.0 GB')).toBeTruthy()
    expect(screen.getByText('72%')).toBeTruthy()
    expect(screen.getByText('70 \u00B0C')).toBeTruthy()
  })
})

describe('EnergyPanel', () => {
  it('renders power and session Wh values', () => {
    const energy: EnergyStats = {
      backend: 'metal',
      sample_count: 100,
      gpu_power_now_w: 15.5,
      system_power_now_w: 0,
      total_wh: 0.1234,
      total_queries: 10,
      wh_per_query: 0.0123,
      co2_local_g: 0.05,
      co2_cloud_g: 0.15,
      electricity_cost_local: 0.000012,
      estimated_cloud_wh: 0.5,
    }
    render(<EnergyPanel energy={energy} />)
    expect(screen.getByText(/Metal/)).toBeTruthy()
    expect(screen.getByText('0.123 Wh')).toBeTruthy()
  })

  it('shows waiting state when energy is null', () => {
    const { container } = render(<EnergyPanel energy={null} />)
    const gpuName = container.querySelector('.gpu-name')
    expect(gpuName?.textContent).toBe('waiting\u2026')
  })
})

describe('EvalPanel', () => {
  it('renders before and after columns', () => {
    renderWithProvider(<EvalPanel />)
    expect(screen.getByText('Before')).toBeTruthy()
    expect(screen.getByText('After')).toBeTruthy()
  })

  it('renders Run Eval button', () => {
    renderWithProvider(<EvalPanel />)
    expect(screen.getByText('Run Eval')).toBeTruthy()
  })

  it('shows placeholder dashes when no eval data', () => {
    renderWithProvider(<EvalPanel />)
    const dashes = screen.getAllByText('--')
    expect(dashes.length).toBe(2)
  })
})
