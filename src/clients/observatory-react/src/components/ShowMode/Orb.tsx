import { useRef, useEffect, useCallback } from 'react'

export type OrbPhase = 'idle' | 'listening' | 'recording' | 'processing' | 'speaking'

const PHASE_COLORS: Record<OrbPhase, string> = {
  idle:       '#0170B9',
  listening:  '#34A853',
  recording:  '#ef4444',
  processing: '#f59e0b',
  speaking:   '#0170B9',
}

const PHASE_GLOW: Record<OrbPhase, string> = {
  idle:       'rgba(1, 112, 185, 0.3)',
  listening:  'rgba(52, 168, 83, 0.4)',
  recording:  'rgba(239, 68, 68, 0.5)',
  processing: 'rgba(245, 158, 11, 0.4)',
  speaking:   'rgba(1, 112, 185, 0.4)',
}

interface OrbProps {
  phase: OrbPhase
  analyser: AnalyserNode | null
}

export function Orb({ phase, analyser }: OrbProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef(0)
  const timeRef = useRef(0)

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Use logical size (ctx is already scaled by DPR)
    const dpr = window.devicePixelRatio || 1
    const W = canvas.width / dpr
    const H = canvas.height / dpr
    const cx = W / 2
    const cy = H / 2
    const baseR = Math.min(W, H) * 0.28
    const t = timeRef.current

    ctx.clearRect(0, 0, W, H)

    const color = PHASE_COLORS[phase]
    const glow = PHASE_GLOW[phase]

    // Get audio data if available — collapse the spectrum to a single
    // average amplitude so the orb pulses symmetrically (mapping per-bin
    // frequency to perimeter angle creates an asymmetric spike on the side
    // dominated by the loudest frequencies).
    let audioAmp = 0  // 0..1
    if (analyser && (phase === 'recording' || phase === 'speaking')) {
      const audioData = new Uint8Array(analyser.frequencyBinCount) as Uint8Array<ArrayBuffer>
      analyser.getByteFrequencyData(audioData)
      let sum = 0
      for (let j = 0; j < audioData.length; j++) sum += audioData[j]
      audioAmp = (sum / audioData.length) / 255
    }

    // Outer glow
    const breathe = phase === 'idle' ? 0.6 + 0.4 * Math.sin(t * 0.8) : 1
    const glowR = baseR * 1.6
    const grad = ctx.createRadialGradient(cx, cy, baseR * 0.3, cx, cy, glowR)
    grad.addColorStop(0, glow.replace(/[\d.]+\)$/, `${0.35 * breathe})`))
    grad.addColorStop(0.5, glow.replace(/[\d.]+\)$/, `${0.12 * breathe})`))
    grad.addColorStop(1, 'transparent')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, W, H)

    // Draw waveform ring
    const segments = 128
    ctx.beginPath()
    for (let i = 0; i <= segments; i++) {
      const angle = (i / segments) * Math.PI * 2 - Math.PI / 2
      let displacement = 0

      if (audioAmp > 0.02) {
        // Real-time audio amplitude → symmetric pulse + subtle organic shimmer
        displacement = audioAmp * baseR * 0.4
          + (Math.sin(angle * 3 + t * 1.4) * 0.025
             + Math.sin(angle * 5 - t * 1.8) * 0.018) * baseR
      } else if (phase === 'processing') {
        // Spinning wave during processing
        displacement = Math.sin(angle * 6 + t * 4) * baseR * 0.12
          + Math.sin(angle * 3 - t * 2.5) * baseR * 0.08
      } else {
        // Organic breathing — layered waves at different frequencies
        displacement = (
          Math.sin(angle * 3 + t * 0.9) * 0.06
          + Math.sin(angle * 5 - t * 1.4) * 0.04
          + Math.sin(angle * 7 + t * 2.1) * 0.025
          + Math.sin(angle * 2 - t * 0.6) * 0.035
          + Math.cos(angle * 4 + t * 1.7) * 0.03
        ) * baseR * breathe
      }

      const r = baseR + displacement
      const x = cx + Math.cos(angle) * r
      const y = cy + Math.sin(angle) * r

      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.closePath()

    // Fill
    const fillGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, baseR * 1.3)
    fillGrad.addColorStop(0, color + '30')
    fillGrad.addColorStop(1, color + '08')
    ctx.fillStyle = fillGrad
    ctx.fill()

    // Stroke
    ctx.strokeStyle = color
    ctx.lineWidth = 2.5
    ctx.shadowColor = glow
    ctx.shadowBlur = phase === 'idle' ? 15 : 25
    ctx.stroke()
    ctx.shadowBlur = 0

    // Inner circle (subtle)
    ctx.beginPath()
    ctx.arc(cx, cy, baseR * 0.15, 0, Math.PI * 2)
    ctx.fillStyle = color + '40'
    ctx.fill()

    timeRef.current += 0.016
    frameRef.current = requestAnimationFrame(draw)
  }, [phase, analyser])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    // Set canvas resolution for retina
    const dpr = window.devicePixelRatio || 1
    const size = 320
    canvas.width = size * dpr
    canvas.height = size * dpr
    canvas.style.width = `${size}px`
    canvas.style.height = `${size}px`
    const ctx = canvas.getContext('2d')
    if (ctx) ctx.scale(dpr, dpr)
  }, [])

  useEffect(() => {
    frameRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(frameRef.current)
  }, [draw])

  return <canvas ref={canvasRef} className="show-orb" />
}
