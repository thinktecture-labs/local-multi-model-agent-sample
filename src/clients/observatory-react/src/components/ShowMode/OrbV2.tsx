import { useEffect, useMemo, useRef } from 'react'
import type { OrbPhase as ShowOrbPhase } from './Orb'

/**
 * Jarvis-style orb — React port of the Angular orb.component.ts on
 * talk-to-tt-vnext's feature/spa-orb-visualization branch.
 *
 * Composition (drawn back-to-front):
 *
 *   1. Outer torus — particle cloud on a (u, v)-parameterised torus,
 *      tilted, with a 3D cross-section wobble + symmetric audio-driven
 *      rim displacement on `listening` / `playing`.
 *   2. Rim contours — wavy strokes tracing outer + inner silhouettes.
 *   3. Concentric shells — four stroked tilted ellipses (78/62/46/30%).
 *   4. Scan arc — bright comet-tail sweeping the outer shell during
 *      `thinking`.
 *   5. TT logo SVG overlay — phase-tinted via CSS variable.
 *
 * Audio reactivity is symmetric (mix of peak + average from FFT) so it
 * never produces the per-segment asymmetric spike the legacy orb had.
 */

// Reuse phases from the legacy Orb to keep ShowScreen wiring simple. Internally
// we map our phases to the reference's set (idle/armed/listening/thinking/playing).
type OrbPhase = 'idle' | 'armed' | 'listening' | 'thinking' | 'playing'

interface OrbColors {
  /** Stroke + bright accents. Hex string. */
  core: string
  /** Halo / outer-glow tint. `rgba(...)` so alpha can be modulated. */
  glow: string
  /** Inner radial-gradient brightest stop. */
  innerBright: string
}

const ORB_PALETTE: Record<OrbPhase, OrbColors> = {
  idle:      { core: '#3D6FB4', glow: 'rgba(61, 111, 180, 0.30)', innerBright: 'rgba(61, 111, 180, 0.50)' },
  armed:     { core: '#3D6FB4', glow: 'rgba(61, 111, 180, 0.40)', innerBright: 'rgba(61, 111, 180, 0.62)' },
  listening: { core: '#FF584F', glow: 'rgba(255, 88, 79, 0.42)',   innerBright: 'rgba(255, 88, 79, 0.68)' },
  thinking:  { core: '#3D6FB4', glow: 'rgba(61, 111, 180, 0.45)', innerBright: 'rgba(61, 111, 180, 0.65)' },
  playing:   { core: '#16A34A', glow: 'rgba(22, 163, 74, 0.65)',  innerBright: 'rgba(22, 163, 74, 0.92)' },
}

/** Map our 5 phases (idle/listening/recording/processing/speaking) to the
 *  reference's 5 phases (idle/armed/listening/thinking/playing). */
function mapPhase(p: ShowOrbPhase): OrbPhase {
  switch (p) {
    case 'idle':       return 'idle'
    case 'listening':  return 'armed'      // wakeword listening = "armed"
    case 'recording':  return 'listening'  // mic capture = the reference "listening"
    case 'processing': return 'thinking'
    case 'speaking':   return 'playing'
  }
}

interface Particle {
  u: number
  v: number
  rOffset: number
  brightness: number
  wavePhase: number
}

interface OrbV2Props {
  phase: ShowOrbPhase
  /** Live audio analyser. Recording-stream during 'listening', playback during
   *  'playing', null otherwise. */
  analyser?: AnalyserNode | null
  /** Freeze the time accumulator. Used by parent to lock the visual when audio
   *  pauses, then resume seamlessly. */
  paused?: boolean
  /** Visible orb diameter in CSS pixels. */
  size?: number
  ariaLabel?: string
}

const PARTICLES = 1700
const TILT = 0.32
const TUBE_RATIO = 0.20
const CANVAS_PADDING = 1.8

function withAlpha(rgba: string, alpha: number): string {
  return rgba.replace(/[\d.]+\)$/, `${Math.max(0, Math.min(1, alpha)).toFixed(3)})`)
}

export function OrbV2({ phase, analyser, paused = false, size = 400 }: OrbV2Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef<number | null>(null)
  const elapsedRef = useRef(0)
  const lastFrameAtRef = useRef<number | null>(null)
  const freqBufferRef = useRef<ArrayBuffer | null>(null)
  // Pin live values into refs so the rAF loop closure stays stable.
  const phaseRef = useRef(phase)
  const analyserRef = useRef<AnalyserNode | null>(analyser ?? null)
  const pausedRef = useRef(paused)
  phaseRef.current = phase
  analyserRef.current = analyser ?? null
  pausedRef.current = paused

  const orbPhase = mapPhase(phase)

  // Particle cloud — stable across renders, reseeded only if PARTICLES changes
  const particles = useMemo<Particle[]>(
    () =>
      Array.from({ length: PARTICLES }, () => ({
        u: Math.random() * Math.PI * 2,
        v: Math.random() * Math.PI * 2,
        rOffset: 0.85 + Math.random() * 0.3,
        brightness: 0.45 + Math.random() * 0.55,
        wavePhase: Math.random() * Math.PI * 2,
      })),
    [],
  )

  // Set up canvas backing-store at DPR. Re-runs when size changes.
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    const canvasPx = size * CANVAS_PADDING
    canvas.width = canvasPx * dpr
    canvas.height = canvasPx * dpr
    canvas.style.width = `${canvasPx}px`
    canvas.style.height = `${canvasPx}px`
    const ctx = canvas.getContext('2d')
    if (ctx) {
      // Reset any prior transform before applying DPR scale.
      ctx.setTransform(1, 0, 0, 1, 0, 0)
      ctx.scale(dpr, dpr)
    }
  }, [size])

  // Animation loop
  useEffect(() => {
    const tick = (now: number) => {
      if (lastFrameAtRef.current === null) lastFrameAtRef.current = now
      const dt = (now - lastFrameAtRef.current) / 1000
      lastFrameAtRef.current = now
      if (!pausedRef.current) {
        elapsedRef.current += Math.min(dt, 0.1)
      }
      draw(
        canvasRef.current,
        size,
        elapsedRef.current,
        mapPhase(phaseRef.current),
        analyserRef.current,
        freqBufferRef,
        particles,
      )
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      lastFrameAtRef.current = null
    }
  }, [size, particles])

  return (
    <div className="orb-v2-host" data-state={orbPhase} style={{ width: size, height: size }}>
      <canvas ref={canvasRef} className="orb-v2-canvas" aria-hidden="true" />
      {/* TT brand mark — phase-tinted via the host's --orb-v2-logo-color
          CSS variable, set by [data-state] selectors in showmode.css. */}
      <svg
        className="orb-v2-logo"
        viewBox="0 0 335.7 782.8"
        xmlns="http://www.w3.org/2000/svg"
        aria-hidden="true"
        focusable="false"
      >
        <polygon
          points="69.8,0 69.8,173 0,173 0,321.2 69.8,321.2 69.8,486.6 69.8,565.1 69.8,579.1 70.4,592.6 71.4,605.7 72.8,618.2 74.7,630.4 76.6,642 79.3,653 82,663.7 85.4,673.8 89,683.2 93,692.4 97.3,701.3 102.2,709.5 107.4,717.2 110.1,720.8 112.8,724.2 115.8,727.5 119,730.9 122,734.3 125.3,737.3 128.7,740.3 132,743.1 139.4,748.6 147,753.5 155.2,758.4 164.1,762.6 173.2,766.3 182.7,769.7 192.8,772.7 203.4,775.5 214.4,777.6 226.1,779.4 237.9,780.9 250.5,781.9 263.2,782.5 335.7,782.8 335.7,650.2 328.6,649.9 323.7,649.6 319.1,648.9 314.6,648.4 310.3,647.4 306,646.2 302.1,644.7 298.1,643.2 294.5,641 290.8,639.1 287.4,636.8 284.4,634.3 281.3,631.6 278.3,628.5 275.6,625.5 272.8,621.8 270.4,618.4 268.2,614.4 266.1,610.5 263.9,606.2 262.1,601.6 260.6,596.7 259.1,591.9 257.8,586.7 256.6,581.5 255.4,575.6 254.5,569.9 253.9,563.8 253.3,557.6 253,551.3 252.6,544.5 252.6,537.5 252.6,486.6 252.6,313.6 178.7,313.6 178.7,180.6 252.6,180.6 252.6,0"
          fill="none"
          stroke="currentColor"
          strokeWidth={16}
          strokeLinejoin="round"
        />
      </svg>
    </div>
  )
}

// ──────────────────────────────────────────────────────────────────────
// Drawing pipeline (free functions — no React, no state)
// ──────────────────────────────────────────────────────────────────────

function draw(
  canvas: HTMLCanvasElement | null,
  size: number,
  t: number,
  phase: OrbPhase,
  analyser: AnalyserNode | null,
  freqBufferRef: React.RefObject<ArrayBuffer | null>,
  particles: Particle[],
): void {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const W = size * CANVAS_PADDING
  const H = W
  const cx = W / 2
  const cy = H / 2
  const baseR = size * 0.36
  const colors = ORB_PALETTE[phase]

  ctx.clearRect(0, 0, W, H)

  const audio = sampleAudio(phase, analyser, freqBufferRef)
  const audioReactive = phase === 'listening' || phase === 'playing'

  drawTorus(
    ctx, cx, cy, baseR, colors, t, phase,
    audioReactive ? audio.frequencies : null,
    audioReactive ? audio.average : 0,
    particles,
  )
  drawRimContour(ctx, cx, cy, baseR, colors, t, phase, audioReactive ? audio.frequencies : null, audioReactive ? audio.average : 0, 'outer')
  drawRimContour(ctx, cx, cy, baseR, colors, t, phase, audioReactive ? audio.frequencies : null, audioReactive ? audio.average : 0, 'inner')
  drawShells(ctx, cx, cy, baseR, colors, t, phase)
  if (phase === 'thinking') {
    drawScanArc(ctx, cx, cy, baseR, colors, t)
  }
}

function sampleAudio(
  phase: OrbPhase,
  analyser: AnalyserNode | null,
  freqBufferRef: React.RefObject<ArrayBuffer | null>,
): { average: number; frequencies: Uint8Array | null } {
  const noAudio = !analyser || (phase !== 'listening' && phase !== 'playing')
  if (noAudio || !analyser) return { average: 0, frequencies: null }

  const N = analyser.frequencyBinCount
  if (!freqBufferRef.current || freqBufferRef.current.byteLength !== N) {
    freqBufferRef.current = new ArrayBuffer(N)
  }
  const freqData = new Uint8Array(freqBufferRef.current)
  analyser.getByteFrequencyData(freqData)
  const usable = Math.floor(freqData.length * 0.66)
  let sum = 0
  let max = 0
  for (let i = 0; i < usable; i++) {
    const v = freqData[i]
    sum += v
    if (v > max) max = v
  }
  const avg = sum / usable / 255
  const peak = max / 255
  // 70/30 mix: peak captures transient response, avg captures broadband energy.
  // Subtract a noise floor so room hum + mic-gain noise doesn't drive bulges
  // before the user actually speaks. Empirically 0.08 cuts ambient cleanly
  // and leaves normal speech (~0.15+) reading clearly. Below the floor we
  // pass null frequencies too so per-bin local bulges also disappear.
  const NOISE_FLOOR = 0.08
  const target = peak * 0.7 + avg * 0.3
  if (target < NOISE_FLOOR) return { average: 0, frequencies: null }
  return { average: target - NOISE_FLOOR, frequencies: freqData }
}

function drawTorus(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, baseR: number,
  colors: OrbColors,
  t: number,
  phase: OrbPhase,
  frequencies: Uint8Array | null,
  audioAmp: number,
  particles: Particle[],
): void {
  const rotation = t * 0.35
  const thinkingSpin = phase === 'thinking' ? t * 0.4 : 0
  const tubeR = baseR * TUBE_RATIO
  const cosTilt = Math.cos(TILT)
  const sinTilt = Math.sin(TILT)
  const usable = frequencies ? Math.max(1, Math.floor(frequencies.length * 0.66)) : 0
  const breathStrength = Math.max(0, Math.min(1, (audioAmp - 0.05) * 4))
  const breathPulse = Math.sin(t * 7.0) * breathStrength
  const tubeAmpMul = 1 + Math.min(0.9, audioAmp * 2.0) + breathPulse * 0.40
  const idleSuppress = Math.max(0.55, 1 - audioAmp * 1.6)

  for (const p of particles) {
    const u = p.u + rotation + thinkingSpin
    const wobble =
      idleSuppress *
        (Math.sin(u * 5 + t * 0.85 + p.wavePhase) * 0.65 +
          Math.sin(u * 9 - t * 0.6) * 0.35 +
          Math.sin(u * 13 + t * 1.2) * 0.15) +
      Math.sin(u * 7 + t * 1.9 + p.wavePhase) * (audioAmp * 2.2)
    const v = p.v + t * 0.5 + wobble

    const wave =
      idleSuppress *
      (Math.sin(u * 3 + p.wavePhase + t * 1.5) * 0.045 +
        Math.sin(v * 4 + u * 2 - t * 1.1) * 0.034 +
        Math.sin(u * 7 + t * 0.75) * 0.027 +
        Math.sin(u * 11 + t * 0.5) * 0.018 +
        Math.sin(u * 13 + t * 1.0 + p.wavePhase) * 0.013 +
        Math.sin(t * 0.8 + p.wavePhase) * 0.014)

    let audioDisp = breathPulse * 0.22
    if (frequencies) {
      const idxA = Math.min(usable - 1, Math.floor(Math.abs(Math.sin(u * 1.3 + 0.4)) * usable))
      const idxB = Math.min(usable - 1, Math.floor(Math.abs(Math.cos(u * 2.7 - 0.9)) * usable))
      const localBulge = (frequencies[idxA] * 0.6 + frequencies[idxB] * 0.4) / 255
      audioDisp += audioAmp * 0.32 + localBulge * 0.40
    }

    const R = baseR * (1 + wave + audioDisp)
    const r = tubeR * p.rOffset * tubeAmpMul

    const cosV = Math.cos(v), sinV = Math.sin(v)
    const cosU = Math.cos(u), sinU = Math.sin(u)
    const px = (R + r * cosV) * cosU
    const pyObj = (R + r * cosV) * sinU
    const pzObj = r * sinV

    const py = pyObj * cosTilt - pzObj * sinTilt
    const pz = pyObj * sinTilt + pzObj * cosTilt

    const sx = cx + px
    const sy = cy + py

    const zRange = baseR + tubeR + 1
    const depth = 0.18 + 0.82 * ((pz + zRange) / (2 * zRange))
    const sz = 0.55 + 1.45 * depth
    const alpha = depth * p.brightness

    ctx.fillStyle = withAlpha(colors.glow, alpha * 0.95)
    ctx.beginPath()
    ctx.arc(sx, sy, sz, 0, Math.PI * 2)
    ctx.fill()
  }
}

function drawShells(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, baseR: number,
  colors: OrbColors,
  t: number,
  phase: OrbPhase,
): void {
  const cosTilt = Math.cos(TILT)
  const breath = 0.55 + 0.15 * Math.sin(t * 0.8)
  const baseAlpha = phase === 'idle' ? 0.20 : 0.32
  const SHELLS = [
    { frac: 0.78, width: 1.1, alphaMul: 1.0 },
    { frac: 0.62, width: 0.95, alphaMul: 0.82 },
    { frac: 0.46, width: 0.8, alphaMul: 0.65 },
    { frac: 0.30, width: 0.65, alphaMul: 0.50 },
  ]
  ctx.save()
  for (const s of SHELLS) {
    const rx = baseR * s.frac
    const ry = rx * cosTilt
    ctx.lineWidth = s.width
    ctx.strokeStyle = withAlpha(colors.glow, baseAlpha * s.alphaMul * breath)
    ctx.beginPath()
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2)
    ctx.stroke()
  }
  ctx.restore()
}

function drawRimContour(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, baseR: number,
  colors: OrbColors,
  t: number,
  phase: OrbPhase,
  frequencies: Uint8Array | null,
  audioAmp: number,
  side: 'outer' | 'inner',
): void {
  const tubeR = baseR * TUBE_RATIO
  const cosTilt = Math.cos(TILT)
  const usable = frequencies ? Math.max(1, Math.floor(frequencies.length * 0.66)) : 0
  const idleSuppress = Math.max(0.55, 1 - audioAmp * 1.6)
  const breathStrength = Math.max(0, Math.min(1, (audioAmp - 0.05) * 4))
  const breathPulse = Math.sin(t * 7.0) * breathStrength

  ctx.save()
  ctx.lineWidth = side === 'outer' ? 0.9 : 0.6
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  const baseAlpha = side === 'outer' ? 0.32 : 0.18
  const phaseAlphaMul = phase === 'idle' ? 0.50 : 0.85
  ctx.strokeStyle = withAlpha(colors.glow, baseAlpha * phaseAlphaMul)

  ctx.beginPath()
  const SEGMENTS = 96
  for (let i = 0; i <= SEGMENTS; i++) {
    const u = (i / SEGMENTS) * Math.PI * 2
    const wave =
      idleSuppress *
      (Math.sin(u * 3 + t * 1.5) * 0.045 +
        Math.sin(u * 7 + t * 0.75) * 0.027 +
        Math.sin(u * 11 + t * 0.5) * 0.018 +
        Math.sin(u * 13 + t * 1.0) * 0.013 +
        Math.sin(t * 0.8) * 0.014)
    let audioDisp = breathPulse * 0.22
    if (frequencies) {
      const idxA = Math.min(usable - 1, Math.floor(Math.abs(Math.sin(u * 1.3 + 0.4)) * usable))
      const idxB = Math.min(usable - 1, Math.floor(Math.abs(Math.cos(u * 2.7 - 0.9)) * usable))
      const localBulge = (frequencies[idxA] * 0.6 + frequencies[idxB] * 0.4) / 255
      audioDisp += audioAmp * 0.32 + localBulge * 0.40
    }
    const R = baseR * (1 + wave + audioDisp)
    const radius = side === 'outer' ? R + tubeR : Math.max(0, R - tubeR)
    const px = radius * Math.cos(u)
    const py = radius * Math.sin(u) * cosTilt
    const sx = cx + px
    const sy = cy + py
    if (i === 0) ctx.moveTo(sx, sy)
    else ctx.lineTo(sx, sy)
  }
  ctx.closePath()
  ctx.stroke()
  ctx.restore()
}

function drawScanArc(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number, baseR: number,
  colors: OrbColors,
  t: number,
): void {
  const shellR = baseR * 0.78
  const rx = shellR
  const ry = shellR * Math.cos(TILT)
  const headAngle = (t * 1.6) % (Math.PI * 2)
  const SEGMENTS = 14
  const SEG_SPAN = Math.PI / 9
  const TRAIL_GAP = Math.PI / 320

  ctx.save()
  ctx.lineWidth = 2.2
  ctx.lineCap = 'round'
  for (let i = 0; i < SEGMENTS; i++) {
    const segEnd = headAngle - i * SEG_SPAN
    const segStart = segEnd - (SEG_SPAN - TRAIL_GAP)
    const fade = Math.pow(1 - i / SEGMENTS, 2.2)
    const alpha = fade * 0.95
    if (alpha < 0.02) continue
    ctx.strokeStyle = withAlpha(colors.innerBright, alpha)
    ctx.beginPath()
    ctx.ellipse(cx, cy, rx, ry, 0, segStart, segEnd)
    ctx.stroke()
  }

  const hx = cx + rx * Math.cos(headAngle)
  const hy = cy + ry * Math.sin(headAngle)
  const headR = 6
  const headGrad = ctx.createRadialGradient(hx, hy, 0, hx, hy, headR)
  headGrad.addColorStop(0, colors.core)
  headGrad.addColorStop(0.5, colors.innerBright)
  headGrad.addColorStop(1, withAlpha(colors.glow, 0))
  ctx.fillStyle = headGrad
  ctx.beginPath()
  ctx.arc(hx, hy, headR, 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
}
