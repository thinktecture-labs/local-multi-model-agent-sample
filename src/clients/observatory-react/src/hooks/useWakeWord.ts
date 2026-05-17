import { useState, useRef, useCallback, useEffect } from 'react'
import { WakeWordEngine } from 'openwakeword-wasm-browser'

/**
 * OpenWakeWord-based wake word detection hook.
 *
 * Runs an ONNX inference pipeline entirely in-browser via AudioWorklet.
 * No API keys, no vendor lock-in, MIT licensed.
 *
 * ONNX model assets must be served from `{baseAssetUrl}/` (default: /app/openwakeword/models).
 */

export type WakeWordState = 'off' | 'loading' | 'listening' | 'detected' | 'error'

/** Built-in keywords shipped with openwakeword-wasm-browser */
export type OpenWakeWordKeyword =
  | 'hey_jarvis'
  | 'alexa'
  | 'hey_mycroft'
  | 'hey_rhasspy'
  | 'timer'
  | 'weather'

const DEFAULT_KEYWORD: OpenWakeWordKeyword = 'hey_jarvis'
// In dev, Vite serves public/ at the base path (/app/). In prod, FastAPI
// mounts the dist directory at /app/static/. Use import.meta.env to pick.
const BASE_ASSET_URL = import.meta.env.DEV
  ? '/app/openwakeword/models'
  : '/app/static/openwakeword/models'

export function useWakeWord(
  keyword: OpenWakeWordKeyword = DEFAULT_KEYWORD,
  onDetected?: () => void,
) {
  const [wakeWordState, setWakeWordState] = useState<WakeWordState>('off')
  const [error, setError] = useState<string | null>(null)
  const engineRef = useRef<WakeWordEngine | null>(null)
  const onDetectedRef = useRef(onDetected)
  onDetectedRef.current = onDetected
  const unsubRef = useRef<(() => void) | null>(null)

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      unsubRef.current?.()
      engineRef.current?.stop().catch(() => {})
      engineRef.current = null
    }
  }, [])

  const startListening = useCallback(async () => {
    if (engineRef.current) {
      console.log('[WakeWord] startListening() called but engine already running — no-op')
      return
    }

    const t0 = performance.now()
    console.log('[WakeWord] startListening() — loading engine…', { keyword })
    setWakeWordState('loading')
    setError(null)

    try {
      const engine = new WakeWordEngine({
        keywords: [keyword],
        baseAssetUrl: BASE_ASSET_URL,
        detectionThreshold: 0.5,
        cooldownMs: 2000,
      })

      // Subscribe BEFORE start() so we don't miss early events.
      engine.on('ready', () => {
        console.log(`[WakeWord] ready (engine processing audio) · +${Math.round(performance.now() - t0)}ms`)
      })

      await engine.load()
      console.log(`[WakeWord] engine.load() done · +${Math.round(performance.now() - t0)}ms`)

      const unsub = engine.on('detect', ({ keyword: kw, score }: { keyword: string; score: number }) => {
        console.log(`[WakeWord] DETECT "${kw}" score=${score.toFixed(3)} threshold=0.5`)
        setWakeWordState('detected')
        const cb = onDetectedRef.current
        if (cb) {
          cb()
        } else {
          console.warn('[WakeWord] detect fired but no onDetected callback registered')
        }
        setTimeout(() => {
          if (engineRef.current) setWakeWordState('listening')
        }, 1500)
      })
      unsubRef.current = unsub

      engine.on('error', (err: unknown) => {
        console.error('[WakeWord] Engine error:', err)
        setError(String(err))
        setWakeWordState('error')
      })

      await engine.start()
      engineRef.current = engine
      setWakeWordState('listening')
      console.log(`[WakeWord] engine.start() done · listening · +${Math.round(performance.now() - t0)}ms`)
    } catch (err) {
      console.error('[WakeWord] Init failed:', err)
      setError(err instanceof Error ? err.message : String(err))
      setWakeWordState('error')
      engineRef.current = null
    }
  }, [keyword])

  const stopListening = useCallback(async () => {
    if (!engineRef.current) return
    try {
      unsubRef.current?.()
      unsubRef.current = null
      await engineRef.current.stop()
    } catch {
      // ignore cleanup errors
    }
    engineRef.current = null
    setWakeWordState('off')
    setError(null)
  }, [])

  const toggleListening = useCallback(async () => {
    if (wakeWordState === 'listening' || wakeWordState === 'detected') {
      await stopListening()
    } else {
      await startListening()
    }
  }, [wakeWordState, startListening, stopListening])

  return {
    wakeWordState,
    wakeWordError: error,
    startListening,
    stopListening,
    toggleListening,
    isListening: wakeWordState === 'listening' || wakeWordState === 'detected',
  }
}
