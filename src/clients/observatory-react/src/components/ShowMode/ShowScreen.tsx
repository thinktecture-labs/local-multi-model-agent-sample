import { useState, useRef, useCallback, useEffect } from 'react'
import { Orb, type OrbPhase } from './Orb.tsx'
import { OrbV2 } from './OrbV2.tsx'
import { ModelStrip } from './ModelStrip.tsx'
import { ResponsePanel } from './ResponsePanel.tsx'
import { voiceChat, queryAgent } from '../../api/client.ts'
import { useWakeWord } from '../../hooks/useWakeWord.ts'
import type { ExecutionStep, StepAction } from '../../types/api.ts'

// OrbV2 (CSS-driven Jarvis-style) is the stage default. Legacy canvas Orb
// kept as a stage escape hatch — opt out via /app?orb-v1.
const USE_ORB_V2 =
  typeof window === 'undefined' ||
  !new URLSearchParams(window.location.search).has('orb-v1')

// Show Mode chips, ordered by stage demo flow:
//   D0 = cold open voice (2 queries) — broad business-analytics setup
//   D1 = Forces teaser (single chip click) — Q3 2024 refrain begins
//   D8 = Final Jarvis voice (Q3 2024 refrain again — bookend sharpens)
//
// Bookend arc: cold open asks 2023 ANNUAL revenue (broad / historical
// context), Final Jarvis asks Q3 2024 SPECIFIC revenue. Same agent,
// narrowing from year to quarter — the audience sees the agent drill
// down and lands on the $84,900 refrain.
const SAMPLE_QUERIES_NEXTERA = [
  { label: 'Top customer',      query: "Who's our top customer?" },
  { label: '2023 revenue',      query: 'What was our revenue in 2023?' },
  { label: 'Q3 revenue',        query: 'What was total revenue in Q3 2024?' },
]

// Spoken on every wakeword/spacebar fire. Synthesised once via Piper on Show
// Mode mount, cached in memory, replayed instantly thereafter.
const JARVIS_GREETING_TEXT = 'All five models online, sir. Ready when you are.'

// STT often mishears the "Hey Jarvis" wakeword phoneme as "Hey, Charles" /
// "Hey, Carlos" / "Hey, Travis" etc. The wakeword detector already confirmed
// the user said "Hey Jarvis", so don't trust STT to re-detect it — normalize
// the leading mishearing so the on-screen Q label stays consistent on stage.
function normalizeWakewordPrefix(text: string): string {
  return text.replace(/^[Hh]ey,?\s+\w+([.,!?…]?)/, 'Hey Jarvis$1')
}

const PHASE_LABELS: Record<OrbPhase, string> = {
  idle:       'Here to help.',
  listening:  'Listening\u2026',
  recording:  'Speak your question\u2026',
  processing: 'Thinking\u2026',
  speaking:   '',
}

interface ShowScreenProps {
  onExit: () => void
}

export function ShowScreen({ onExit }: ShowScreenProps) {
  const [phase, setPhase] = useState<OrbPhase>('idle')
  // Mirror phase into a ref so callbacks (e.g. wake-word detect) can read
  // the current phase synchronously without restaling closures.
  const phaseRef = useRef(phase)
  phaseRef.current = phase
  const [query, setQuery] = useState<string | null>(null)
  const [response, setResponse] = useState<string | null>(null)
  const [steps, setSteps] = useState<ExecutionStep[]>([])
  const [activeAction, setActiveAction] = useState<StepAction | null>(null)
  const [brand, setBrand] = useState('Nextera AI')

  useEffect(() => {
    fetch('/scenario').then(r => r.json()).then(d => {
      setBrand(d.brand)
    }).catch(() => {})
  }, [])

  // Pre-warm the Jarvis greeting: synthesise once via Piper, cache as Blob.
  // Replayed via HTMLAudioElement (separate audio path from the wakeword
  // AudioContext, avoids AudioWorklet contention that caused stuttering).
  useEffect(() => {
    let cancelled = false
    const warm = async () => {
      try {
        const url = `/voice/synthesize?text=${encodeURIComponent(JARVIS_GREETING_TEXT)}&language=en`
        const resp = await fetch(url, { method: 'POST' })
        if (!resp.ok) throw new Error(`synth ${resp.status}`)
        const blob = await resp.blob()
        if (!cancelled) greetingBlobRef.current = blob
      } catch (err) {
        console.warn('Jarvis greeting warmup failed (will retry on first trigger):', err)
      }
    }
    warm()
    return () => { cancelled = true }
  }, [])

  const SAMPLE_QUERIES = SAMPLE_QUERIES_NEXTERA

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const micStreamRef = useRef<MediaStream | null>(null)
  // Analyser is React state (not just a ref) so changes propagate to the orb
  // via re-render. Setting analyserRef.current alone wouldn't trigger one,
  // and the orb's draw loop would keep reading a stale value.
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null)
  // Last spoken answer — bound to the R key for instant replay (re-synth via Piper).
  const lastAnswerTextRef = useRef<string | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null)
  // Currently-playing HTMLAudio element (greeting / text-query / replay paths).
  // Tracked so Space / orb-click during 'speaking' can interrupt cleanly.
  const currentHtmlAudioRef = useRef<HTMLAudioElement | null>(null)
  // Stable reference to the mic analyser so subsequent recordings can re-attach
  // it to the orb even if `ensureMicStream` short-circuits on an already-active
  // stream (without this, the analyser state was nulled at end-of-playback and
  // never restored, so the orb stopped reacting to mic input).
  const micAnalyserRef = useRef<AnalyserNode | null>(null)
  // Cached Blob for the Jarvis greeting — synthesised once on mount, replayed via HTMLAudio.
  // (Switched away from Web Audio AudioBuffer because sharing the AudioContext
  // with the wakeword AudioWorklet caused stuttering playback.)
  const greetingBlobRef = useRef<Blob | null>(null)

  // Wake word "hey jarvis" plays the Jarvis greeting before recording (audible
  // address → audible acknowledgement). Spacebar is a silent failsafe — no
  // greeting, just go straight to recording.
  const triggerListenWithGreetingRef = useRef<() => void>(() => {})
  const triggerListenSilentRef = useRef<() => void>(() => {})
  const { wakeWordState, wakeWordError, startListening, stopListening } = useWakeWord(
    'hey_jarvis',
    () => {
      // Engine stays alive across all phases (see lifecycle effect below) so
      // the mic + AudioWorklet + AGC remain warm — previously each tear-down/
      // restart cold-started the pipeline and the first "Hey Jarvis" after
      // every answer would miss. Gate detections here: only honor wake-word
      // during 'idle' or 'recording'. 'processing'/'speaking' are ignored so
      // Jarvis's own voice (or other noise during TTS) can't retrigger.
      const p = phaseRef.current
      if (p !== 'idle' && p !== 'recording') {
        console.log(`[ShowMode] wake-word detect SUPPRESSED (phase=${p})`)
        return
      }
      console.log(`[ShowMode] wake-word detect ACCEPTED (phase=${p}) → triggerListenWithGreeting`)
      triggerListenWithGreetingRef.current()
    },
  )

  // Trace wakeword engine lifecycle
  useEffect(() => {
    console.log('[ShowMode] wakeWordState=', wakeWordState, wakeWordError ? `error=${wakeWordError}` : '')
  }, [wakeWordState, wakeWordError])

  // Trace phase transitions — useful for correlating with wake-word events.
  useEffect(() => {
    console.log(`[ShowMode] phase → ${phase}`)
  }, [phase])

  // Start the wake-word engine once on mount and keep it alive across phase
  // transitions. The detect callback (above) filters out triggers when we're
  // mid-answer or speaking. This avoids the cold-start path that caused the
  // first "Hey Jarvis" after every answer to miss.
  useEffect(() => {
    startListening()
    return () => { stopListening() }
  }, [startListening, stopListening])

  // ESC to exit · Space toggles listen · R replays the last answer
  // (all skipped when typing in the text input).
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onExit()
        return
      }
      const target = e.target as HTMLElement | null
      const tag = target?.tagName
      const inInput = tag === 'INPUT' || tag === 'TEXTAREA'
      if (inInput) return
      if (e.code === 'Space') {
        e.preventDefault()
        // If something is currently speaking, stop it instead of starting a recording.
        if (stopSpeakingRef.current()) return
        triggerListenSilentRef.current()
        return
      }
      if (e.code === 'KeyR') {
        e.preventDefault()
        replayLastAnswerRef.current()
        return
      }
      if (e.code === 'KeyS') {
        e.preventDefault()
        stopSpeakingRef.current()
        return
      }
      if (e.code === 'KeyC') {
        e.preventDefault()
        clearShowModeRef.current()
        return
      }
      if (e.code === 'KeyJ') {
        // Test trigger — fires the same greeting flow as the wakeword without
        // having to speak "Hey Jarvis". Dev/dress-rehearsal convenience only.
        e.preventDefault()
        triggerListenWithGreetingRef.current()
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [onExit])

  const ensureMicStream = useCallback(async () => {
    if (micStreamRef.current?.active) return micStreamRef.current
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    micStreamRef.current = stream

    // Set up analyser for orb visualization
    const ctx = audioContextRef.current || new AudioContext()
    audioContextRef.current = ctx
    // New AudioContexts start in 'suspended' state until a user gesture.
    // The spacebar/orb-click that triggered this is the gesture — resume now
    // so the analyser actually sees mic data instead of returning zeros.
    if (ctx.state === 'suspended') {
      try { await ctx.resume() } catch { /* ignore */ }
    }
    const source = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 256
    // smoothingTimeConstant is 0.8 by default — reduce so transients (speech
    // onsets) read crisply on the orb's audio-driven bulge instead of being
    // averaged into a sluggish drift.
    analyser.smoothingTimeConstant = 0.5
    source.connect(analyser)
    micAnalyserRef.current = analyser
    setAnalyser(analyser)

    return stream
  }, [])

  const processAudio = useCallback(async (blob: Blob) => {
    setPhase('processing')
    setSteps([])
    setResponse(null)
    setActiveAction(null)

    const collectedSteps: ExecutionStep[] = []

    try {
      await voiceChat(blob, (evType, data) => {
        const d = data as Record<string, unknown>

        if (evType === 'error') {
          // No speech / blank audio — silently reset
          setQuery(null)
          setResponse(null)
          setSteps([])
          setActiveAction(null)
          setPhase('idle')
          return
        }

        if (evType === 'transcription') {
          const raw = (d.text as string) || ''
          const normalized = normalizeWakewordPrefix(raw)
          setQuery(normalized)
          setActiveAction('classify_intent')
          collectedSteps.push({
            action: 'voice_transcribe',
            model: 'whisper',
            duration_ms: (d.duration_ms as number) || 0,
            tokens_used: 0,
            details: { text: normalized, language: d.language },
          })
          setSteps([...collectedSteps])
        }

        if (evType === 'agent_step') {
          const step = d as unknown as ExecutionStep
          collectedSteps.push(step)
          setSteps([...collectedSteps])
          setActiveAction(step.action)
        }

        if (evType === 'response') {
          const text = (d.text as string) || ''
          setResponse(text)
          if (text) lastAnswerTextRef.current = text
          setActiveAction(null)
        }

        if (evType === 'audio') {
          const url = d.url as string
          collectedSteps.push({
            action: 'voice_synthesize',
            model: 'piper',
            duration_ms: (d.duration_ms as number) || 0,
            tokens_used: 0,
            details: {},
          })
          setSteps([...collectedSteps])

          if (url) {
            // Stay in 'processing' until the audio actually starts. Setting
            // 'speaking' before the analyser exists meant the orb re-rendered
            // with a stale analyser and never picked up the playback FFT.
            fetch(url)
              .then(r => r.arrayBuffer())
              .then(buf => {
                const ctx = audioContextRef.current || new AudioContext()
                audioContextRef.current = ctx
                return ctx.decodeAudioData(buf)
              })
              .then(audioBuffer => {
                const ctx = audioContextRef.current!
                if (ctx.state === 'suspended') {
                  ctx.resume().catch(() => {})
                }

                // Set up analyser for speaking visualization (used by both
                // legacy Orb and OrbV2 — both consume AnalyserNode FFT data).
                const playbackAnalyser = ctx.createAnalyser()
                playbackAnalyser.fftSize = 256
                playbackAnalyser.smoothingTimeConstant = 0.5

                const source = ctx.createBufferSource()
                source.buffer = audioBuffer
                source.connect(playbackAnalyser)
                playbackAnalyser.connect(ctx.destination)
                audioSourceRef.current = source
                source.onended = () => {
                  audioSourceRef.current = null
                  setAnalyser(null)
                  setPhase('idle')
                }
                // Set analyser THEN phase — re-render carries fresh analyser.
                setAnalyser(playbackAnalyser)
                setPhase('speaking')
                source.start()
              })
              .catch(() => setPhase('idle'))
          } else {
            setPhase('idle')
          }
        }
      })
    } catch (err) {
      console.error('ShowMode voice error:', err)
      setPhase('idle')
    }

    // If no audio event came, go back to idle
    if (collectedSteps.every(s => s.action !== 'voice_synthesize')) {
      setPhase('idle')
    }
  }, [])

  const startRecording = useCallback(async () => {
    console.log('[ShowMode] startRecording called, phase=', phase)
    if (phase !== 'idle') return
    // Clear any previous result so the orb goes full-size
    setQuery(null)
    setResponse(null)
    setSteps([])
    setActiveAction(null)
    // Wake word stays active during recording (managed by the phase effect) so
    // the user can interrupt with "Hey Jarvis" and trigger the greeting.
    try {
      const stream = await ensureMicStream()
      audioChunksRef.current = []
      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' })
      recorder.ondataavailable = e => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data)
      }
      recorder.onstop = () => {
        if (audioChunksRef.current.length > 0) {
          processAudio(new Blob(audioChunksRef.current, { type: 'audio/webm' }))
        }
      }
      recorder.start()
      mediaRecorderRef.current = recorder
      // Re-attach mic analyser to React state — ensureMicStream short-circuits
      // on subsequent recordings if the stream's already active, so without
      // this the orb keeps the (now-null) playback analyser and stops reacting
      // to mic input. Setting before phase change so the re-render carries it.
      if (micAnalyserRef.current) setAnalyser(micAnalyserRef.current)
      setPhase('recording')
    } catch (err) {
      console.error('Mic error:', err)
    }
  }, [phase, ensureMicStream, processAudio])

  const stopRecording = useCallback(() => {
    if (phase !== 'recording' || !mediaRecorderRef.current) return
    mediaRecorderRef.current.stop()
    // phase transitions to 'processing' in processAudio
  }, [phase])

  // Wakeword path: display + speak the canned Jarvis greeting, then return to
  // idle. Does NOT auto-record. Speaker uses spacebar to ask follow-up questions.
  // Can fire during 'recording' too (user said "Hey Jarvis" mid-spacebar-capture)
  // — in that case, discard the in-progress recording and play greeting cleanly.
  const triggerListenWithGreeting = useCallback(async () => {
    console.log('[ShowMode] triggerListenWithGreeting fired (wakeword path), phase=', phase)
    if (phase !== 'idle' && phase !== 'recording') return

    // If actively recording from spacebar/orb-click, discard it cleanly:
    // detach handlers BEFORE stop() so the trailing ondataavailable + onstop
    // can't fire processAudio on the captured "Hey Jarvis" audio.
    if (phase === 'recording' && mediaRecorderRef.current) {
      const r = mediaRecorderRef.current
      r.ondataavailable = () => { /* discarded */ }
      r.onstop = () => { /* discarded */ }
      try { r.stop() } catch { /* ignore */ }
      mediaRecorderRef.current = null
      audioChunksRef.current = []
    }

    // Wake-word engine stays alive across the greeting — the detect callback
    // (in useWakeWord wiring) gates by phase, so any false-positive during
    // 'speaking' is silently suppressed. Tearing the engine down here was the
    // root cause of the "fires once, then never again" failure mode.

    // Pin the canned Jarvis exchange to the on-screen Q/A. No agent call —
    // the greeting IS the response.
    setQuery('Hey Jarvis!')
    setResponse(JARVIS_GREETING_TEXT)
    setSteps([])
    setActiveAction(null)

    setPhase('speaking')
    try {
      let blob = greetingBlobRef.current
      if (!blob) {
        // Cache miss — synthesise on-the-fly (slower path, ~50–200 ms latency)
        const url = `/voice/synthesize?text=${encodeURIComponent(JARVIS_GREETING_TEXT)}&language=en`
        const resp = await fetch(url, { method: 'POST' })
        if (!resp.ok) throw new Error(`synth ${resp.status}`)
        blob = await resp.blob()
        greetingBlobRef.current = blob
      }

      // Brief pause so the user can finish saying "Hey Jarvis" before the
      // greeting starts — without it the wakeword fires mid-utterance and the
      // greeting overlaps the tail of the trigger phrase, which reads as a
      // hall/echo effect.
      await new Promise(r => setTimeout(r, 350))

      // Play via HTMLAudio routed through Web Audio so the orb sees FFT data
      // for the speaking-state bulge. The wake-word engine has its OWN
      // AudioContext (separate from this one), so the two AudioWorklets don't
      // contend even though the engine keeps running through the greeting.
      const blobUrl = URL.createObjectURL(blob)
      const audio = new Audio(blobUrl)
      currentHtmlAudioRef.current = audio
      try {
        const ctx = audioContextRef.current || new AudioContext()
        audioContextRef.current = ctx
        if (ctx.state === 'suspended') {
          try { await ctx.resume() } catch { /* ignore */ }
        }
        const elementSource = ctx.createMediaElementSource(audio)
        const playbackAnalyser = ctx.createAnalyser()
        playbackAnalyser.fftSize = 256
        playbackAnalyser.smoothingTimeConstant = 0.5
        elementSource.connect(playbackAnalyser)
        playbackAnalyser.connect(ctx.destination)
        setAnalyser(playbackAnalyser)
      } catch (err) {
        console.warn('Greeting analyser tap failed; playing without orb reactivity:', err)
      }
      console.log('[ShowMode] greeting audio.play()')
      try {
        await new Promise<void>((resolve, reject) => {
          audio.onended = () => { console.log('[ShowMode] greeting ended'); resolve() }
          audio.onerror = () => reject(new Error('greeting audio play error'))
          audio.play().catch(reject)
        })
      } finally {
        URL.revokeObjectURL(blobUrl)
        if (currentHtmlAudioRef.current === audio) currentHtmlAudioRef.current = null
        // Restore the mic analyser if we had one, otherwise clear.
        setAnalyser(micAnalyserRef.current ?? null)
      }
    } catch (err) {
      console.warn('Jarvis greeting playback failed:', err)
    }

    // Return to idle — wake-word engine has been alive the whole time, so
    // the next "Hey Jarvis" will be detected immediately.
    setPhase('idle')
  }, [phase])

  // Silent listen path — used by spacebar (no greeting, just record).
  const triggerListenSilent = useCallback(async () => {
    console.log('[ShowMode] triggerListenSilent fired (spacebar path), phase=', phase)
    if (phase === 'recording') {
      stopRecording()
      return
    }
    if (phase !== 'idle') return
    // Engine stays alive — it ran during 'recording' even in the old code, so
    // running it during a spacebar-triggered recording is unchanged behavior.
    startRecording()
  }, [phase, stopRecording, startRecording])

  // Keep refs in sync: wakeword → greeting + listen, spacebar → silent listen
  triggerListenWithGreetingRef.current = triggerListenWithGreeting
  triggerListenSilentRef.current = triggerListenSilent

  // Replay the last spoken answer — re-synth via Piper, play via HTMLAudio.
  const replayLastAnswer = useCallback(async () => {
    if (phase !== 'idle') return
    const text = lastAnswerTextRef.current
    if (!text) return
    setPhase('speaking')
    try {
      const url = `/voice/synthesize?text=${encodeURIComponent(text)}&language=en`
      const resp = await fetch(url, { method: 'POST' })
      if (!resp.ok) throw new Error(`synth ${resp.status}`)
      const blob = await resp.blob()
      const blobUrl = URL.createObjectURL(blob)
      const audio = new Audio(blobUrl)
      currentHtmlAudioRef.current = audio
      try {
        const ctx = audioContextRef.current || new AudioContext()
        audioContextRef.current = ctx
        if (ctx.state === 'suspended') {
          try { await ctx.resume() } catch { /* ignore */ }
        }
        const elementSource = ctx.createMediaElementSource(audio)
        const playbackAnalyser = ctx.createAnalyser()
        playbackAnalyser.fftSize = 256
        playbackAnalyser.smoothingTimeConstant = 0.5
        elementSource.connect(playbackAnalyser)
        playbackAnalyser.connect(ctx.destination)
        setAnalyser(playbackAnalyser)
      } catch { /* fall back to silent reactivity */ }
      try {
        await new Promise<void>((resolve) => {
          audio.onended = () => resolve()
          audio.onerror = () => resolve()
          audio.play().catch(() => resolve())
        })
      } finally {
        URL.revokeObjectURL(blobUrl)
        if (currentHtmlAudioRef.current === audio) currentHtmlAudioRef.current = null
        setAnalyser(micAnalyserRef.current ?? null)
      }
    } catch (err) {
      console.warn('Replay failed:', err)
    }
    setPhase('idle')
  }, [phase])

  const replayLastAnswerRef = useRef<() => void>(() => {})
  replayLastAnswerRef.current = replayLastAnswer

  // Text query (typed or chip click) — uses the REST API, no voice
  const submitTextQuery = useCallback(async (text: string) => {
    if (phase !== 'idle') return
    setPhase('processing')
    setQuery(text)
    setResponse(null)
    setSteps([])
    setActiveAction('classify_intent')

    try {
      const result = await queryAgent(text)
      setSteps(result.steps)
      setResponse(result.response)
      if (result.response) lastAnswerTextRef.current = result.response
      setActiveAction(null)

      // Read the answer aloud via Piper TTS — same path as voice queries.
      // HTMLAudio playback (separate from wakeword AudioContext, no stutter).
      if (result.response) {
        try {
          const ttsUrl = `/voice/synthesize?text=${encodeURIComponent(result.response)}&language=en`
          const resp = await fetch(ttsUrl, { method: 'POST' })
          if (resp.ok) {
            const blob = await resp.blob()
            const blobUrl = URL.createObjectURL(blob)
            const audio = new Audio(blobUrl)
            currentHtmlAudioRef.current = audio
            try {
              const ctx = audioContextRef.current || new AudioContext()
              audioContextRef.current = ctx
              if (ctx.state === 'suspended') {
                try { await ctx.resume() } catch { /* ignore */ }
              }
              const elementSource = ctx.createMediaElementSource(audio)
              const playbackAnalyser = ctx.createAnalyser()
              playbackAnalyser.fftSize = 256
              playbackAnalyser.smoothingTimeConstant = 0.5
              elementSource.connect(playbackAnalyser)
              playbackAnalyser.connect(ctx.destination)
              setAnalyser(playbackAnalyser)
            } catch { /* fall back to silent reactivity */ }
            setPhase('speaking')
            try {
              await new Promise<void>((resolve) => {
                audio.onended = () => resolve()
                audio.onerror = () => resolve()
                audio.play().catch(() => resolve())
              })
            } finally {
              URL.revokeObjectURL(blobUrl)
              if (currentHtmlAudioRef.current === audio) currentHtmlAudioRef.current = null
              setAnalyser(micAnalyserRef.current ?? null)
            }
          }
        } catch (ttsErr) {
          console.warn('TTS readback failed (answer still displayed):', ttsErr)
        }
      }
      setPhase('idle')
    } catch (err) {
      console.error('ShowMode text query error:', err)
      setPhase('idle')
    }
  }, [phase])

  const [textInput, setTextInput] = useState('')
  const [inputVisible, setInputVisible] = useState(false)
  const [lightTheme, setLightTheme] = useState(false)

  const handleTextSubmit = useCallback(() => {
    const trimmed = textInput.trim()
    if (!trimmed) return
    setTextInput('')
    submitTextQuery(trimmed)
  }, [textInput, submitTextQuery])

  // Unified "stop whatever audio is playing" — covers both BufferSource (voice/chat
  // answer) and HTMLAudio (greeting / text-query / replay). Returns true if it
  // actually stopped something, so callers can branch on whether to also do other work.
  const stopSpeaking = useCallback((): boolean => {
    let stopped = false
    if (audioSourceRef.current) {
      try { audioSourceRef.current.stop() } catch { /* ignore */ }
      audioSourceRef.current = null
      stopped = true
    }
    const a = currentHtmlAudioRef.current
    if (a) {
      try { a.pause(); a.currentTime = 0 } catch { /* ignore */ }
      currentHtmlAudioRef.current = null
      stopped = true
    }
    if (stopped && phase === 'speaking') setPhase('idle')
    return stopped
  }, [phase])

  const stopSpeakingRef = useRef<() => boolean>(() => false)
  stopSpeakingRef.current = stopSpeaking

  // Clear / reset — stop any audio, wipe Q/A and steps, return to idle.
  const clearShowMode = useCallback(() => {
    stopSpeaking()
    setQuery(null)
    setResponse(null)
    setSteps([])
    setActiveAction(null)
    setPhase('idle')
  }, [stopSpeaking])

  const clearShowModeRef = useRef<() => void>(() => {})
  clearShowModeRef.current = clearShowMode

  const handleOrbClick = useCallback(() => {
    console.log('[ShowMode] handleOrbClick fired, phase=', phase)
    if (phase === 'speaking') { stopSpeaking(); return }
    if (phase === 'idle') startRecording()
    else if (phase === 'recording') stopRecording()
  }, [phase, startRecording, stopRecording, stopSpeaking])

  return (
    <div className={`show-screen${lightTheme ? ' light' : ''}`}>
      <div className="show-header">
        <span className="show-badge">LOCAL ONLY</span>
        <span className="show-title">{brand}</span>
        <div className="show-header-actions">
          <button
            className="show-exit"
            onClick={clearShowMode}
            title="Clear / Reset (C)"
          >
            {'\u21BA'}
          </button>
          <button
            className="show-exit"
            onClick={() => setLightTheme(v => !v)}
            title="Toggle light/dark"
          >
            {lightTheme ? '\u263E' : '\u2600'}
          </button>
          <button className="show-exit" onClick={onExit} title="Exit (Esc)">
            {'\u2715'}
          </button>
        </div>
      </div>

      <div className={`show-center${response ? ' compact' : ''}`}>
        <div className={`show-orb-wrap${response ? ' compact' : ''}`} onClick={handleOrbClick}>
          {USE_ORB_V2
            ? <OrbV2 phase={phase} analyser={analyser} />
            : <Orb phase={phase} analyser={analyser} />}
        </div>
        <div className={`show-phase-label${response ? ' hidden' : ''}`}>
          {PHASE_LABELS[phase]}
        </div>
      </div>

      <ModelStrip
        steps={steps}
        activeAction={activeAction}
        greetingMode={phase === 'speaking' && steps.length === 0 && response === JARVIS_GREETING_TEXT}
      />

      <ResponsePanel
        query={query}
        response={response}
        steps={steps}
        visible={phase === 'processing' || phase === 'speaking' || phase === 'idle'}
        greetingMode={phase === 'speaking' && steps.length === 0 && response === JARVIS_GREETING_TEXT}
      />

      {/* Sample query chips + text input — hidden by default, toggle with keyboard icon */}
      <div className={`show-input-area${inputVisible ? ' visible' : ''}`}>
        <div className="show-chips">
          {SAMPLE_QUERIES.map(sq => (
            <button
              key={sq.label}
              className="show-chip"
              onClick={() => submitTextQuery(sq.query)}
              disabled={phase !== 'idle'}
              title={sq.query}
            >
              {sq.label}
            </button>
          ))}
        </div>
        <div className="show-text-row">
          <input
            className="show-text-input"
            type="text"
            placeholder="Type a query..."
            value={textInput}
            onChange={e => setTextInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleTextSubmit() }}
            disabled={phase !== 'idle'}
          />
          <button
            className="show-text-send"
            onClick={handleTextSubmit}
            disabled={phase !== 'idle' || !textInput.trim()}
          >
            {'\u25B6'}
          </button>
        </div>
      </div>

      <div className="show-footer">
        <span>0 bytes sent externally</span>
        <span>All models on-device</span>
        <span>Cost: $0.00</span>
        <button
          className="show-kb-toggle"
          onClick={() => setInputVisible(v => !v)}
          title="Toggle text input"
        >
          {'\u2328'}
        </button>
      </div>
    </div>
  )
}
