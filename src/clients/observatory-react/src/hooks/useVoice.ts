import { useState, useRef, useCallback } from 'react'
import { voiceChat } from '../api/client.ts'
import { useAppState, useAppDispatch } from '../state/AppContext.tsx'
import type { ExecutionStep } from '../types/api.ts'
import { useWakeWord, type WakeWordState } from './useWakeWord.ts'

export type MicState = 'idle' | 'recording' | 'processing'

export function useVoice() {
  const state = useAppState()
  const dispatch = useAppDispatch()

  const [micState, setMicState] = useState<MicState>('idle')
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const micStreamRef = useRef<MediaStream | null>(null)
  const audioSourceRef = useRef<AudioBufferSourceNode | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)

  const ensureMicStream = useCallback(async () => {
    if (micStreamRef.current?.active) return micStreamRef.current
    micStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true })
    return micStreamRef.current
  }, [])

  const processAudio = useCallback(async (blob: Blob) => {
    const idx = state.exchanges.length
    dispatch({
      type: 'ADD_EXCHANGE',
      exchange: { query: '(voice...)', images: [], imageDataUrls: [], result: null, voiceMode: true },
    })
    dispatch({ type: 'SET_ACTIVE_IDX', idx })
    dispatch({ type: 'SET_LOADING', loading: true })
    dispatch({ type: 'SET_TRACE_COLLAPSED', collapsed: false })

    const agentSteps: ExecutionStep[] = []
    let responseText = ''

    try {
      await voiceChat(blob, (evType, data) => {
        const d = data as Record<string, unknown>

        if (evType === 'transcription') {
          const sttMs = (d.duration_ms as number) || 0
          agentSteps.unshift({
            action: 'voice_transcribe',
            model: 'whisper',
            duration_ms: sttMs,
            tokens_used: 0,
            details: { text: d.text, language: d.language },
          })
          dispatch({
            type: 'UPDATE_EXCHANGE',
            idx,
            updates: { query: d.text as string },
          })
        }

        if (evType === 'agent_step') {
          agentSteps.push(d as unknown as ExecutionStep)
        }

        if (evType === 'response') {
          responseText = (d.text as string) || ''
          dispatch({
            type: 'UPDATE_EXCHANGE',
            idx,
            updates: {
              result: {
                intent: (d.intent as string) || 'voice',
                response: responseText,
                execution_time_ms: (d.duration_ms as number) || 0,
                steps: [...agentSteps],
                models_used: [...new Set(agentSteps.map(s => s.model))],
                total_tokens: agentSteps.reduce((a, s) => a + (s.tokens_used || 0), 0),
              } as never,
            },
          })
        }

        if (evType === 'audio') {
          const url = d.url as string
          setAudioUrl(url)
          agentSteps.push({
            action: 'voice_synthesize',
            model: 'piper',
            duration_ms: (d.duration_ms as number) || 0,
            tokens_used: 0,
            details: { language: 'en' },
          })

          // Play audio
          if (url) {
            fetch(url)
              .then(r => r.arrayBuffer())
              .then(buf => {
                const ctx = audioContextRef.current || new AudioContext()
                audioContextRef.current = ctx
                return ctx.decodeAudioData(buf)
              })
              .then(audioBuffer => {
                const ctx = audioContextRef.current!
                const source = ctx.createBufferSource()
                source.buffer = audioBuffer
                source.connect(ctx.destination)
                source.start()
                audioSourceRef.current = source
                source.onended = () => {
                  audioSourceRef.current = null
                  setAudioUrl(null)
                }
              })
              .catch(console.error)
          }
        }
      })
    } catch (err) {
      console.error('Voice chat failed:', err)
    } finally {
      dispatch({ type: 'SET_LOADING', loading: false })
      setMicState('idle')
    }
  }, [state.exchanges.length, dispatch])

  const startRecording = useCallback(async () => {
    if (micState !== 'idle') return
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
      setMicState('recording')
    } catch (err) {
      console.error('Microphone access denied:', err)
    }
  }, [micState, ensureMicStream, processAudio])

  const stopRecording = useCallback(() => {
    if (micState !== 'recording' || !mediaRecorderRef.current) return
    mediaRecorderRef.current.stop()
    setMicState('processing')
  }, [micState])

  const toggleRecording = useCallback(() => {
    if (micState === 'recording') stopRecording()
    else startRecording()
  }, [micState, startRecording, stopRecording])

  const stopAudio = useCallback(() => {
    audioSourceRef.current?.stop()
    audioSourceRef.current = null
    setAudioUrl(null)
  }, [])

  // Wake word: when detected, auto-start recording
  const onWakeWordDetected = useCallback(() => {
    if (micState === 'idle') {
      startRecording()
    }
  }, [micState, startRecording])

  const {
    wakeWordState,
    wakeWordError,
    toggleListening: toggleWakeWord,
    isListening: wakeWordActive,
  } = useWakeWord('hey_jarvis', onWakeWordDetected)

  return {
    micState,
    audioUrl,
    toggleRecording,
    stopAudio,
    // Wake word — always available (no API key needed)
    wakeWordAvailable: true,
    wakeWordState,
    wakeWordError,
    wakeWordActive,
    toggleWakeWord,
  }
}

export type { WakeWordState }
