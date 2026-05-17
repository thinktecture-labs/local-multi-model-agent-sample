import { useCallback } from 'react'
import { flushSync } from 'react-dom'
import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'
import { queryAgentStream, compareQuery } from '../../api/client.ts'
import { ExchangeList } from './ExchangeList.tsx'
import { QueryInput } from '../Input/QueryInput.tsx'
import { UploadZone } from '../Documents/UploadZone.tsx'
import type { Exchange } from '../../types/state.ts'
import type { HealthStatus, QueryResult, ThreePathResult } from '../../types/api.ts'

interface ConversationPaneProps {
  health: HealthStatus | null
}

export function ConversationPane({ health }: ConversationPaneProps) {
  const state = useAppState()
  const dispatch = useAppDispatch()

  const whisperAvailable = !!health?.models?.['WHISPER']

  const handleSend = useCallback(
    async (text: string) => {
      const images = [...state.pendingImages]
      const imageDataUrls = [...state.pendingImageDataUrls]

      const exchange: Exchange = {
        query: text,
        images,
        imageDataUrls,
        result: null,
      }

      const idx = state.exchanges.length
      dispatch({ type: 'ADD_EXCHANGE', exchange })
      dispatch({ type: 'SET_LOADING', loading: true })
      dispatch({ type: 'SET_ACTIVE_IDX', idx })
      dispatch({ type: 'SET_TRACE_COLLAPSED', collapsed: false })
      dispatch({ type: 'CLEAR_IMAGES' })

      try {
        const { demoMode } = state

        if (demoMode === 'all') {
          // Progressive three-path: run all backends via streaming in parallel,
          // update each column as it completes (fastest appears first)
          const threePathResult: ThreePathResult = { multi_models: null as unknown as QueryResult, qwen: null, cloud: null }
          dispatch({
            type: 'UPDATE_EXCHANGE', idx,
            updates: {
              threePathResult: { ...threePathResult },
              threePathStreaming: { multi_models: '', qwen: '', cloud: '' },
            },
          })

          const runBackend = (backend: 'multi-models' | 'qwen' | 'cloud', key: keyof ThreePathResult) => {
            let accText = ''
            let accSteps: QueryResult['steps'] = []
            const colKey = key as 'multi_models' | 'qwen' | 'cloud'

            return queryAgentStream(text, {
              onStep: (step) => { accSteps.push(step) },
              onToken: (token) => {
                accText += token
                dispatch({ type: 'APPEND_COLUMN_TOKEN', idx, key: colKey, text: token })
              },
              onDone: (meta) => {
                const result: QueryResult = {
                  intent: meta.intent as QueryResult['intent'],
                  response: accText,
                  execution_time_ms: meta.execution_time_ms,
                  steps: accSteps,
                  models_used: meta.models_used,
                  total_tokens: meta.total_tokens,
                  prompt_tokens: meta.prompt_tokens,
                  completion_tokens: meta.completion_tokens,
                  cloud_cost: meta.cloud_cost ?? undefined,
                }
                threePathResult[key] = result
                dispatch({
                  type: 'UPDATE_EXCHANGE', idx,
                  updates: { threePathResult: { ...threePathResult } },
                })
                if (key === 'multi_models') {
                  dispatch({ type: 'UPDATE_EXCHANGE', idx, updates: { result } })
                  dispatch({
                    type: 'UPDATE_TOKENS',
                    prompt: meta.prompt_tokens ?? 0,
                    completion: meta.completion_tokens ?? 0,
                    total: meta.total_tokens,
                  })
                }
              },
              onError: () => {
                // Clear streaming text so the column shows "Not available" instead of dots
                dispatch({ type: 'CLEAR_COLUMN_STREAMING', idx, key: key as 'multi_models' | 'qwen' | 'cloud' })
              },
            }, images.length ? images : undefined, backend)
          }

          await Promise.all([
            runBackend('multi-models', 'multi_models'),
            runBackend('qwen', 'qwen').catch(() => {}),
            runBackend('cloud', 'cloud').catch(() => {}),
          ])
        } else if (state.compareMode) {
          const compareResult = await compareQuery(text, images.length ? images : undefined)
          dispatch({
            type: 'UPDATE_EXCHANGE',
            idx,
            updates: {
              result: {
                intent: compareResult.intent,
                response: compareResult.local_response,
                execution_time_ms: compareResult.execution_time_ms,
                steps: compareResult.steps,
                models_used: [],
                total_tokens: compareResult.total_tokens,
                prompt_tokens: compareResult.prompt_tokens,
                completion_tokens: compareResult.completion_tokens,
              },
              compareResult,
            },
          })
          dispatch({
            type: 'UPDATE_TOKENS',
            prompt: compareResult.prompt_tokens ?? 0,
            completion: compareResult.completion_tokens ?? 0,
            total: compareResult.total_tokens,
          })
        } else if (demoMode === 'multi-models' || demoMode === 'cloud' || demoMode === 'qwen') {
          // Streaming path for all single-backend modes
          dispatch({
            type: 'UPDATE_EXCHANGE', idx,
            updates: {
              result: { intent: 'direct_answer' as const, response: '', execution_time_ms: 0, steps: [], models_used: [], total_tokens: 0 },
              // Don't set streamingText yet — keep typing dots visible during step processing.
              // APPEND_TOKEN will initialize streamingText on first token arrival.
              streamingText: undefined,
            },
          })
          let firstToken = true
          await queryAgentStream(text, {
            onStep: (step) => dispatch({ type: 'APPEND_STEP', idx, step }),
            onToken: (token) => flushSync(() => {
              if (firstToken) {
                firstToken = false
                // Initialize streamingText on first token — transitions from typing dots to streaming text
                dispatch({ type: 'UPDATE_EXCHANGE', idx, updates: { streamingText: '' } })
              }
              dispatch({ type: 'APPEND_TOKEN', idx, text: token })
            }),
            onDone: (meta) => {
              dispatch({ type: 'FINALIZE_STREAM', idx, meta })
              dispatch({
                type: 'UPDATE_TOKENS',
                prompt: meta.prompt_tokens ?? 0,
                completion: meta.completion_tokens ?? 0,
                total: meta.total_tokens,
              })
            },
            onError: (msg) => {
              dispatch({
                type: 'UPDATE_EXCHANGE', idx,
                updates: {
                  streamingText: undefined,
                  result: { intent: 'direct_answer' as const, response: `Error: ${msg}`, execution_time_ms: 0, steps: [], models_used: [], total_tokens: 0 },
                },
              })
            },
          }, images.length ? images : undefined, demoMode, state.activeDocumentId)
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : 'Unknown error'
        dispatch({
          type: 'UPDATE_EXCHANGE',
          idx,
          updates: {
            streamingText: undefined,
            result: {
              intent: 'direct_answer',
              response: `Error: ${errorMsg}`,
              execution_time_ms: 0,
              steps: [],
              models_used: [],
              total_tokens: 0,
            },
          },
        })
      } finally {
        // Safety net: ensure streamingText is always cleared (prevents stuck cursor)
        dispatch({ type: 'UPDATE_EXCHANGE', idx, updates: { streamingText: undefined } })
        dispatch({ type: 'SET_LOADING', loading: false })
      }
    },
    [state.pendingImages, state.pendingImageDataUrls, state.exchanges.length, state.compareMode, state.demoMode, state.activeDocumentId, dispatch],
  )

  const handleImageFile = useCallback(
    (file: File) => {
      const reader = new FileReader()
      reader.onload = () => {
        const dataUrl = reader.result as string
        const base64 = dataUrl.split(',')[1] ?? ''
        dispatch({ type: 'ADD_PENDING_IMAGE', base64, dataUrl })
      }
      reader.readAsDataURL(file)
    },
    [dispatch],
  )

  return (
    <section className="conv-pane">
      <div className="conv-scroll-area">
        <UploadZone />
        <ExchangeList />
      </div>
      <QueryInput
        onSend={handleSend}
        onImageFile={handleImageFile}
        voiceAvailable={whisperAvailable}
        activeDocumentName={state.activeDocumentName}
        onClearDocument={() => dispatch({ type: 'CLEAR_ACTIVE_DOCUMENT' })}
      />
    </section>
  )
}
