import { useState, useRef, useCallback, type DragEvent, type KeyboardEvent, type ChangeEvent } from 'react'
import { SuggestionChips } from './SuggestionChips.tsx'
import { ImagePreview } from './ImagePreview.tsx'
import { useVoice, type MicState, type WakeWordState } from '../../hooks/useVoice.ts'

const MIC_LABELS: Record<MicState, string> = {
  idle: 'Record voice',
  recording: 'Stop recording',
  processing: 'Processing\u2026',
}

const WAKE_WORD_LABELS: Record<WakeWordState, string> = {
  off: 'Enable wake word ("Hey Jarvis")',
  loading: 'Initializing wake word\u2026',
  listening: 'Listening for "Hey Jarvis"\u2026 (click to disable)',
  detected: 'Wake word detected!',
  error: 'Wake word error (click to retry)',
}

export function QueryInput({
  onSend,
  onImageFile,
  voiceAvailable,
  activeDocumentName,
  onClearDocument,
}: {
  onSend: (text: string) => void
  onImageFile: (file: File) => void
  voiceAvailable: boolean
  activeDocumentName?: string | null
  onClearDocument?: () => void
}) {
  const [text, setText] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const {
    micState,
    audioUrl,
    toggleRecording,
    stopAudio,
    wakeWordAvailable,
    wakeWordState,
    wakeWordActive,
    toggleWakeWord,
  } = useVoice()

  const handleSend = useCallback(() => {
    const trimmed = text.trim()
    if (!trimmed) return
    onSend(trimmed)
    setText('')
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
    // Re-focus after send
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
    })
  }, [text, onSend])

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setText(e.target.value)
    // Auto-height
    const el = e.target
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = () => {
    setDragOver(false)
  }

  const handleDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const files = e.dataTransfer.files
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      if (file.type.startsWith('image/')) {
        onImageFile(file)
      }
    }
  }

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return
    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      if (file.type.startsWith('image/')) {
        onImageFile(file)
      }
    }
    // Reset so the same file can be selected again
    e.target.value = ''
  }

  return (
    <div
      className={`input-area${dragOver ? ' drag-over' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <SuggestionChips onSend={onSend} />
      {activeDocumentName && (
        <div className="document-chat-badge">
          <span className="doc-badge-icon">{'\uD83D\uDCC4'}</span>
          <span className="doc-badge-text">Chatting with: {activeDocumentName}</span>
          <button
            className="doc-badge-clear"
            onClick={onClearDocument}
            title="Stop chatting with this document"
          >
            {'\u2715'}
          </button>
        </div>
      )}
      <div className="input-row">
        <textarea
          ref={textareaRef}
          className="query-input"
          placeholder={activeDocumentName ? `Ask about ${activeDocumentName}\u2026` : 'Ask a question\u2026'}
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          rows={1}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          multiple
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
        <button
          className="image-upload-btn"
          onClick={() => fileInputRef.current?.click()}
          title="Attach image"
        >
          {'\uD83D\uDCCE'}
        </button>
        {voiceAvailable && (
          <>
            {wakeWordAvailable && (
              <button
                className={`wake-word-btn${wakeWordActive ? ' active' : ''}${wakeWordState === 'detected' ? ' detected' : ''}${wakeWordState === 'loading' ? ' loading' : ''}${wakeWordState === 'error' ? ' error' : ''}`}
                onClick={toggleWakeWord}
                title={WAKE_WORD_LABELS[wakeWordState]}
                disabled={wakeWordState === 'loading'}
              >
                {wakeWordState === 'detected' ? '\u2728' : wakeWordActive ? '\uD83D\uDC42' : '\uD83D\uDCA4'}
              </button>
            )}
            <button
              className={`mic-btn${micState !== 'idle' ? ` ${micState}` : ''}`}
              onClick={toggleRecording}
              title={MIC_LABELS[micState]}
              disabled={micState === 'processing'}
            >
              {micState === 'recording' ? '\u23F9' : '\uD83C\uDF99'}
            </button>
            {audioUrl && (
              <button
                className="mic-btn stop-audio"
                onClick={stopAudio}
                title="Stop playback"
              >
                {'\uD83D\uDD07'}
              </button>
            )}
          </>
        )}
        <button
          className="send-btn"
          onClick={handleSend}
          disabled={!text.trim()}
          title="Send (Enter)"
        >
          {'\u25B6'}
        </button>
      </div>
      <ImagePreview />
    </div>
  )
}
