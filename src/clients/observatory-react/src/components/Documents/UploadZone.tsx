import { useState, useRef, useCallback, useEffect } from 'react'
import { uploadDocument, extractData, type ExtractionResult } from '../../api/client.ts'
import { useAppDispatch } from '../../state/AppContext.tsx'
import { formatMs } from '../../utils/format.ts'

interface UploadProgress {
  filename: string
  stage: string
  stageOrder: string[]
  stages: Record<string, 'pending' | 'active' | 'done'>
  stageDetails: Record<string, Record<string, unknown>>
  embedProgress: number
  totalChunks: number | null
  elapsed: number | null
  done: boolean
  documentId: string
  extraction: ExtractionResult | null
  extracting: boolean
}

// Base stages — ocr_extraction is added dynamically only when the server emits it
const BASE_STAGES = ['parsing', 'chunking', 'embedding', 'indexed'] as const
const STAGE_LABELS: Record<string, string> = {
  parsing: 'Parsing',
  ocr_extraction: 'OCR extraction',
  chunking: 'Chunking',
  embedding: 'Embedding',
  indexed: 'Indexed',
}

function formatStageDetail(stage: string, detail: Record<string, unknown>, status: string): string {
  if (status === 'pending') return STAGE_LABELS[stage] ?? stage

  switch (stage) {
    case 'parsing': {
      const bytes = detail.size_bytes as number | undefined
      if (bytes) {
        const kb = bytes / 1024
        return kb > 1024
          ? `Parsing (${(kb / 1024).toFixed(1)} MB)...`
          : `Parsing (${Math.round(kb)} KB)...`
      }
      return 'Parsing document...'
    }
    case 'ocr_extraction': {
      const method = detail.method as string | undefined
      const ocrPages = detail.ocr_pages as number | undefined
      const currentPage = detail.current_page as number | undefined
      const ocrExtracted = detail.ocr_extracted as number | undefined
      const ocrMs = detail.ocr_ms as number | undefined
      if (method === 'pypdf-only') return 'OCR: not needed (pypdf OK)'
      if (method === 'fallback-pypdf') return 'OCR: fallback to pypdf'
      if (currentPage && ocrPages) {
        return `OCR: page ${currentPage}/${ocrPages}${ocrExtracted ? ` (${ocrExtracted} extracted)` : ''}${ocrMs && ocrMs > 100 ? ` ${formatMs(ocrMs)}` : ''}...`
      }
      if (status === 'done' && ocrPages) {
        return `OCR: ${ocrExtracted ?? 0}/${ocrPages} pages${ocrMs ? ` (${formatMs(ocrMs)})` : ''}`
      }
      return 'OCR scanning...'
    }
    case 'chunking': {
      const pages = detail.pages as number | undefined
      const chars = detail.total_chars as number | undefined
      if (pages) {
        return `Chunking ${pages} page${pages > 1 ? 's' : ''}${chars ? `, ${chars.toLocaleString()} chars` : ''}...`
      }
      return 'Smart chunking...'
    }
    case 'embedding': {
      const current = detail.current as number | undefined
      const total = detail.total as number | undefined
      if (current && total) return `Embedding chunk ${current}/${total}...`
      return 'Embedding with embeddinggemma...'
    }
    case 'indexed': {
      const chunks = detail.total_chunks as number | undefined
      const totalMs = detail.total_ms as number | undefined
      const ocrMs = detail.ocr_ms as number | undefined
      let msg = chunks ? `${chunks} chunks indexed` : 'Indexed'
      if (totalMs) msg += ` in ${formatMs(totalMs)}`
      if (ocrMs && ocrMs > 100) msg += ` (OCR: ${formatMs(ocrMs)})`
      return msg
    }
    default:
      return STAGE_LABELS[stage] ?? stage
  }
}

/** Live elapsed timer — ticks every 100ms while any upload is in progress. */
function useLiveTimer(uploads: UploadProgress[], startTimes: React.MutableRefObject<number[]>) {
  const [, setTick] = useState(0)
  const anyActive = uploads.some(u => !u.done)
  useEffect(() => {
    if (!anyActive) return
    const id = setInterval(() => setTick(t => t + 1), 100)
    return () => clearInterval(id)
  }, [anyActive])
  return (idx: number) => {
    const u = uploads[idx]
    if (!u) return '\u2014'
    if (u.done && u.elapsed != null) return formatMs(u.elapsed)
    const start = startTimes.current[idx]
    if (!start) return '\u2014'
    return formatMs(Math.round(performance.now() - start))
  }
}

export function UploadZone() {
  const dispatch = useAppDispatch()
  const [dragActive, setDragActive] = useState(false)
  const [uploads, setUploads] = useState<UploadProgress[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const startTimes = useRef<number[]>([])
  const getElapsed = useLiveTimer(uploads, startTimes)

  const handleExtract = useCallback(async (idx: number) => {
    const u = uploads[idx]
    if (!u || !u.done || u.extracting) return

    setUploads(prev => {
      const updated = [...prev]
      updated[idx] = { ...updated[idx], extracting: true, extraction: null }
      return updated
    })

    try {
      const result = await extractData(u.documentId)
      setUploads(prev => {
        const updated = [...prev]
        updated[idx] = { ...updated[idx], extracting: false, extraction: result }
        return updated
      })
    } catch (err) {
      setUploads(prev => {
        const updated = [...prev]
        updated[idx] = {
          ...updated[idx],
          extracting: false,
          extraction: { success: false, extracted: null, raw_output: null, stored: false, error: String(err), execution_time_ms: 0 },
        }
        return updated
      })
    }
  }, [uploads])

  const handleFile = useCallback(async (file: File) => {
    const allowed = ['.pdf', '.txt', '.md']
    const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
    if (!allowed.includes(ext)) {
      alert('Unsupported format. Use PDF, TXT, or MD.')
      return
    }

    const idx = uploads.length
    const initialOrder = [...BASE_STAGES] as string[]
    const stem = file.name.replace(/\.[^.]+$/, '')
    const documentId = stem.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
    const initial: UploadProgress = {
      filename: file.name,
      stage: 'parsing',
      stageOrder: initialOrder,
      stages: Object.fromEntries(initialOrder.map(s => [s, 'pending' as const])),
      stageDetails: {},
      embedProgress: 0,
      totalChunks: null,
      elapsed: null,
      done: false,
      documentId,
      extraction: null,
      extracting: false,
    }
    setUploads(prev => [...prev, initial])
    const t0 = performance.now()
    startTimes.current[idx] = t0

    try {
      await uploadDocument(file, (event) => {
        setUploads(prev => {
          const updated = [...prev]
          const u = { ...updated[idx] }

          // Dynamically add OCR stage when server emits it
          let order = [...u.stageOrder]
          if (event.stage === 'ocr_extraction' && !order.includes('ocr_extraction')) {
            const parseIdx = order.indexOf('parsing')
            order.splice(parseIdx + 1, 0, 'ocr_extraction')
            u.stageOrder = order
          }

          const currentStageIdx = order.indexOf(event.stage)

          const newStages = { ...u.stages }
          order.forEach(s => { if (!(s in newStages)) newStages[s] = 'pending' })
          order.forEach((s, i) => {
            if (i < currentStageIdx) newStages[s] = 'done'
            else if (i === currentStageIdx) newStages[s] = 'active'
          })
          u.stages = newStages
          u.stage = event.stage

          // Store stage details for rich display
          if (event.detail) {
            u.stageDetails = { ...u.stageDetails, [event.stage]: event.detail as Record<string, unknown> }
          }

          if (event.stage === 'embedding' && event.detail?.progress != null) {
            u.embedProgress = (event.detail.progress as number) * 100
          }

          if (event.stage === 'indexed') {
            order.forEach(s => { u.stages[s] = 'done' })
            u.embedProgress = 100
            u.done = true
            u.elapsed = Math.round(performance.now() - t0)
            u.totalChunks = (event.detail?.total_chunks as number) ?? null
          }

          updated[idx] = u
          return updated
        })
      })

      dispatch({ type: 'CLEAR_HISTORY' })
      dispatch({ type: 'SET_ACTIVE_DOCUMENT', documentId, documentName: file.name })
    } catch (err) {
      console.error('Upload failed:', err)
    }
  }, [uploads.length, dispatch])

  // Paste handler: Cmd-V with text → upload as .txt document
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      // Only handle if no input/textarea is focused (avoid hijacking normal paste)
      const tag = (e.target as HTMLElement)?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea') return

      const text = e.clipboardData?.getData('text/plain')
      if (!text || text.trim().length < 50) return  // ignore tiny pastes

      e.preventDefault()
      const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
      const filename = `paste-${ts}.txt`
      const blob = new Blob([text], { type: 'text/plain' })
      const file = new File([blob], filename, { type: 'text/plain' })
      handleFile(file)
    }
    document.addEventListener('paste', handlePaste)
    return () => document.removeEventListener('paste', handlePaste)
  }, [handleFile])

  return (
    <>
      <div
        className={`doc-drop-zone${dragActive ? ' drag-active' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragActive(true) }}
        onDragLeave={() => setDragActive(false)}
        onDrop={e => {
          e.preventDefault()
          setDragActive(false)
          for (const f of Array.from(e.dataTransfer.files)) handleFile(f)
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md"
          style={{ display: 'none' }}
          onChange={e => {
            const f = e.target.files?.[0]
            if (f) handleFile(f)
            e.target.value = ''
          }}
        />
        <div className="drop-icon">{'\uD83D\uDCC4'}</div>
        <div className="drop-text">Drop a document to index</div>
        <div className="drop-formats">PDF, TXT, MD, or paste text &mdash; never leaves this machine</div>
      </div>

      <div className="doc-uploads-list">
        {uploads.map((u, i) => (
          <div key={i} className={`doc-progress${u.done ? ' indexed' : ''}`}>
            <div className="doc-progress-header">
              <span className="doc-progress-filename">{u.filename}</span>
              <span className="time-badge">{getElapsed(i)}</span>
            </div>
            <div className="doc-progress-steps">
              {u.stageOrder.map(stage => {
                const status = u.stages[stage] ?? 'pending'
                const detail = u.stageDetails[stage] ?? {}
                return (
                  <div key={stage} className={`doc-progress-step${status === 'active' ? ' active' : ''}${status === 'done' ? ' done' : ''}`}>
                    <span className="step-icon">{status === 'done' ? '\u2713' : status === 'active' ? '\u25CF' : '\u25CB'}</span>
                    <span>{formatStageDetail(stage, detail, status)}</span>
                  </div>
                )
              })}
            </div>
            <div className="doc-embed-bar">
              <div className="doc-embed-bar-fill" style={{ width: `${u.embedProgress}%` }} />
            </div>

            {u.done && !u.extraction && (
              <button
                className="extract-btn"
                disabled={u.extracting}
                onClick={(e) => { e.stopPropagation(); handleExtract(i) }}
              >
                {u.extracting ? 'Extracting...' : '\u2728 Extract structured data'}
              </button>
            )}

            {u.extraction && (
              <ExtractionInline result={u.extraction} />
            )}
          </div>
        ))}
      </div>
    </>
  )
}

function ExtractionInline({ result }: { result: ExtractionResult }) {
  const [showRaw, setShowRaw] = useState(false)

  if (!result.success) {
    return (
      <div className="extraction-inline error">
        <span className="extraction-icon">{'\u2717'}</span>
        Extraction failed: {result.error}
      </div>
    )
  }

  const extracted = result.extracted ?? {}
  const fields = Object.entries(extracted).filter(
    ([k]) => !['source_document', 'extracted_at'].includes(k)
  )
  const nonNull = fields.filter(([, v]) => v !== null)

  return (
    <div className="extraction-inline">
      <div className="extraction-header">
        <span className="extraction-icon">{'\u2713'}</span>
        <span>Extracted {nonNull.length} fields</span>
        <span className="extraction-time">{formatMs(result.execution_time_ms)}</span>
        {result.stored && <span className="extraction-stored">stored in DB</span>}
      </div>
      <div className="extraction-fields">
        {fields.map(([k, v]) => (
          <div key={k} className={`extraction-field-row${v === null ? ' null' : ''}`}>
            <span className="extraction-key">{k}</span>
            <span className="extraction-val">
              {v === null ? 'null' : typeof v === 'number' ? v.toLocaleString('en-US') : String(v)}
            </span>
          </div>
        ))}
      </div>
      <button className="raw-toggle" onClick={() => setShowRaw(p => !p)}>
        {showRaw ? '\u25BC' : '\u25B6'} Raw LLM output
      </button>
      {showRaw && result.raw_output && (
        <pre className="extraction-raw">{result.raw_output}</pre>
      )}
    </div>
  )
}
