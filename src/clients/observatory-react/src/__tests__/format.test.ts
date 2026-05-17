import {
  formatMs,
  escapeHtml,
  intentLabel,
  intentBadgeLabel,
  shortModel,
  modelKey,
  modelColor,
  ACTION_LABEL,
  SUGGESTIONS,
  formatResponse,
  formatBytes,
} from '../utils/format.ts'

describe('formatMs', () => {
  it('returns -- for null', () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(formatMs(null as any)).toBe('--')
  })

  it('returns -- for undefined', () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expect(formatMs(undefined as any)).toBe('--')
  })

  it('returns milliseconds for values < 1000', () => {
    expect(formatMs(0)).toBe('0 ms')
    expect(formatMs(42)).toBe('42 ms')
    expect(formatMs(999)).toBe('999 ms')
  })

  it('returns seconds with two decimals for values >= 1000', () => {
    expect(formatMs(1000)).toBe('1.00 s')
    expect(formatMs(1500)).toBe('1.50 s')
    expect(formatMs(2345)).toBe('2.35 s')
  })

  it('rounds milliseconds to nearest integer', () => {
    expect(formatMs(42.7)).toBe('43 ms')
    expect(formatMs(42.3)).toBe('42 ms')
  })
})

describe('escapeHtml', () => {
  it('escapes ampersands', () => {
    expect(escapeHtml('a & b')).toBe('a &amp; b')
  })

  it('escapes angle brackets', () => {
    expect(escapeHtml('<div>')).toBe('&lt;div&gt;')
  })

  it('escapes double quotes', () => {
    expect(escapeHtml('"hello"')).toBe('&quot;hello&quot;')
  })

  it('returns empty string for null', () => {
    expect(escapeHtml(null)).toBe('')
  })

  it('returns empty string for undefined', () => {
    expect(escapeHtml(undefined)).toBe('')
  })

  it('handles combined special characters', () => {
    expect(escapeHtml('<a href="x">&')).toBe('&lt;a href=&quot;x&quot;&gt;&amp;')
  })
})

describe('intentLabel', () => {
  it('maps rag_query', () => {
    expect(intentLabel('rag_query')).toBe('RAG Query')
  })

  it('maps tool_use', () => {
    expect(intentLabel('tool_use')).toBe('Tool Use')
  })

  it('maps direct_answer', () => {
    expect(intentLabel('direct_answer')).toBe('Direct Answer')
  })

  it('maps image_query', () => {
    expect(intentLabel('image_query')).toBe('Image Query')
  })

  it('maps voice', () => {
    expect(intentLabel('voice')).toBe('Voice')
  })

  it('returns raw string for unknown intent', () => {
    expect(intentLabel('some_unknown')).toBe('some_unknown')
  })
})

describe('intentBadgeLabel', () => {
  it('maps rag_query to RAG', () => {
    expect(intentBadgeLabel('rag_query')).toBe('RAG')
  })

  it('maps tool_use to TOOL', () => {
    expect(intentBadgeLabel('tool_use')).toBe('TOOL')
  })

  it('maps direct_answer to DIRECT', () => {
    expect(intentBadgeLabel('direct_answer')).toBe('DIRECT')
  })

  it('maps image_query to IMAGE', () => {
    expect(intentBadgeLabel('image_query')).toBe('IMAGE')
  })

  it('maps voice to VOICE', () => {
    expect(intentBadgeLabel('voice')).toBe('VOICE')
  })

  it('returns raw string for unknown', () => {
    expect(intentBadgeLabel('unknown')).toBe('unknown')
  })
})

describe('shortModel', () => {
  it('keeps full model name, only strips -ft and -merged suffixes', () => {
    expect(shortModel('gemma3-1B')).toBe('gemma3-1B')
    expect(shortModel('gemma3-4b-vision')).toBe('gemma3-4b-vision')
    expect(shortModel('qwen3.5-4b')).toBe('qwen3.5-4b')
    expect(shortModel('embeddinggemma-308M')).toBe('embeddinggemma-308M')
    expect(shortModel('whisper-large-v3')).toBe('whisper-large-v3')
    expect(shortModel('piper-tts')).toBe('piper-tts')
  })

  it('strips -ft and -merged suffixes', () => {
    expect(shortModel('gemma3-1B-ft')).toBe('gemma3-1B')
    expect(shortModel('qwen3.5-4b-merged')).toBe('qwen3.5-4b')
  })

  it('maps local_execution to local', () => {
    expect(shortModel('local_execution')).toBe('local')
  })

  it('maps local to local', () => {
    expect(shortModel('local')).toBe('local')
  })

  it('returns local for empty input', () => {
    expect(shortModel('')).toBe('local')
  })

  it('returns original for unknown model', () => {
    expect(shortModel('gpt-4')).toBe('gpt-4')
  })
})

describe('modelKey', () => {
  it('maps gemma3 variants to gemma3', () => {
    expect(modelKey('gemma3-1B')).toBe('gemma3')
    expect(modelKey('gemma3-4b-vision')).toBe('gemma3')
  })

  it('maps qwen variants', () => {
    expect(modelKey('qwen3.5-4b')).toBe('qwen')
    expect(modelKey('qwen3.5-35b-a3b')).toBe('qwen')
  })

  it('maps embeddinggemma variants', () => {
    expect(modelKey('embeddinggemma-308M')).toBe('embeddinggemma')
  })

  it('maps whisper and piper', () => {
    expect(modelKey('whisper')).toBe('whisper')
    expect(modelKey('piper')).toBe('piper')
  })

  it('maps cloud models', () => {
    expect(modelKey('gpt-4')).toBe('cloud')
    expect(modelKey('claude-3')).toBe('cloud')
  })

  it('maps local and unknown to local', () => {
    expect(modelKey('local')).toBe('local')
    expect(modelKey('heuristic')).toBe('local')
    expect(modelKey('')).toBe('local')
  })
})

describe('modelColor', () => {
  it('returns correct CSS var for gemma3', () => {
    expect(modelColor('gemma3-1B')).toBe('var(--c-gemma3)')
  })

  it('returns correct CSS var for qwen', () => {
    expect(modelColor('qwen3.5-4b')).toBe('var(--c-qwen)')
  })

  it('returns correct CSS var for embeddinggemma', () => {
    expect(modelColor('embeddinggemma')).toBe('var(--c-embedding)')
  })

  it('returns correct CSS var for local', () => {
    expect(modelColor('local')).toBe('var(--c-local)')
  })

  it('returns hex color for whisper', () => {
    expect(modelColor('whisper')).toBe('#e67e22')
  })

  it('returns hex color for piper', () => {
    expect(modelColor('piper')).toBe('#9b59b6')
  })

  it('returns cloud color for gpt/claude models', () => {
    expect(modelColor('gpt-4')).toBe('var(--c-cloud)')
    expect(modelColor('claude-3')).toBe('var(--c-cloud)')
  })

  it('returns fallback for unknown model', () => {
    expect(modelColor('some-random-model')).toBe('var(--c-local)')
  })
})

describe('ACTION_LABEL', () => {
  it('has entries for all known actions', () => {
    const expectedActions = [
      'classify_intent',
      'rewrite_query',
      'vector_search',
      'synthesize_response',
      'select_tool',
      'execute_tool',
      'format_response',
      'direct_response',
      'analyse_image',
      'confidence_assessment',
      'cloud_escalation',
      'decompose_query',
      'concretize_step',
      'voice_transcribe',
      'voice_synthesize',
    ]
    for (const action of expectedActions) {
      expect(ACTION_LABEL[action]).toBeDefined()
      expect(typeof ACTION_LABEL[action]).toBe('string')
    }
  })
})

describe('SUGGESTIONS', () => {
  it('is a non-empty array of strings', () => {
    expect(Array.isArray(SUGGESTIONS)).toBe(true)
    expect(SUGGESTIONS.length).toBeGreaterThan(0)
    for (const s of SUGGESTIONS) {
      expect(typeof s).toBe('string')
    }
  })
})

describe('formatResponse', () => {
  it('wraps text in <p> tags', () => {
    expect(formatResponse('hello')).toBe('<p>hello</p>')
  })

  it('replaces double newlines with paragraph breaks', () => {
    expect(formatResponse('a\n\nb')).toBe('<p>a</p><p>b</p>')
  })

  it('replaces single newlines with <br>', () => {
    expect(formatResponse('a\nb')).toBe('<p>a<br>b</p>')
  })

  it('returns empty string for empty input', () => {
    expect(formatResponse('')).toBe('')
  })
})

describe('formatBytes', () => {
  it('returns 0 bytes for zero', () => {
    expect(formatBytes(0)).toBe('0 bytes')
  })

  it('returns bytes with B suffix for < 1024', () => {
    expect(formatBytes(512)).toBe('512 B')
  })

  it('returns KB for >= 1024', () => {
    expect(formatBytes(1024)).toBe('1.0 KB')
    expect(formatBytes(2560)).toBe('2.5 KB')
  })
})
