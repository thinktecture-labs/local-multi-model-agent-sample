import { useState, useEffect } from 'react'
import { SUGGESTIONS } from '../../utils/format.ts'

type SuggestionItem = string | { label: string; query: string; group?: string }

function unpack(s: SuggestionItem): { label: string; query: string; group?: string } {
  return typeof s === 'string' ? { label: s, query: s } : s
}

export function SuggestionChips({ onSend }: { onSend: (text: string) => void }) {
  const [collapsed, setCollapsed] = useState(true)
  const [suggestions, setSuggestions] = useState<SuggestionItem[]>(SUGGESTIONS)

  useEffect(() => {
    fetch('/scenario').then(r => r.json()).then(d => {
      if (d.suggestions?.length) setSuggestions(d.suggestions)
    }).catch(() => {})
  }, [])

  return (
    <div className={`suggestions${collapsed ? ' collapsed' : ''}`}>
      <button
        className="suggestions-toggle"
        onClick={() => setCollapsed(prev => !prev)}
      >
        <span className="suggestions-arrow">{collapsed ? '▶' : '▼'}</span>
        <span className="suggestions-label">Try:</span>
      </button>
      {!collapsed && (
        <div className="suggestions-list">
          {suggestions.map((s) => {
            const item = unpack(s)
            return (
              <button
                key={item.label}
                className="suggestion-chip"
                data-group={item.group}
                onClick={() => onSend(item.query)}
                title={item.query}
              >
                {item.label}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}
