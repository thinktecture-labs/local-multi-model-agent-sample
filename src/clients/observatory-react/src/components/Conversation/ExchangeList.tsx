import { useEffect, useRef } from 'react'
import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'
import { ExchangeCard } from './ExchangeCard.tsx'

export function ExchangeList() {
  const { exchanges, activeIdx } = useAppState()
  const dispatch = useAppDispatch()
  const listRef = useRef<HTMLDivElement>(null)
  const prevLengthRef = useRef(exchanges.length)

  // Auto-scroll to bottom when new exchanges are added.
  // Walk up to the closest scrollable ancestor (now .conv-scroll-area in
  // the parent ConversationPane) — exchange-list itself no longer scrolls.
  useEffect(() => {
    if (exchanges.length > prevLengthRef.current && listRef.current) {
      const scroller = listRef.current.closest<HTMLElement>('.conv-scroll-area') ?? listRef.current
      scroller.scrollTo({
        top: scroller.scrollHeight,
        behavior: 'smooth',
      })
    }
    prevLengthRef.current = exchanges.length
  }, [exchanges.length])

  if (exchanges.length === 0) {
    return (
      <div className="exchange-list" ref={listRef}>
        <div className="exchange-empty">
          <div className="icon">{'\u25C8'}</div>
          <div>Ask anything &mdash; the agent routes your query<br/>through five specialized local models</div>
        </div>
      </div>
    )
  }

  return (
    <div className="exchange-list" ref={listRef}>
      {exchanges.map((exchange, idx) => (
        <ExchangeCard
          key={idx}
          exchange={exchange}
          idx={idx}
          isActive={idx === activeIdx}
          onClick={() => {
            dispatch({ type: 'SET_ACTIVE_IDX', idx })
            dispatch({ type: 'SET_TRACE_COLLAPSED', collapsed: false })
          }}
        />
      ))}
    </div>
  )
}
