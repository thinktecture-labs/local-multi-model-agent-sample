import { useEffect, useRef } from 'react'
import type { ExecutionStep } from '../../types/api.ts'

type CardType = 'text' | 'kpi' | 'bar-chart' | 'ranked-bars' | 'table'

interface SmartCardProps {
  response: string
  steps: ExecutionStep[]
}

interface SqlRow {
  [key: string]: unknown
}

function getSqlRows(steps: ExecutionStep[]): SqlRow[] | null {
  const toolStep = steps.find(s => s.action === 'execute_tool')
  if (!toolStep) return null
  const rows = (toolStep.details?.result as { rows?: SqlRow[] })?.rows
  if (!rows || !Array.isArray(rows) || rows.length === 0) return null
  return rows
}

function detectCardType(steps: ExecutionStep[]): CardType {
  const rows = getSqlRows(steps)
  if (!rows) return 'text'

  const cols = Object.keys(rows[0])

  // Single value -> KPI card
  if (rows.length === 1 && cols.length <= 2) return 'kpi'

  // Time series (has quarter/month/year column) -> bar chart
  if (cols.some(c => /quarter|q[1-4]|month|year|period/i.test(c))) return 'bar-chart'

  // Ranked list (has name + numeric column, <=10 rows) -> horizontal bars
  if (rows.length <= 10 && cols.some(c => /name|customer|tier|product|plan/i.test(c))) return 'ranked-bars'

  // Fallback -> table
  return 'table'
}

function KpiCard({ rows }: { rows: SqlRow[] }) {
  const row = rows[0]
  const cols = Object.keys(row)
  const numCol = cols.find(c => typeof row[c] === 'number') || cols[cols.length - 1]
  const labelCol = cols.find(c => c !== numCol) || null
  const value = row[numCol]
  const label = labelCol ? String(row[labelCol]) : numCol
  const counterRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (!counterRef.current || typeof value !== 'number') return
    const target = value
    const duration = 800
    const start = performance.now()
    const tick = (now: number) => {
      const elapsed = now - start
      const pct = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - pct, 3) // ease-out cubic
      const current = target * eased
      if (counterRef.current) {
        counterRef.current.textContent = current >= 1000
          ? `$${(current).toLocaleString(undefined, { maximumFractionDigits: 0 })}`
          : current.toLocaleString(undefined, { maximumFractionDigits: 1 })
      }
      if (pct < 1) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
  }, [value])

  return (
    <div className="sc-kpi">
      <span ref={counterRef} className="sc-kpi-value">
        {typeof value === 'number' ? value.toLocaleString() : String(value)}
      </span>
      <span className="sc-kpi-label">{String(label).replace(/_/g, ' ')}</span>
    </div>
  )
}

function BarChart({ rows }: { rows: SqlRow[] }) {
  const cols = Object.keys(rows[0])
  const labelCol = cols.find(c => /quarter|q[1-4]|month|year|period|name/i.test(c)) || cols[0]
  const valueCol = cols.find(c => c !== labelCol && typeof rows[0][c] === 'number') || cols[1]
  const max = Math.max(...rows.map(r => Number(r[valueCol]) || 0))

  return (
    <div className="sc-bars">
      {rows.map((row, i) => {
        const val = Number(row[valueCol]) || 0
        const pct = max > 0 ? (val / max) * 100 : 0
        return (
          <div key={i} className="sc-bar-row" style={{ animationDelay: `${i * 80}ms` }}>
            <span className="sc-bar-label">{String(row[labelCol])}</span>
            <div className="sc-bar-track">
              <div className="sc-bar-fill" style={{ '--target-width': `${pct}%` } as React.CSSProperties} />
            </div>
            <span className="sc-bar-value">
              {val >= 1000 ? `$${val.toLocaleString()}` : val.toLocaleString()}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function RankedBars({ rows }: { rows: SqlRow[] }) {
  // Same as BarChart but horizontal emphasis on ranking
  return <BarChart rows={rows} />
}

function StyledTable({ rows }: { rows: SqlRow[] }) {
  const cols = Object.keys(rows[0])
  return (
    <div className="sc-table-wrap">
      <table className="sc-table">
        <thead>
          <tr>{cols.map(c => <th key={c}>{c.replace(/_/g, ' ')}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ animationDelay: `${i * 50}ms` }}>
              {cols.map(c => (
                <td key={c}>
                  {typeof row[c] === 'number'
                    ? (row[c] as number).toLocaleString()
                    : String(row[c] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

import { marked } from 'marked'

// Configure marked for clean output
marked.setOptions({ breaks: true, gfm: true })

export function SmartCard({ response, steps }: SmartCardProps) {
  const cardType = detectCardType(steps)
  const rows = getSqlRows(steps)

  return (
    <div className="sc-container">
      {cardType !== 'text' && rows && (
        <div className="sc-card">
          {cardType === 'kpi' && <KpiCard rows={rows} />}
          {cardType === 'bar-chart' && <BarChart rows={rows} />}
          {cardType === 'ranked-bars' && <RankedBars rows={rows} />}
          {cardType === 'table' && <StyledTable rows={rows} />}
        </div>
      )}
      <div
        className="sc-text"
        dangerouslySetInnerHTML={{ __html: marked.parse(response) as string }}
      />
    </div>
  )
}
