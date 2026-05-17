import { useState, useEffect } from 'react'
import type { HealthStatus } from '../../types/api.ts'
import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'
import { HealthPills } from './HealthPills.tsx'
import { CostCounter } from './CostCounter.tsx'
import { DemoModeSelector } from './DemoModeSelector.tsx'
import { NetworkToggle } from './NetworkToggle.tsx'
import { RoutingToggle } from './RoutingToggle.tsx'
import { ModelModeToggle } from './ModelModeToggle.tsx'
import { ThemeToggle } from './ThemeToggle.tsx'
import { PrivacyBadge } from './PrivacyBadge.tsx'

/** Default favicon: gradient circle with brand initial. */
function defaultFaviconSvg(letter: string): string {
  return `<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><defs><linearGradient id='g' x1='0%25' y1='0%25' x2='100%25' y2='100%25'><stop offset='0%25' stop-color='%234285f4'/><stop offset='100%25' stop-color='%2334a853'/></linearGradient></defs><circle cx='50' cy='50' r='45' fill='url(%23g)'/><text x='50' y='62' font-size='40' text-anchor='middle' fill='white' font-family='system-ui' font-weight='bold'>${letter}</text></svg>`
}

export function Header({ health, onShowMode }: { health: HealthStatus | null; onShowMode?: () => void }) {
  const { traceCollapsed } = useAppState()
  const dispatch = useAppDispatch()
  const [modelsExpanded, setModelsExpanded] = useState(false)
  const [brand, setBrand] = useState('Nextera')
  const [logoSvg, setLogoSvg] = useState('')

  useEffect(() => {
    fetch('/scenario').then(r => r.json()).then(d => {
      setBrand(d.brand)
      document.title = `${d.brand} — Agent Observatory`
      document.documentElement.setAttribute('data-scenario', d.scenario)

      // Inline logo next to brand (from scenario config, or empty)
      if (d.logo_svg) setLogoSvg(d.logo_svg)

      // Favicon (from scenario config, or default gradient circle)
      const favSvg = d.favicon_svg || defaultFaviconSvg(d.brand.charAt(0))
      const link = document.querySelector("link[rel='icon']") as HTMLLinkElement
      if (link) link.href = `data:image/svg+xml,${favSvg}`
    }).catch(() => {})
  }, [])

  const handleTraceToggle = () => {
    dispatch({ type: 'SET_TRACE_COLLAPSED', collapsed: !traceCollapsed })
  }

  const handleClearHistory = () => {
    dispatch({ type: 'CLEAR_HISTORY', resetEval: true })
  }

  return (
    <header>
      <div className="header-row header-row-top">
        <div className="brand">
          {logoSvg && <span className="brand-logo" dangerouslySetInnerHTML={{ __html: logoSvg }} />}
          {brand} &mdash; <em>Agent</em> Observatory
        </div>
        <CostCounter />
        <DemoModeSelector health={health} />
        <PrivacyBadge />
        <div className="header-spacer"></div>
        <NetworkToggle />
        <RoutingToggle />
        <div className="header-spacer"></div>
        <ModelModeToggle />
        <button className="theme-btn" onClick={handleClearHistory} title="Clear history">
          {'\uD83E\uDDF9'}
        </button>
        <ThemeToggle />
        {onShowMode && (
          <button className="theme-btn show-mode-btn" onClick={onShowMode} title="Show Mode (Ctrl+Shift+P)">
            {'\uD83C\uDFAC'}
          </button>
        )}
        <button className="theme-btn" onClick={handleTraceToggle} title="Toggle trace pane">
          {'\u25E8'}
        </button>
      </div>
      <div className={`header-row header-row-bottom${modelsExpanded ? ' expanded' : ''}`}>
        <button
          className="models-toggle"
          onClick={() => setModelsExpanded(prev => !prev)}
          title={modelsExpanded ? 'Collapse model status' : 'Show model status'}
        >
          <span className="models-toggle-arrow">{modelsExpanded ? '\u25B2' : '\u25BC'}</span>
          <span className="models-toggle-label">Models</span>
        </button>
        {modelsExpanded && <HealthPills health={health} />}
      </div>
    </header>
  )
}
