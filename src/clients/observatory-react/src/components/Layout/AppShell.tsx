import type { ReactNode } from 'react'
import { useAppState } from '../../state/AppContext.tsx'

interface AppShellProps {
  children: ReactNode
}

export function AppShell({ children }: AppShellProps) {
  const { traceCollapsed } = useAppState()

  return (
    <div className="app">
      <main className={traceCollapsed ? 'trace-collapsed' : ''}>
        {children}
      </main>
    </div>
  )
}
