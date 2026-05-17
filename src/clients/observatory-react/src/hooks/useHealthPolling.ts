import { useState, useEffect, useRef } from 'react'
import type { HealthStatus } from '../types/api.ts'
import { fetchHealth } from '../api/client.ts'

export function useHealthPolling() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const data = await fetchHealth()
        if (!cancelled) setHealth(data)
      } catch {
        // Ignore fetch errors during polling
      }
    }

    poll()
    intervalRef.current = setInterval(poll, 5000)

    return () => {
      cancelled = true
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [])

  return health
}
