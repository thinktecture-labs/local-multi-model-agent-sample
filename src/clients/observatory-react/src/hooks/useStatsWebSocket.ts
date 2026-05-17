import { useState, useEffect, useRef } from 'react'
import type { GpuStats, EnergyStats, StatsMessage } from '../types/api.ts'

export function useStatsWebSocket() {
  const [gpu, setGpu] = useState<GpuStats | null>(null)
  const [energy, setEnergy] = useState<EnergyStats | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true

    function connect() {
      if (!mountedRef.current) return

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const url = `${protocol}//${window.location.host}/ws/stats`
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as StatsMessage
          if (mountedRef.current) {
            if (data.gpu) setGpu(data.gpu)
            if (data.energy) setEnergy(data.energy)
          }
        } catch {
          // Ignore parse errors
        }
      }

      ws.onclose = () => {
        if (mountedRef.current) {
          reconnectTimer.current = setTimeout(connect, 3000)
        }
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      mountedRef.current = false
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return { gpu, energy }
}
