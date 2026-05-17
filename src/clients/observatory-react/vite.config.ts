/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: '/app/',
  envDir: '../../../',  // read .env from repo root
  server: {
    port: 3000,
    proxy: {
      '/health': 'http://localhost:8000',
      '/query': 'http://localhost:8000',
      '/compare': 'http://localhost:8000',
      '/escalate': 'http://localhost:8000',
      '/models': 'http://localhost:8000',
      '/upload-document': 'http://localhost:8000',
      '/eval': 'http://localhost:8000',
      '/network-mode': 'http://localhost:8000',
      '/routing-mode': 'http://localhost:8000',
      '/voice': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    alias: {
      'openwakeword-wasm-browser': new URL('./src/test/__mocks__/openwakeword-wasm-browser.ts', import.meta.url).pathname,
    },
  },
})
