/** Mock for openwakeword-wasm-browser — used in tests (WASM/ONNX can't load in jsdom) */

export const MODEL_FILE_MAP: Record<string, string> = {
  hey_jarvis: 'hey_jarvis_v0.1.onnx',
  alexa: 'alexa_v0.1.onnx',
  hey_mycroft: 'hey_mycroft_v0.1.onnx',
  hey_rhasspy: 'hey_rhasspy_v0.1.onnx',
  timer: 'timer_v0.1.onnx',
  weather: 'weather_v0.1.onnx',
}

export class WakeWordEngine {
  constructor() {}
  async load() {}
  async start() {}
  async stop() {}
  setGain() {}
  setActiveKeywords() {}
  async runWav() { return 0 }
  on() { return () => {} }
  off() {}
}

export default WakeWordEngine
