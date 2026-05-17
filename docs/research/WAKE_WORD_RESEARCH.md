# Wake Word / Keyword Spotting Research

> **Decision (2026-04-02):** Replaced Picovoice Porcupine with **OpenWakeWord** (`openwakeword-wasm-browser`, MIT license). "Hey Jarvis" keyword, in-browser ONNX inference, no API key. See `src/clients/observatory-react/src/hooks/useWakeWord.ts`.

**Date:** 2026-03-26
**Context:** Local-first, privacy-focused AI agent system running entirely on-device.
**Target platforms:** Browser (React SPA), macOS (M5 Max, Metal), Linux (NVIDIA CUDA, Vulkan/AMD)

---

## 1. Solution Comparison Table

| Criteria | **Porcupine (Picovoice)** | **OpenWakeWord** | **TensorFlow.js Speech Commands** | **Silero VAD + @ricky0123/vad** | **Mycroft Precise** | **Snowboy** |
|---|---|---|---|---|---|---|
| **License** | Apache 2.0 (code); proprietary AccessKey required; free tier = 3 active users/month | Apache 2.0 (code); **CC BY-NC-SA 4.0** (pre-trained models -- non-commercial!) | Apache 2.0 | MIT (zero strings attached) | Apache 2.0 | Abandoned (Kitt-AI shut down 2020) |
| **Custom wake words** | Yes -- type phrase in console, model trained in <10s via transfer learning | Yes -- train via Colab notebook in <1hr using synthetic TTS data (Piper). Linux only for training. | Limited -- transfer learning on ~18 base commands. Must collect/record samples. Not true wake word. | N/A -- VAD only (speech vs. silence), not keyword-specific | Yes -- record ~3 samples, GRU-based training | Dead project. Seasalt-AI fork exists but minimal activity. |
| **Browser support** | **Native**: `@picovoice/porcupine-web` + React binding. WASM + Web Audio + Web Workers. ~2 MB bundle. | **No official browser port.** Community WASM port exists (Deep Core Labs). Official path: WebSocket streaming to Python backend. | **Native**: runs entirely in browser via WebGL/WASM. `@tensorflow-models/speech-commands` npm package. | **Native**: `@ricky0123/vad-web` + `@ricky0123/vad-react`. ONNX Runtime Web. Works great in browser. | No browser support | No browser support |
| **Latency** | ~30-50 ms per frame. End-to-end: <200 ms from utterance end to detection. | 80 ms per frame. End-to-end: ~200-500 ms from utterance end to detection. | ~200-400 ms (WebGL inference on spectrograms) | <1 ms per 30 ms audio chunk on CPU | ~200 ms (lightweight GRU) | N/A |
| **Model size** | ~1 MB (standard), ~200 KB (tiny), ~2 MB (browser bundle) | ~1.5 MB shared backbone (melspec + embedding models) + ~40-200 KB per wake word model | ~2-4 MB (base model + custom head) | ~1.8 MB (ONNX model) | ~1-2 MB (GRU model) | N/A |
| **Memory footprint** | ~2-3 MB runtime | ~10-30 MB (Python + ONNX runtime) | ~15-30 MB (TF.js runtime + WebGL) | ~5-10 MB in browser | ~10-20 MB (Python) | N/A |
| **Accuracy** | 97%+ detection rate; <1 false alarm per 10 hours (with background speech/noise) | <5% false-reject; <0.5 false-accept/hour (with threshold tuning + Silero VAD gating) | Varies widely with custom models; 85-95% on built-in commands; custom wake words significantly worse | High accuracy for speech vs. silence; not keyword-specific | Moderate; community reports mixed results with custom words | N/A |
| **GPU required?** | No -- CPU only, extremely lightweight | No -- CPU only (ONNX runtime) | No (WebGL preferred but WASM fallback) | No -- CPU only | No -- CPU only | N/A |
| **Platforms** | Linux, macOS (arm64/x86), Windows, iOS, Android, Web, Raspberry Pi, ARM Cortex-M | Linux (full), Windows/macOS (ONNX only), no mobile | Browser only (TF.js) | Browser (ONNX Web), Python (PyTorch/ONNX), Node.js | Linux primarily, macOS partial | Dead |
| **Active maintenance** | Very active. v4.0.2 released Feb 2026. 4.8k GitHub stars. | Moderately active. v0.6.0 released Feb 2024. ~2k GitHub stars. Last commit activity in 2024. | Low activity. TF.js speech-commands has infrequent updates. | Very active. Silero VAD v6.2.1 released Feb 2026. 8.6k stars. @ricky0123/vad v0.0.30 Nov 2025. | **Dead.** Mycroft AI ceased development early 2023 (patent troll lawsuit). OVOS fork exists. | **Dead.** Archived 2020. |
| **Languages** | EN, ZH, FR, DE, IT, JA, KO, PT, ES (9 languages) | English primarily; custom models possible for any language with TTS data | English only (pre-trained) | Language-agnostic (detects any speech) | English primarily | Dead |

---

## 2. Eliminated Solutions

### Snowboy
**Status:** Dead. Kitt-AI archived all repos in 2020. A Seasalt-AI fork exists but has minimal activity. Not recommended for any new project.

### Mycroft Precise
**Status:** Dead. Mycroft AI ceased operations in early 2023. OpenVoiceOS (OVOS) maintains a `precise-lite` fork, but it targets embedded Linux devices, has no browser support, and limited community. Not recommended.

### Web Speech API for Wake Words
**Status:** Not viable. The Web Speech API (`SpeechRecognition`) sends audio to a cloud server by default (Google/Apple's servers). Even with the on-device mode (Chrome 121+), it is designed for dictation, not keyword spotting. It shows a visible microphone indicator, cannot run silently in the background, and has no mechanism for efficient always-on wake word detection. It fundamentally violates the local-first requirement unless the browser's on-device mode is confirmed -- and even then, you cannot control the model or behavior.

### Whisper-based Continuous Listening in Browser
**Status:** Not feasible. Whisper operates on 30-second chunks, has 3-5 second latency (vanilla) or 380-520 ms (optimized with GPU), and requires significant compute. Running it continuously in a browser would destroy battery life and CPU. The consensus across all sources is clear: **never use ASR for wake word detection**. Use a dedicated lightweight wake word model, then hand off to Whisper.

---

## 3. Viable Solutions -- Detailed Analysis

### 3.1 Porcupine by Picovoice

**Strengths:**
- Best-in-class accuracy (97%+ detection, <1 false alarm/10h)
- First-party browser SDK with React bindings (`@picovoice/porcupine-web-react`)
- Custom wake word training in seconds (type phrase, get model)
- Runs entirely on-device, no server needed in browser
- Multi-platform: same engine on macOS/Linux/browser
- 9 languages including English and German
- Tiny footprint (~1 MB model, ~2 MB browser bundle)
- Very actively maintained (latest release Feb 2026)

**Weaknesses:**
- **Requires Picovoice AccessKey** -- even free tier needs account registration and key
- **Free tier limited to 3 active users/month** -- beyond that, paid plans kick in
- Custom wake word models generated via Console may have **time limits and platform restrictions** on free tier
- Built-in keywords (e.g., "Computer", "Jarvis") are unrestricted, but custom ones like "Hey Nextera" require Console training
- Proprietary model format (.ppn) -- you cannot inspect or modify models
- Vendor lock-in risk -- if Picovoice changes pricing or shuts down, models stop working

**Licensing verdict for commercial/demo:** Viable for demos and prototypes with the free tier. For production with >3 users, requires a paid plan. Built-in wake words work without restrictions.

### 3.2 OpenWakeWord

**Strengths:**
- Fully open source (Apache 2.0 code)
- Custom wake word training via synthetic speech -- no real recordings needed
- Competitive accuracy vs. Porcupine on benchmarks (outperformed Porcupine on "alexa" test)
- Silero VAD integration reduces false positives
- Lightweight per-model cost (~40-200 KB per additional wake word on shared backbone)
- Used by Home Assistant community (battle-tested)
- Can train "Hey Nextera" via Colab notebook in <1 hour

**Weaknesses:**
- **Pre-trained models are CC BY-NC-SA 4.0 (non-commercial!)**. You MUST train your own models for commercial use.
- **No official browser/JS port.** Options:
  - (a) WebSocket streaming from browser to Python backend (official example)
  - (b) Community WASM port via Deep Core Labs (experimental, uses ONNX Runtime Web with hybrid WASM/WebGPU backends)
- Custom model training **only supported on Linux** (requires Piper TTS)
- Python-only runtime (onnxruntime dependency)
- Last release v0.6.0 was Feb 2024 -- update pace slower than Porcupine
- Requires Python backend process running alongside your agent server

**Licensing verdict for commercial/demo:** Code is Apache 2.0. Pre-trained models are **non-commercial**. You must train your own model for "Hey Nextera" to use commercially -- which is doable and the trained model you create is yours.

### 3.3 Silero VAD + @ricky0123/vad (VAD-based approach)

**Strengths:**
- MIT license -- zero restrictions, no keys, no registration, no expiration
- Extremely fast (<1 ms per chunk on CPU)
- Excellent browser support via `@ricky0123/vad-web` and `@ricky0123/vad-react`
- 8.6k GitHub stars, very actively maintained (v6.2.1, Feb 2026)
- Tiny model (1.8 MB ONNX)
- Works on all platforms (PyTorch, ONNX, browser ONNX Runtime Web)
- Perfect complement to any wake word solution

**Weaknesses:**
- **Not a wake word detector** -- only detects speech vs. silence
- Cannot distinguish "Hey Nextera" from "Hello world" or any other speech
- Useful only as a pre-filter or as a component in a larger pipeline

**Role in architecture:** VAD is not a replacement for wake word detection, but an essential companion. Use Silero VAD to:
1. Gate wake word processing (only run wake word model when speech is detected)
2. Detect end-of-utterance after wake word triggers recording
3. Reduce false positives when combined with wake word scores

### 3.4 TensorFlow.js Speech Commands

**Strengths:**
- Fully open source (Apache 2.0)
- Runs entirely in browser (WebGL/WASM)
- Transfer learning API for custom sounds
- No server or key required

**Weaknesses:**
- **Not designed for custom wake words** -- transfer learning works for short commands but not reliable multi-word wake phrases like "Hey Nextera"
- Requires collecting real audio samples for training (no synthetic generation)
- Lower accuracy than purpose-built wake word engines
- Limited to English
- Infrequent updates, not purpose-built for always-on listening
- Larger memory footprint than Porcupine or Silero VAD

**Verdict:** Not recommended for wake word detection. Better suited for simple command recognition ("yes"/"no"/"start"/"stop").

---

## 4. Browser Continuous Listening Architecture

### 4.1 AudioWorklet vs. MediaRecorder

| Aspect | **AudioWorklet** | **MediaRecorder** |
|---|---|---|
| Latency | Very low (~3 ms at 44.1 kHz, 128-sample blocks) | Higher (event-driven, buffered chunks) |
| Processing | Real-time DSP in separate thread | Records to blob, post-processing only |
| Raw PCM access | Yes -- direct float32 samples every 128 frames | No -- encoded format (WebM/Opus) |
| WASM integration | Excellent -- can run WASM models inside worklet | Not applicable |
| CPU efficiency | High -- dedicated audio thread, no GC pauses | Moderate -- main thread encoding overhead |
| Use for wake word | **Ideal** -- continuous low-latency PCM stream to WASM model | **Not suitable** -- designed for recording segments |

**Recommendation:** Use **AudioWorklet** for continuous wake word listening. The existing `useVoice.ts` hook uses MediaRecorder for press-to-talk recording, which is correct for that use case. Wake word detection requires a separate always-on AudioWorklet pipeline.

### 4.2 Recommended Browser Audio Pipeline

```
navigator.mediaDevices.getUserMedia({ audio: true })
    |
    v
AudioContext (16 kHz, mono)
    |
    v
AudioWorkletProcessor (runs in audio thread)
    |-- Resamples to 16 kHz if needed
    |-- Buffers 80-512 ms frames
    |
    v
[Option A: In-browser WASM model]         [Option B: WebSocket to server]
    |-- Porcupine WASM                         |-- Stream PCM chunks
    |-- or openWakeWord ONNX Runtime Web       |-- to Python openWakeWord
    |                                          |-- detection result back
    v                                          v
Wake word detected event
    |
    v
Switch to MediaRecorder (record full utterance)
    |
    v
POST /voice/chat (existing Whisper pipeline)
```

### 4.3 Privacy Implications of Always-On Microphone

**Browser protections (2025+):**
- Permission prompt required before any microphone access
- Visible recording indicator (red dot / tab icon) when mic is active
- Permissions expire after 90 days of inactivity (Chrome 121+, Edge 121+, Firefox 123+)
- Some browsers expire mic permission when tab is closed
- `Permissions-Policy: microphone` HTTP header controls iframe access

**Privacy-by-design for this project:**
- All audio processing happens in-browser (AudioWorklet + WASM model) -- no audio leaves the device during wake word listening
- Audio is only sent to the local server (localhost) after wake word triggers recording
- No cloud services involved at any point
- Clear UI indicator showing "listening for wake word" state vs. "recording" state
- User must explicitly enable always-on listening (opt-in, not default)
- Provide "push-to-talk" as alternative (existing behavior)

### 4.4 Battery / CPU Impact

| Approach | CPU Usage | Battery Impact |
|---|---|---|
| Silero VAD only (browser WASM) | ~0.4% of one CPU core | Negligible |
| Porcupine WASM (continuous) | ~1-3% of one CPU core | Low |
| openWakeWord via WebSocket | ~1-2% browser + ~2-5% server | Low-moderate |
| TensorFlow.js speech commands | ~5-15% (WebGL inference) | Moderate |
| Whisper continuous (hypothetical) | ~30-80% GPU | Severe -- not viable |

AudioWorklet runs in a dedicated thread and does not block the main thread or cause UI jank. The wake word models are small enough that continuous inference is sustainable on modern hardware.

---

## 5. Integration with Existing Whisper STT Pipeline

The current project architecture (from `c4-4f-voice-pipeline.puml` and `useVoice.ts`) uses a press-to-talk model:

```
User clicks mic -> MediaRecorder -> POST /voice/chat -> Whisper -> Agent -> Piper TTS
```

### Proposed wake word-enhanced pipeline:

```
[Always-on layer -- browser]
AudioWorklet -> Wake Word Model (WASM) -> "Hey Nextera" detected!
                                              |
                                              v
[Recording layer -- browser]              Start MediaRecorder
User speaks query...                      (same as current useVoice.ts)
Silence detected (Silero VAD)             Stop MediaRecorder
                                              |
                                              v
[Processing layer -- server]              POST /voice/chat
                                          (existing pipeline unchanged)
                                          Whisper -> Agent -> Piper TTS
                                              |
                                              v
[Playback layer -- browser]               Play TTS audio
                                          Return to always-on layer
```

**Key points:**
- The existing `/voice/chat` endpoint and `useVoice.ts` processAudio function remain unchanged
- Wake word adds a new "idle listening" state before recording begins
- Silero VAD can detect end-of-utterance to auto-stop recording (replacing manual stop-button press)
- MicState type extends from `'idle' | 'recording' | 'processing'` to `'listening' | 'wakeword' | 'recording' | 'processing'`
- The server-side pipeline (Whisper, Agent, Piper) is untouched

---

## 6. In-Browser Wake Word Detection (Preferred)

Running wake word detection entirely in the browser is the better architecture: audio never leaves the browser until the wake word fires, no WebSocket streaming needed, simpler implementation.

### 6.1 Option A: Porcupine (`@picovoice/porcupine-react`) — Best for Demo

**npm packages:**
- `@picovoice/porcupine-react` v4.0.0 (14.3 MB, wraps porcupine-web)
- `@picovoice/web-voice-processor` v4.0.9 (3.4 MB, AudioWorklet mic capture)

**Architecture:** WASM core + Web Worker (inference off main thread) + AudioWorklet (mic capture at 16kHz)

**React integration (~30 lines):**

```tsx
import { usePorcupine } from "@picovoice/porcupine-react";
import { BuiltInKeyword } from "@picovoice/porcupine-web";

export default function WakeWordListener() {
  const { keywordDetection, isLoaded, isListening, init, start, stop, release } = usePorcupine();

  useEffect(() => {
    init("YOUR_ACCESS_KEY", [{ builtin: BuiltInKeyword.Computer }], { publicPath: "porcupine_params.pv" });
    return () => { release(); };
  }, []);

  useEffect(() => {
    if (keywordDetection !== null) {
      // Wake word detected → trigger existing recording flow
      startRecording();
    }
  }, [keywordDetection]);

  return (
    <button onClick={isListening ? stop : start}>
      {isListening ? "Listening..." : "Enable Wake Word"}
    </button>
  );
}
```

**14 built-in wake words:** Alexa, Americano, Blueberry, Bumblebee, **Computer**, Grapefruit, Grasshopper, Hey Google, Hey Siri, **Jarvis**, OK Google, Picovoice, Porcupine, Terminator

**Custom "Hey Nextera":** Type phrase in Picovoice Console → model trains in seconds → download `.ppn` file → load in React app.

**Constraints:**
- AccessKey always required (free signup, no credit card)
- Free tier: 3 active users/month (fine for keynote demo)
- Custom .ppn files may have time limits on free tier
- Proprietary model format — vendor lock-in

| Metric | Value |
|--------|-------|
| Latency | <200ms from utterance end to detection |
| Accuracy | 97%+, <1 false alarm/10h |
| Bundle | ~14 MB |
| CPU | ~1-3% of one core |
| Languages | EN, DE, FR, IT, JA, KO, PT, ES, ZH |

### 6.2 Option B: `openwakeword-wasm-browser` — Best for Production

**npm package:** `openwakeword-wasm-browser` v0.1.1 (19 MB, MIT license, depends on `onnxruntime-web`)

**Architecture:** Full openWakeWord ONNX pipeline running in-browser:

```
AudioWorklet (80ms chunks at 16kHz)
    → melspectrogram.onnx (WASM)    — PCM to mel spectrogram
    → embedding_model.onnx (WASM)   — mel frames to feature vectors
    → hey_nextera.onnx (WASM)       — embeddings to detection score
    → silero_vad.onnx (WASM)        — VAD confirmation gate
```

All 4 ONNX models run in WASM. ~5-15ms inference per 80ms chunk on desktop (well within real-time budget).

**React integration (~20 lines):**

```tsx
import WakeWordEngine from 'openwakeword-wasm-browser';

export default function WakeWordDemo() {
  const [detected, setDetected] = useState(null);
  const engine = useMemo(() => new WakeWordEngine({
    baseAssetUrl: '/openwakeword/models',
    keywords: ['hey_nextera'],
    detectionThreshold: 0.5,
    cooldownMs: 2000,
  }), []);

  useEffect(() => {
    let unsub;
    engine.load().then(() => {
      unsub = engine.on('detect', ({ keyword, score }) => {
        setDetected(keyword);
        startRecording();  // trigger existing voice pipeline
      });
      engine.start();
    });
    return () => { unsub?.(); engine.stop(); };
  }, [engine]);

  return <p>{detected ? `Detected: ${detected}` : 'Listening...'}</p>;
}
```

**Events:** `detect` (keyword, score, timestamp), `speech-start`, `speech-end`, `error`, `ready`

**Required assets in `public/`:**

```
public/openwakeword/models/
    melspectrogram.onnx          # shared backbone (~1.5 MB)
    embedding_model.onnx         # shared backbone (~1 MB)
    silero_vad.onnx              # VAD gate (~1.8 MB)
    hey_nextera.onnx             # custom trained (~200 KB)
  ort/
    ort-wasm-simd.wasm           # ONNX Runtime WASM
```

**Built-in wake words:** hey_jarvis, alexa, hey_mycroft, hey_rhasspy, timer, weather

**Custom "Hey Nextera" training:**
1. Generate ~13,000 synthetic positive samples using Piper TTS (speed/pitch variations)
2. Collect negative samples (similar-sounding phrases)
3. Train on RTX box using openWakeWord training pipeline (~1 hour)
4. Output: `hey_nextera.onnx` (~200 KB), Apache 2.0 licensed (you own it)
5. Tools: [openwakeword-trainer](https://github.com/lgpearson1771/openwakeword-trainer) or [openwakeword.com](https://openwakeword.com/) (web-based)
6. Drop into `public/openwakeword/models/` — no code changes needed

**Constraints:**
- v0.1.1 (single maintainer) — vendor the dependency
- 19 MB bundle size
- Custom training requires Linux + GPU
- Requires COOP/COEP headers for SharedArrayBuffer (SIMD-threaded WASM)

| Metric | Value |
|--------|-------|
| Latency | 200-500ms from utterance end to detection |
| Accuracy | <5% false-reject, <0.5 false-accept/hour (with VAD gating) |
| Bundle | ~19 MB |
| CPU | ~5-15% of one core |
| Languages | Any (custom training with TTS in any language) |

### 6.3 Head-to-Head Comparison

| | Porcupine | openwakeword-wasm-browser |
|---|---|---|
| **Lines of code** | ~30 | ~20 |
| **API key** | Required (always) | None |
| **Custom "Hey Nextera"** | Seconds (web console) | ~1hr (train on RTX) |
| **User limit** | 3/month free | Unlimited |
| **Bundle size** | 14 MB | 19 MB |
| **Latency** | <200ms | 200-500ms |
| **Maturity** | Production (v4.0) | Early (v0.1.1) |
| **License** | Proprietary models | MIT + Apache 2.0 |
| **Fits local-first ethos** | Mostly (needs key) | Perfectly |
| **Server involvement** | None | None |

### 6.4 Eliminated Browser Options

- **TensorFlow.js Speech Commands** — not designed for custom wake words, unreliable for multi-word phrases
- **`use-ear` npm** — wraps Web Speech API (cloud-based), not real wake word detection
- **`web-wake-word` npm** — commercial (DaVoice.io), proprietary models
- **`react-native-wakeword`** — React Native only, not browser

---

## 7. Recommendation

### For keynote demo: **Porcupine** with "Computer" or "Jarvis"

30 lines of React, works in 5 minutes, impressive. Get a free AccessKey from Picovoice Console. Use a built-in wake word — no training needed. Swap to custom "Hey Nextera" later via Console if desired.

### For production / open-source narrative: **`openwakeword-wasm-browser`**

Train "Hey Nextera" on the RTX box, ship the 200KB ONNX model with the app. Zero vendor dependency, unlimited users, MIT license. The v0.1.1 risk is manageable since the underlying ONNX models are the battle-tested openWakeWord ones from the Home Assistant community.

### Integration with existing voice pipeline

Either option plugs into the existing `useVoice.ts` identically:

```
[Always-on layer — browser]
AudioWorklet → Wake Word Model (WASM) → "Hey Nextera" detected!
                                              |
                                              v
[Recording layer — browser]              startRecording() in useVoice.ts
User speaks query...                     (existing MediaRecorder flow)
Silero VAD detects silence               stopRecording() auto
                                              |
                                              v
[Processing layer — server]              POST /voice/chat (unchanged)
                                         Whisper → Agent → Piper TTS
```

The existing `/voice/chat` endpoint, `useVoice.ts`, Whisper, and Piper pipeline remain **completely unchanged**.

MicState extends from `'idle' | 'recording' | 'processing'` to `'listening' | 'wakeword' | 'recording' | 'processing'`.

### Implementation phases

| Phase | What | Effort |
|---|---|---|
| Phase 1 | Add `@ricky0123/vad-react` for VAD-based auto-stop recording (replaces manual stop button) | 1-2 days |
| Phase 2 | Integrate wake word engine (Porcupine for demo OR openwakeword-wasm-browser for production) | 1-2 days |
| Phase 3 | Train custom "Hey Nextera" model on RTX box (openWakeWord path only) | 1 day |
| Phase 4 | UI states: listening indicator, wake word animation, opt-in toggle, push-to-talk fallback | 1-2 days |

**Total estimated effort: 4-7 days** (simpler than server-side approach — no WebSocket streaming needed)

---

## Sources

- [Porcupine GitHub - Picovoice](https://github.com/Picovoice/porcupine)
- [Porcupine Wake Word Docs](https://picovoice.ai/docs/porcupine/)
- [Porcupine Web Quick Start](https://picovoice.ai/docs/quick-start/porcupine-web/)
- [Picovoice Free Tier Announcement](https://picovoice.ai/blog/introducing-picovoices-free-tier/)
- [Picovoice Wake Word Guide 2026](https://picovoice.ai/blog/complete-guide-to-wake-word/)
- [OpenWakeWord GitHub](https://github.com/dscripka/openWakeWord)
- [OpenWakeWord Web Example](https://github.com/dscripka/openWakeWord/blob/main/examples/web/README.md)
- [Open Wake Word on the Web - Deep Core Labs](https://deepcorelabs.com/open-wake-word-on-the-web/)
- [voice-satellite-card - openWakeWord in Browser via ONNX Runtime Web](https://github.com/emme99/voice-satellite-card)
- [Silero VAD GitHub](https://github.com/snakers4/silero-vad)
- [@ricky0123/vad - Voice Activity Detection for JavaScript](https://github.com/ricky0123/vad)
- [@ricky0123/vad-react Documentation](https://docs.vad.ricky0123.com/user-guide/browser/)
- [TensorFlow.js Speech Commands](https://github.com/tensorflow/tfjs-models/tree/master/speech-commands)
- [Snowboy GitHub (Archived)](https://github.com/Kitt-AI/snowboy)
- [Mycroft Precise GitHub](https://github.com/MycroftAI/mycroft-precise)
- [OpenVoiceOS Precise Plugin](https://github.com/OpenVoiceOS/ovos-ww-plugin-precise)
- [VAD + Wake Word + Whisper Pipeline Architecture](https://thomasthelliez.com/blog/voice-activity-detection-and-wake-word-setup-for-whisper-based-voice-interfaces/)
- [Whisper as Wake Word Detection - Discussion](https://github.com/KoljaB/RealtimeSTT/issues/24)
- [AudioWorklet Design Patterns - Chrome](https://developer.chrome.com/blog/audio-worklet-design-pattern/)
- [AudioWorklet MDN Reference](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)
- [Browser Microphone Permissions - MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Permissions-Policy/microphone)
- [Best VAD Comparison 2025 - Picovoice](https://picovoice.ai/blog/best-voice-activity-detection-vad-2025/)
