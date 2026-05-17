# Live PII Redaction Overlay — Keynote Demo Sketch

> Browser-based real-time PII detection and redaction on a live camera feed.
> Gemma 4 vision via WebGPU + Transformers.js. Zero server, zero cloud, zero data leaves the tab.

---

## The Keynote Moment

You hold up a printed document to your laptop camera — a patient referral, a military personnel file, an employee record. Within seconds, red redaction boxes appear over the PII in the live video feed: name, date of birth, address, insurance ID, phone number. The audience sees the document AND the redaction happening in real-time. You flip to a second document — new redactions appear.

Then you open DevTools Network tab: **zero requests**. Open Task Manager: **no server process**. "This 4-billion-parameter model is running entirely inside this Chrome tab."

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ BROWSER TAB                                                  │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐   │
│  │  getUserMedia │───>│ Frame Capture│───>│  Gemma 4     │   │
│  │  (webcam)     │    │ (1 per 3-5s) │    │  Vision VLM  │   │
│  └──────────────┘    └──────────────┘    │  (WebGPU)    │   │
│         │                                 └──────┬───────┘   │
│         │                                        │           │
│         ▼                                        ▼           │
│  ┌──────────────┐                        ┌──────────────┐   │
│  │ <video> live  │                        │ PII entities │   │
│  │ camera feed   │                        │ + text spans │   │
│  └──────┬───────┘                        └──────┬───────┘   │
│         │                                        │           │
│         ▼                                        ▼           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              <canvas> Composite Layer                 │   │
│  │                                                       │   │
│  │   Live video frame                                    │   │
│  │   + Red redaction boxes over PII regions              │   │
│  │   + Entity labels ("NAME", "DOB", "ADDRESS")          │   │
│  │   + Confidence badges                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Entity Sidebar (scrolling)                │   │
│  │  ● Maria Gonzalez ......... NAME        [PII] redacted│   │
│  │  ● 04/22/1965 ............. DOB         [PII] redacted│   │
│  │  ● 742 Evergreen Terrace .. ADDRESS     [PII] redacted│   │
│  │  ○ Essential hypertension . CONDITION         visible │   │
│  │  ○ Metoprolol 50mg ........ MEDICATION        visible │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Two-Phase Inference

VLMs like Gemma 4 output text, not bounding boxes. So we need two phases:

### Phase 1: PII Identification (Gemma 4 Vision)

Capture a video frame, send to Gemma 4 with prompt:

```
You are a PII detection system analyzing a document image.
List every piece of personally identifiable information visible.
For each item return: the exact text as it appears, the PII type
(NAME, DOB, ADDRESS, PHONE, ID, SSN, EMAIL), and the approximate
location (top/middle/bottom, left/center/right).
Return JSON array only, no explanation.
```

Response (~3-5 seconds on M5 Max at ~55 tok/s):
```json
[
  {"text": "Maria Gonzalez", "type": "NAME", "location": "top-left"},
  {"text": "04/22/1965", "type": "DOB", "location": "top-right"},
  {"text": "742 Evergreen Terrace", "type": "ADDRESS", "location": "middle-left"},
  {"text": "+49-711-555-0842", "type": "PHONE", "location": "bottom-left"}
]
```

### Phase 2: Text Localization (Browser-side OCR)

Use **Tesseract.js** (WASM, runs in browser) or the **Text Detection API** (Chrome origin trial) to find the pixel coordinates of each PII text string in the frame:

1. Run OCR on the captured frame → get word bounding boxes
2. Match PII strings from Phase 1 against OCR word boxes (fuzzy match)
3. Merge adjacent word boxes into redaction rectangles
4. Draw red overlay boxes on the canvas

Alternative: skip exact localization and use the coarse location hints from Gemma 4 ("top-left") to place approximate redaction zones. Less precise but simpler and avoids the Tesseract.js dependency.

---

## UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🔴 LIVE PII SCANNER                    ● 0 network requests   │
│  Gemma 4 (4B) · WebGPU · In-Browser     Model: loaded (1.8 GB) │
├───────────────────────────────────┬─────────────────────────────┤
│                                   │                             │
│                                   │  DETECTED PII               │
│                                   │                             │
│     ┌─────────────────────┐       │  ██ Maria Gonzalez   NAME   │
│     │                     │       │  ██ 04/22/1965       DOB    │
│     │   LIVE VIDEO FEED   │       │  ██ 742 Evergreen    ADDR   │
│     │                     │       │  ██ +49-711-555-..   PHONE  │
│     │   [red boxes over   │       │  ██ BC-2847193       ID     │
│     │    PII regions]     │       │                             │
│     │                     │       │  ─────────────────────────  │
│     │                     │       │  NON-PII ENTITIES           │
│     └─────────────────────┘       │                             │
│                                   │  ○ Essential hypertension   │
│   ◉ Scanning... (frame 4)        │  ○ Metoprolol 50mg daily    │
│   Last scan: 3.2s · 312 tokens   │  ○ Dr. Thomas Weber         │
│                                   │  ○ Stress echocardiography  │
│   [▶ Scan] [⏸ Pause] [📷 Snap]  │                             │
├───────────────────────────────────┴─────────────────────────────┤
│  MODEL TRACE (streaming tokens)                                  │
│  [{"text":"Maria Gonzalez","type":"NAME","location":"top-left"}, │
│   {"text":"04/22/1965","type":"DOB","location":"top-right"}, ... │
└─────────────────────────────────────────────────────────────────┘
```

---

## Interaction Modes

| Mode | Behavior | Keynote Use |
|------|----------|-------------|
| **Continuous scan** | Capture frame every 5s, auto-analyze | "Watch the boxes appear as I hold up the page" |
| **Snap** | Manual single-frame capture + analyze | Precise control on stage |
| **Freeze + redact** | Freeze frame, run analysis, overlay boxes, then export redacted image | "Here's the redacted version — download it. Still zero network." |

---

## Keynote Script (Act 5b)

**Setup:** The WebGPU document triage demo (Act 5a) has just finished. Model is already loaded.

1. "We just analyzed a document by dragging a file. But what about the physical world?"
2. *Click camera icon — live video feed appears*
3. *Hold up printed patient referral to camera*
4. "Gemma 4 is analyzing what it sees..."
5. *Red boxes start appearing over PII — name, DOB, address*
6. *Entity sidebar fills up — PII items marked red, clinical entities in green*
7. "Six pieces of personal data found and redacted. Zero network requests."
8. *Swap to a second document — German military personnel file*
9. *New redactions appear — Dienstgrad, Personalnummer, Geburtsdatum*
10. "Works in German too. Same model, same browser tab."
11. *Open DevTools Network tab — 0 requests*
12. "The camera feed never left Chrome. The pixels were analyzed by a 4-billion-parameter model running on your GPU — through WebGPU. No server. No API key. No data anywhere but your browser's memory."
13. *Close tab* — "And now the model is gone. Unloaded. No trace."

---

## Performance Budget

| Step | Time | Notes |
|------|------|-------|
| Frame capture | <1ms | canvas.drawImage from video element |
| Gemma 4 inference | 3-6s | ~250-400 tokens at 55-65 tok/s (M5 Max) |
| OCR localization (optional) | 1-2s | Tesseract.js WASM on single frame |
| Box overlay rendering | <1ms | Canvas 2D rectangles |
| **Total per scan** | **4-8s** | Acceptable for keynote pacing |

Not real-time video (30fps) — it's a **scan cadence** (~1 analysis per 5s). But the live video feed itself is smooth 30fps. The redaction boxes persist and animate in as results arrive. Visually feels responsive.

---

## Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Model | Gemma 4 4B (ONNX Q4 via Transformers.js) | Vision-capable, fits in browser GPU memory, good multilingual |
| PII localization | Coarse (top/middle/bottom + left/center/right) | Avoids Tesseract.js dependency, good enough for demo. Upgrade to exact OCR-based boxes later if needed |
| Video capture | getUserMedia + canvas snapshot | Standard, no dependencies |
| Redaction style | Semi-transparent red boxes + type labels | Visually clear on stage, audience can still see what's being redacted |
| Streaming | Token-by-token in trace panel | Proves model is running, builds tension |

---

## Dependencies

- `@huggingface/transformers` ^4.x (already in webgpu client)
- Gemma 4 ONNX Q4 model from HuggingFace Hub (~1.8-2.5 GB, cached after first load)
- Chrome 113+ with WebGPU (already required)
- Optional: `tesseract.js` for precise text localization

---

## Relationship to Existing Code

| Existing | Reuse |
|----------|-------|
| `src/clients/webgpu/index.html` | Model loading, Transformers.js pipeline, JSON extraction, streaming, Service Worker caching |
| `src/clients/webgpu/sw.js` | Model file caching (cache-first) |
| Entity type system + colors | Same 12 types, same color coding, same PII flag logic |
| Post-processing pipeline | JSON repair, entity normalization, type mapping |

The new code is primarily: camera capture, canvas overlay rendering, vision prompt (instead of text prompt), and the scan loop.

---

## Open Questions

1. **Gemma 4 ONNX availability** — Is Gemma 4 vision already exported to ONNX for Transformers.js? The HF Space runs it, but we need the ONNX Q4 variant. Check `webml-community/Gemma-4-WebGPU` files.
2. **GPU memory** — Gemma 4 4B Q4 might need ~3-4 GB GPU memory. M5 Max is fine. Will it fit alongside the main Observatory demo's llama-servers? Probably yes — browser uses system/unified memory, llama-servers use Metal.
3. **Vision prompt quality** — How well does Gemma 4 follow structured JSON output instructions for vision inputs? May need the same bad-word suppression + JsonStructureProcessor from the existing WebGPU demo.
4. **Multilingual PII** — German PII patterns (Personalnummer, Dienstgrad, Geburtsdatum) need testing. The model may need German-specific prompt variants.
5. **Tesseract.js vs. coarse localization** — Test both. If coarse "quadrant" boxes look silly on stage, add Tesseract.js for precise word-level boxes.
