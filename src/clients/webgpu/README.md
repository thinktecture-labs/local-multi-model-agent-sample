# WebGPU Document Triage Demo

Browser-based document analysis running **LFM 2.5 1.2B Instruct** (Liquid AI) entirely in-browser via WebGPU + Transformers.js.
No server, no API key, no data leaves the browser tab.

Part of the conference talk demo — the "in the browser, zero server" act.

## Quick Start

### 1. Install dependencies

```bash
cd webgpu-demo
npm install
```

This installs `@huggingface/transformers` locally (used via import map, no build step needed).

### 2. Serve the demo

The demo must be served over HTTP (not `file://`) for ES modules and Service Worker to work:

```bash
python3 -m http.server 8080
```

Then open **Chrome** at `http://localhost:8080`.

### 3. Load the model

Click **Download & Load Model** in the app. The model (~1.2 GB ONNX Q4) is downloaded from HuggingFace and cached in the browser for future sessions.

No HuggingFace account or token required — the model is public.

### 4. Use the demo

- Select a sample document or paste/drag-drop your own text or PDF
- Click **Process** to run analysis
- Results: document classification, entity extraction with PII flags, summary, routing suggestion

## Browser Requirements

- **Chrome 113+** with WebGPU enabled (default in recent Chrome)
- ~2 GB available GPU memory for the model
- Other browsers: WebGPU support varies (Firefox Nightly, Safari Technology Preview)

## Sample PDFs

Two **synthetic** sample PDFs are included in `samples/`:

- `patient_intake_referral.pdf` — Fictional cardiology referral with fabricated patient PII
- `lab_report_metabolic.pdf` — Fictional lab report with tables, flagged values

All names, identifiers, addresses, phone numbers, dates, and clinical findings
are entirely fabricated for the on-device entity-extraction demo. **No real
patient data.** Both PDFs carry a "SYNTHETIC SAMPLE" footer.

To regenerate them:

```bash
pip install fpdf2
python generate_samples.py
```

## Architecture

```text
webgpu-demo/
├── index.html              Single-file SPA (HTML + CSS + JS)
├── sw.js                   Service Worker for model caching
├── package.json            Transformers.js dependency
├── node_modules/           Local @huggingface/transformers (gitignored)
├── generate_samples.py     PDF sample generator (fpdf2)
├── samples/                Pre-built sample documents
│   ├── patient_intake_referral.pdf
│   └── lab_report_metabolic.pdf
├── .gitignore
└── README.md
```

**Stack**: Vanilla JS, no build system, no framework. Transformers.js via local import map.
Matches the `static/index.html` single-file pattern used by the main Observatory UI.

## How It Works

### Inference Pipeline

1. **Model loading**: Transformers.js `pipeline("text-generation")` loads the ONNX Q4 model with WebGPU backend. Model files are auto-downloaded from HuggingFace Hub and cached in the browser's Cache API via a Service Worker (`sw.js`, cache-first strategy for `.onnx`, `.onnx_data`, `.bin`, `.json` files)
2. **Prompt engineering**: Document text is wrapped in a structured system+user prompt requesting a JSON object with `document_type`, `summary`, `entities[]` (12 clinical types with PII flags), and `routing`. The system prompt enforces JSON-only output with no markdown or preamble
3. **Generation constraints**: Two mechanisms ensure clean JSON output:
   - **Bad word suppression**: Blocks markdown fences (`` ``` ``), preamble phrases ("Here is", "Sure,", "Certainly") at the token level
   - **JsonStructureProcessor**: Custom `LogitsProcessor` that suppresses the EOS token while JSON brackets/braces are unbalanced, preventing premature truncation. Includes an 800-token safety valve to prevent runaway generation
4. **Streaming inference**: `TextStreamer` streams tokens via callback, displayed live in the Model Trace panel. Greedy decoding (`do_sample: false`) with `repetition_penalty: 1.05` for deterministic output
5. **Progressive rendering**: As tokens stream in, result cards populate incrementally — classification badge appears first, then entities fade in one-by-one, followed by summary and routing. Each entity is deduplicated against previously rendered entities in real time

### Post-Processing Pipeline

After the model finishes generating, a multi-stage post-processing pipeline cleans and augments the results:

6. **JSON extraction**: Robust multi-strategy parser — tries direct parse, markdown-fenced JSON, balanced-brace scanning, truncated-JSON repair (closes unmatched brackets/braces), and field-level regex extraction as a last resort
7. **Entity repair**: Malformed entities are fixed — strings wrapped in objects, missing `type` inferred from text content via regex heuristics (lab values, medication dosages, physician titles, ID patterns), missing `is_pii` inferred from entity type
8. **Type normalization**: ~60+ model-emitted type variants mapped to 12 canonical types via `TYPE_MAP` (e.g., `drug` → `medication`, `vital_sign` → `vital`, `hospital` → `organization`). Certain types are silently dropped (`emergency_contact`, `family_member`, `spouse`)
9. **Re-typing misclassified entities**: Lab test names typed as MEDICATION are corrected to VITAL (Alk Phosphatase, AST/SGOT, ALT/SGPT, etc.). Patient names typed as CONDITION are detected via proper-noun regex. IDs/dates typed as CONDITION are re-typed based on content patterns
10. **Noise filtering**: Drops negated findings ("No cardiac history"), narrative/status text ("Diabetes education completed", "73-year-old male"), column headers, standalone labels, blob entities >150 chars
11. **Blob splitting**: Entities where the model crammed multiple items into one text field (e.g., "Augmentin 875mg BID, Metformin 1000mg BID, Lisinopril 20mg") are split into individual entities
12. **Post-processing document scan**: Regex-scans the original document text for vitals, dates, IDs, phone numbers, and addresses that the 1.2B model missed. The model tends to skip structured/tabular sections (vitals tables, lab values). Each candidate is checked against existing entities (exact + substring match) to avoid duplicates. Covers 24 vital/lab patterns, 7 date formats, 6 ID patterns, phone numbers, and street addresses
13. **Cross-type dedup**: If the same text appears as both MEDICATION and VITAL, the MEDICATION duplicate is removed
14. **PDF text extraction**: Drag-dropped PDFs are parsed via pdf.js (loaded dynamically from CDN) with Y-position line-break detection for structured text output
15. **Network proof**: `PerformanceObserver` counts requests during analysis — always 0

### Entity Types

| Type | Color | PII | Examples |
|------|-------|-----|----------|
| `patient` | Blue | Yes | Maria Gonzalez, Ahmed Al-Rashid |
| `physician` | Light blue | No | Dr. Thomas Weber, MD |
| `date` | Purple | DOB only | 04/22/1965 (PII), 2026-02-25 (non-PII) |
| `medication` | Green | No | Metoprolol 50mg daily, Augmentin 875mg BID |
| `condition` | Red | No | Community-acquired pneumonia, Essential hypertension |
| `allergy` | Red | No | Penicillin (anaphylaxis — SEVERE) |
| `procedure` | Light blue | No | Stress echocardiography, Repeat chest X-ray |
| `vital` | Green | No | Blood Pressure 142/88 mmHg, HbA1c 6.8% |
| `id` | Orange | Yes | NEX-20231547, BC-2847193, ***-**-4821 |
| `address` | Pink | Yes | 742 Evergreen Terrace, Stuttgart, 70174 |
| `phone` | Pink | Yes | +49-711-555-0842 |
| `organization` | Teal | No | Nextera Medical Center, Emergency Department |

## Performance

- Model: LFM 2.5 1.2B Instruct (Liquid AI) — ONNX Q4 quantization
- Download: ~1.2 GB (cached after first load)
- Speed: ~55-65 tok/s on modern GPU hardware (Apple M-series), TTFT ~850-1250ms
- Typical analysis: ~250-380 tokens, 4-8 seconds total depending on document length
- Entity yield: 28-42 entities per document (20-35 from model + 5-8 from post-processing scan)

### Per-Document Results

| Document | Entities | PII | Tokens | Time | Speed |
|----------|----------|-----|--------|------|-------|
| Patient Intake Referral | 28 | 6 | 249 | 4.7s | 65 tok/s |
| Lab Report — Metabolic | 39 | 4 | 373 | 7.5s | 58 tok/s |
| Discharge Summary | 42 | 7 | 380 | 7.7s | 57 tok/s |

## Stage Demo Flow

1. Open Chrome tab — app loads with Nextera branding (light mode default)
2. Click **Download & Load Model** (or pre-load before going on stage)
3. Select "Patient Intake Referral" from samples dropdown (or drag-drop the PDF)
4. Click **Process**
5. Watch streaming analysis: classification badge appears first, then entities fade in one-by-one with color-coded types and red PII dots, followed by summary and routing
6. Scroll entity list — point out PII detection (red dots for patient name, DOB, address, phone, insurance ID)
7. Open DevTools > Network tab: **zero requests** during processing
8. Toggle browser offline, process another document — same result
9. "Every web developer can add private AI to their application today"
