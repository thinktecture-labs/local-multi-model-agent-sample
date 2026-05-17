# Model Licenses — Read Before Redistributing

The Apache-2.0 [`LICENSE`](../LICENSE) at the repo root covers **this repo's code**.

It does **not** cover the model weights you download, train against, or merge into
GGUFs. Those follow their respective base-model licenses. If you fine-tune one of
these models and publish the result (e.g. push to HuggingFace, ship in a product),
the base-model terms come with it.

---

## Base models used by this pipeline

| Model | HF ID | License | Click-through? |
|---|---|---|---|
| Gemma 3 (1B, 4B, EmbeddingGemma 300M) | `google/gemma-3-*`, `google/embeddinggemma-300m` | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | **Yes — once per HF account** |
| Qwen 3.5-4B | `Qwen/Qwen3.5-4B` | [Tongyi Qianwen License Agreement](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE) | No |
| GLM-OCR (optional, for OCR upload path) | `zai-org/GLM-OCR` | [MIT](https://huggingface.co/zai-org/GLM-OCR) (check current README) | No |

> The `ggml-org/*-GGUF` repos referenced in `setup.sh` are official llama.cpp
> quantizations of the same upstream models above — same license terms apply.

---

## What "derivative work" means here

When you run `python -m finetune.train_gemma3` and produce
`models/gemma3-1b-ft-merged/gemma3-ft-<scenario>.gguf`, that file is a **derivative
work of Gemma 3**. Anyone who downloads it must accept the Gemma Terms of Use
the same way you did when you first pulled the base weights from HuggingFace.

Practical implications:
- **Inside this repo only**: nothing to do. The base weights are gitignored
  (`models/*.gguf`), the FT outputs are gitignored, and the training data is
  scenario-specific synthetic content owned by you.
- **Publishing a fine-tuned GGUF**: include the Gemma / Qwen license file in
  the release, name the base model in the model card, and follow the upstream
  attribution requirements.
- **Building a product on top**: the model is "used", not redistributed —
  most permissive license terms allow that. Read the actual license; this is
  not legal advice.

---

## Gemma 3 — quick reference

The [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (you accept them
when you click "Agree" on the HuggingFace page) permit:

- Commercial use, distribution, modification, fine-tuning
- Creating derivative models (including merged + quantized GGUFs)

And require:
- Including the prohibited-use policy in any redistribution
- Marking outputs from Gemma as AI-generated when relevant
- Not redistributing without including the Terms

There's **no patent grant** in the Gemma Terms (unlike Apache-2.0). For most
AI-application scenarios this is benign, but it's the main thing that
distinguishes "Gemma derivative" from "Apache-2.0 derivative."

---

## Qwen 3.5 — quick reference

The [Tongyi Qianwen License](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE)
permits commercial use up to a monthly active user threshold (re-check the
current text — Alibaba has revised this between Qwen versions). Above the
threshold a separate commercial license is required.

For typical local-AI / on-prem deployments well under that MAU bar, the license
behaves like an Apache-2.0-equivalent. Attribution to Qwen / Alibaba Cloud is
required in any distribution.

---

## Your fine-tuned outputs

The training data, prompts, and scenario configurations in this repo are MIT/Apache-2.0
licensed (per the repo `LICENSE`). The **merged GGUF model files** that
the training pipeline produces are:

- a derivative of the base model → covered by **Gemma Terms / Qwen License**
- AND contain weights influenced by your training data → covered by **your own license**

In practical terms, when you publish a fine-tuned GGUF you should ship:
1. The base-model license (Gemma Terms or Qwen License, alongside the GGUF)
2. A model card describing your training data + intended use (you can use
   the existing `data/training-data/*.jsonl` as the data card)
3. Your own license for the training-data-derived artifacts (Apache-2.0
   inherited from this repo's `LICENSE` is the default)

---

## Not legal advice

This is a one-page operator's summary. For anything material (commercial release,
public model card, enterprise deployment) consult an actual lawyer who reads
AI licenses for a living — the Gemma and Qwen license texts have evolved
multiple times and the obligations may have changed since this document was
written.
