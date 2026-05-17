#!/usr/bin/env python3
"""Helper for show_stack.sh — dump GGUF metadata for given paths.

Reads `general.size_label`, `general.file_type`, and `<arch>.context_length`.
Caches results to `scripts/.cache/gguf_meta.json` keyed by (abs_path, mtime).
GGUFReader is lazy-imported only on cache miss — keeps cached lookups fast.

Usage:  ./scripts/_gguf_meta.py PORT=PATH PORT=PATH ...
Output: one CSV line per arg → "PORT,SIZE_LABEL,QUANT,TRAINED_CTX"
        or "PORT,?,?,?" on read failure.

Requires: gguf python package available at vendor/llama.cpp/gguf-py
(invoked via PYTHONPATH=... by show_stack.sh) — only on cache miss.
"""
import json
import os
import sys

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "gguf_meta.json")

QUANT = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0",
    12: "Q4_K_S", 14: "Q4_K", 15: "Q4_K_M",
    17: "Q5_K_M", 18: "Q6_K", 32: "BF16",
}

# --- Load cache ---
cache = {}
try:
    with open(CACHE_FILE) as f:
        cache = json.load(f)
except Exception:
    pass

# --- Lazy GGUFReader import (only on cache miss) ---
_GGUFReader = None
def _get_reader():
    global _GGUFReader
    if _GGUFReader is None:
        from gguf import GGUFReader as _R
        _GGUFReader = _R
    return _GGUFReader


def read_gguf(abs_path):
    """Read size_label, file_type, trained_ctx from GGUF metadata. Returns dict."""
    r = _get_reader()(abs_path)
    sl, ft, ctx = "?", "?", "?"
    for f in r.fields.values():
        if f.name == "general.size_label":
            try:
                sl = bytes(f.parts[-1]).decode("utf-8")
            except Exception:
                pass
        elif f.name == "general.file_type":
            ft = QUANT.get(int(f.parts[-1][0]), "?")
        elif f.name.endswith(".context_length") and f.name.startswith("general.") is False:
            try:
                ctx = int(f.parts[-1][0])
            except Exception:
                pass
    return {"size_label": sl, "file_type": ft, "trained_ctx": ctx}


# --- Main ---
dirty = False
for arg in sys.argv[1:]:
    if "=" not in arg:
        continue
    port, path = arg.split("=", 1)
    abs_path = os.path.abspath(path)

    try:
        mtime = int(os.path.getmtime(abs_path))
    except OSError:
        print(f"{port},?,?,?")
        continue

    cached = cache.get(abs_path)
    if cached and cached.get("mtime") == mtime:
        meta = cached
    else:
        try:
            meta = read_gguf(abs_path)
            meta["mtime"] = mtime
            cache[abs_path] = meta
            dirty = True
        except Exception:
            print(f"{port},?,?,?")
            continue

    print(f"{port},{meta['size_label']},{meta['file_type']},{meta['trained_ctx']}")

# --- Save cache if anything changed ---
if dirty:
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass
