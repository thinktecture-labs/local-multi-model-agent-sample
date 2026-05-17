"""
Training and evaluation routes — live fine-tuning with SSE progress.

Routes:
  POST /train         — start training with SSE progress
  GET  /train/status  — current training status
  POST /eval          — run model evaluation
  GET  /eval/results  — stored before/after snapshots
  POST /eval/reset    — clear stored eval results
"""

import asyncio
import json as _json
import os
import re as _re
import subprocess
import sys
import time as _time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .models import TrainRequest, EvalRequest, SwapRequest
from .state import state, training_lock, FT_PORTS

router = APIRouter()

_EPOCH_RE = _re.compile(r"'epoch':\s*([\d.]+)")
_LOSS_RE = _re.compile(r"'loss':\s*([\d.]+)")


def _train_sse(stage: str, message: str, **extra) -> str:
    """Format a training SSE event."""
    data = {"stage": stage, "message": message, **extra}
    return f"event: progress\ndata: {_json.dumps(data)}\n\n"


async def _simulate_training(total_epochs: int = 7):
    """Demo mode: emit realistic training progress events without GPU."""
    import math
    yield _train_sse("preparing", "Loading gemma3 1B model...", progress=0.0)
    await asyncio.sleep(1.5)
    yield _train_sse("preparing", "Preparing training data...", progress=0.05)
    await asyncio.sleep(1.0)

    for epoch in range(1, total_epochs + 1):
        loss = 2.5 * math.exp(-0.45 * epoch) + 0.15
        progress = epoch / total_epochs
        yield _train_sse(
            "training", f"Epoch {epoch}/{total_epochs} — loss: {loss:.3f}",
            epoch=epoch, total_epochs=total_epochs, loss=round(loss, 3),
            progress=round(progress, 3),
        )
        state.training_stage = "training"
        await asyncio.sleep(2.5)

    yield _train_sse("converting", "Converting to GGUF format...", progress=0.0)
    state.training_stage = "converting"
    await asyncio.sleep(3.0)
    yield _train_sse("converting", "GGUF conversion complete", progress=1.0)
    await asyncio.sleep(0.5)


async def _run_real_training(task: str):
    """Spawn training subprocess and yield SSE events from stdout."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable, "-m", "finetune.train_gemma3", "--task", task]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    yield _train_sse("preparing", "Spawning training process...", progress=0.0)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=repo_root,
        env=env,
    )

    total_epochs = 7
    last_heartbeat = _time.time()
    state.training_stage = "training"

    async for line_bytes in proc.stdout:
        line = line_bytes.decode(errors="replace").strip()
        if not line:
            continue

        epoch_m = _EPOCH_RE.search(line)
        loss_m = _LOSS_RE.search(line)

        if epoch_m or loss_m:
            epoch = float(epoch_m.group(1)) if epoch_m else 0
            loss = float(loss_m.group(1)) if loss_m else 0
            progress = epoch / total_epochs if total_epochs else 0
            yield _train_sse(
                "training",
                f"Epoch {int(epoch)}/{total_epochs} — loss: {loss:.3f}",
                epoch=round(epoch, 2), total_epochs=total_epochs,
                loss=round(loss, 3), progress=round(min(progress, 1.0), 3),
            )
        elif "loading" in line.lower() or "Loading" in line:
            yield _train_sse("preparing", line, progress=0.0)
        elif "saved" in line.lower() or "merged" in line.lower():
            yield _train_sse("training", line, progress=1.0)
        else:
            now = _time.time()
            if now - last_heartbeat > 2.0:
                yield ": heartbeat\n\n"
                last_heartbeat = now

    await proc.wait()
    if proc.returncode != 0:
        yield _train_sse("error", f"Training failed (exit code {proc.returncode})")
        return

    yield _train_sse("converting", "Converting to GGUF format...", progress=0.0)
    state.training_stage = "converting"
    conv = await asyncio.to_thread(
        subprocess.run,
        ["bash", "finetune/convert_gemma3_to_gguf.sh"],
        capture_output=True, text=True, timeout=120, cwd=repo_root,
    )
    if conv.returncode != 0:
        yield _train_sse("error", f"GGUF conversion failed: {conv.stderr[:200]}")
        return
    yield _train_sse("converting", "GGUF conversion complete", progress=1.0)


@router.post("/train", tags=["Fine-Tuning"])
async def start_training(request: TrainRequest):
    """
    Start live fine-tuning with SSE progress streaming.

    Returns a Server-Sent Event stream with stages:
    preparing → training (epoch/loss) → converting → swapping → complete.
    On completion, the fine-tuned model is auto-deployed.
    """
    async with training_lock:
        if state.training_running:
            raise HTTPException(status_code=409, detail="Training already in progress")
        state.training_running = True
        state.training_stage = "preparing"

    async def event_stream():
        t0 = _time.time()

        try:
            if request.demo_mode:
                async for event in _simulate_training():
                    yield event
            else:
                async for event in _run_real_training(request.task):
                    yield event
                    if '"stage": "error"' in event:
                        return

            # Start FT servers on secondary ports, then swap
            yield _train_sse("swapping", "Starting fine-tuned servers...", progress=0.0)
            state.training_stage = "swapping"
            try:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ft_start = await asyncio.to_thread(
                    subprocess.run,
                    ["bash", "scripts/start_servers.sh", "--bg", "--ft-extra"],
                    capture_output=True, text=True, timeout=90, cwd=repo_root,
                )
                if ft_start.returncode != 0:
                    raise RuntimeError(f"FT server start failed: {ft_start.stderr}")
                yield _train_sse("swapping", "Waiting for FT servers to be healthy...", progress=0.3)

                import httpx
                ft_ports = [FT_PORTS["inference"], FT_PORTS["function"], FT_PORTS["embedding"]]
                async with httpx.AsyncClient(timeout=5.0) as http:
                    for attempt in range(30):
                        try:
                            checks = [(await http.get(f"http://localhost:{p}/health")).status_code == 200 for p in ft_ports]
                            if all(checks):
                                break
                        except Exception:
                            pass
                        await asyncio.sleep(1)
                        yield _train_sse("swapping", f"Waiting for FT servers... {attempt + 1}s", progress=0.3 + 0.5 * (attempt / 30))

                # Import swap_models from system_routes to perform the swap
                from .system_routes import swap_models
                swap_result = await swap_models(SwapRequest(mode="finetuned"))
                yield _train_sse("swapping", swap_result.message, progress=1.0)
            except Exception as exc:
                yield _train_sse("error", f"Model swap failed: {exc}")
                return

            elapsed = _time.time() - t0
            yield _train_sse(
                "complete",
                f"Training complete in {elapsed:.0f}s — fine-tuned model deployed",
                progress=1.0, elapsed_s=round(elapsed, 1),
            )
            state.training_stage = "complete"

        except Exception as exc:
            yield _train_sse("error", str(exc))
            state.training_stage = "error"
        finally:
            state.training_running = False

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/train/status", tags=["Fine-Tuning"])
async def training_status() -> dict:
    """Return the current training status."""
    return {
        "running": state.training_running,
        "stage": state.training_stage,
    }


@router.post("/eval", tags=["Fine-Tuning"])
async def run_evaluation(request: EvalRequest):
    """
    Run intent classification evaluation against a fixed 60-query test set.

    Returns overall accuracy and per-class breakdown. Results are stored
    as 'before' or 'after' snapshots for side-by-side comparison.
    """
    if state.training_running:
        raise HTTPException(status_code=409, detail="Training in progress — eval blocked")

    from finetune.eval_gemma3 import run_eval, score

    results = await run_eval(state.client)
    scored = score(results)

    label = request.save_as
    if not label:
        label = "before" if "before" not in state.eval_results else "after"

    response_data = {
        "model": results.get("model", "unknown"),
        "timestamp": results.get("timestamp", ""),
        "overall_accuracy": scored["overall_accuracy"],
        "overall_correct": scored["overall_correct"],
        "n": scored["n"],
        "per_class": scored["per_class"],
        "saved_as": label,
    }
    state.eval_results[label] = response_data
    return response_data


@router.get("/eval/results", tags=["Fine-Tuning"])
async def get_eval_results() -> dict:
    """Return stored before/after eval snapshots."""
    return state.eval_results


@router.post("/eval/reset", tags=["Fine-Tuning"])
async def reset_eval_results() -> dict:
    """Clear stored eval results."""
    state.eval_results = {}
    return {"status": "cleared"}
