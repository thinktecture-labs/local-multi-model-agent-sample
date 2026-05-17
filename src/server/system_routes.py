"""
System routes — GPU stats, energy, privacy, network/routing toggles, model swap.

Routes:
  GET  /gpu           — real-time GPU statistics
  GET  /energy        — accumulated energy consumption for the session
  GET  /privacy       — zero-exfiltration proof
  POST /network-mode  — toggle online/offline
  POST /routing-mode  — toggle local-only/hybrid
  POST /models/swap   — hot-swap base / fine-tuned models
  GET  /models/mode   — current model mode
"""

import asyncio
import platform
import subprocess
import time as _time

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from src.engine.inference.config import SCENARIO_CONFIG
from .models import EnergyStats, GpuStats, PrivacyStats, SwapRequest, SwapResponse
from .state import state, BASE_PORTS, FT_PORTS, STARTUP_TIME

router = APIRouter()


@router.get("/scenario", tags=["System"])
async def get_scenario() -> dict:
    """Return the active scenario config metadata + UI suggestions."""
    import json as _json
    from pathlib import Path as _Path
    extra: dict = {}
    try:
        spath = _Path(__file__).resolve().parents[2] / "scenarios" / f"{SCENARIO_CONFIG.name}.json"
        with open(spath) as f:
            raw = _json.load(f)
        for key in ("suggestions", "logo_svg", "favicon_svg"):
            if raw.get(key):
                extra[key] = raw[key]
    except Exception:
        pass
    return {
        "scenario": SCENARIO_CONFIG.name,
        "brand": SCENARIO_CONFIG.brand,
        "label": SCENARIO_CONFIG.label,
        "language": SCENARIO_CONFIG.language,
        **extra,
    }


@router.get("/gpu", response_model=GpuStats, tags=["System"])
async def gpu_stats() -> GpuStats:
    """Real-time GPU statistics. Supports CUDA and Metal (via mactop), degrades to CPU."""
    # Try NVIDIA CUDA
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            gpu_power = float(parts[5]) if len(parts) > 5 else 0.0
            await _accumulate_energy(gpu_power, 0.0, "cuda")
            return GpuStats(
                available=True, backend="cuda",
                name=parts[0].strip(),
                vram_used_mb=float(parts[1]),
                vram_total_mb=float(parts[2]),
                utilization_pct=float(parts[3]),
                temperature_c=float(parts[4]),
                gpu_power_w=round(gpu_power, 2),
            )
    except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
        pass

    # Try macOS Metal via mactop (headless JSON, no sudo required)
    if platform.system() == "Darwin":
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["mactop", "--headless", "--count", "1"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                import json as _json
                samples = _json.loads(result.stdout)
                sample = samples[0] if isinstance(samples, list) else samples
                soc = sample.get("soc_metrics", {})
                mem = sample.get("memory", {})
                total_mb = mem.get("total", 0) / (1024 * 1024)
                used_mb = mem.get("used", 0) / (1024 * 1024)
                sys_info = sample.get("system_info", {})
                chip_name = sys_info.get("name", "Apple Silicon")
                gpu_cores = sys_info.get("gpu_core_count", 0)
                name_str = f"{chip_name} ({gpu_cores}-core GPU, Metal)" if gpu_cores else f"{chip_name} (Metal)"
                gpu_pw = round(soc.get("gpu_power", 0), 2)
                sys_pw = round(soc.get("system_power", 0), 2)
                await _accumulate_energy(gpu_pw, sys_pw, "metal")
                return GpuStats(
                    available=True, backend="metal",
                    name=name_str,
                    vram_used_mb=round(used_mb, 0),
                    vram_total_mb=round(total_mb, 0),
                    utilization_pct=round(sample.get("gpu_usage", 0), 1),
                    temperature_c=round(soc.get("gpu_temp", 0), 1),
                    gpu_power_w=gpu_pw,
                    system_power_w=sys_pw,
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass

        # Fallback: system_profiler (static info only)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                import json as _json
                data = _json.loads(result.stdout)
                gpu_info = data.get("SPDisplaysDataType", [{}])[0]
                name = gpu_info.get("sppci_model", "Apple GPU")
                return GpuStats(available=True, backend="metal", name=name)
        except Exception:
            pass

    return GpuStats(available=False, backend="cpu", name="CPU mode")


async def _accumulate_energy(gpu_power_w: float, system_power_w: float, backend: str) -> None:
    """Integrate GPU power over time to accumulate watt-hours."""
    now = _time.monotonic()
    async with state._energy_lock:
        state.energy_backend = backend
        if state.energy_last_sample_time > 0 and gpu_power_w > 0:
            dt_hours = (now - state.energy_last_sample_time) / 3600.0
            # Use trapezoidal integration: average of last and current power
            avg_power = (state.energy_last_gpu_w + gpu_power_w) / 2.0
            state.energy_wh += avg_power * dt_hours
        state.energy_last_sample_time = now
        state.energy_last_gpu_w = gpu_power_w
        state.energy_last_sys_w = system_power_w
        state.energy_samples += 1


# UK grid carbon intensity ~230 g CO2/kWh (2024 avg)
# US data center mix ~390 g CO2/kWh
_CO2_LOCAL_G_PER_KWH = 230.0
_CO2_CLOUD_G_PER_KWH = 390.0
# EU electricity ~34c/kWh = ~EUR 0.34/kWh
_ELECTRICITY_EUR_PER_KWH = 0.34
# Cloud energy multiplier: cloud GPU inference uses ~10-50x more energy
# for SLM-class queries (network overhead, cooling, larger GPUs, utilization waste)
_CLOUD_ENERGY_MULTIPLIER = 15.0


@router.get("/energy", response_model=EnergyStats, tags=["System"])
async def energy_stats() -> EnergyStats:
    """Accumulated energy consumption for the session.

    Integrates GPU power draw over time (sampled from /gpu polls).
    Works with both CUDA (nvidia-smi power.draw) and Metal (mactop gpu_power).
    """
    total_wh = state.energy_wh
    queries = state.agent.interaction_count
    uptime = _time.time() - STARTUP_TIME
    estimated_cloud_wh = total_wh * _CLOUD_ENERGY_MULTIPLIER

    return EnergyStats(
        total_wh=round(total_wh, 4),
        total_queries=queries,
        wh_per_query=round(total_wh / queries, 4) if queries > 0 else 0.0,
        gpu_power_now_w=state.energy_last_gpu_w,
        system_power_now_w=state.energy_last_sys_w,
        co2_local_g=round(total_wh / 1000.0 * _CO2_LOCAL_G_PER_KWH, 4),
        co2_cloud_g=round(estimated_cloud_wh / 1000.0 * _CO2_CLOUD_G_PER_KWH, 4),
        electricity_cost_local=round(total_wh / 1000.0 * _ELECTRICITY_EUR_PER_KWH, 6),
        estimated_cloud_wh=round(estimated_cloud_wh, 4),
        backend=state.energy_backend,
        sample_count=state.energy_samples,
        uptime_seconds=round(uptime, 1),
    )


@router.get("/privacy", response_model=PrivacyStats, tags=["System"])
async def privacy_stats() -> PrivacyStats:
    """Prove zero external data transfer. All processing is local."""
    return PrivacyStats(
        total_queries=state.agent.interaction_count,
        total_tokens_generated=state.agent.total_tokens_generated,
        external_bytes_sent=state.cloud_bytes_sent,
        uptime_seconds=round(_time.time() - STARTUP_TIME, 1),
        network_mode=state.network_mode,
        routing_mode=state.routing_mode,
    )


@router.post("/network-mode", tags=["System"])
async def toggle_network_mode() -> dict:
    """Toggle between online and offline mode.

    When offline, all cloud API calls are blocked. The local agent
    continues to work — proving zero-cloud operation.
    """
    async with state._mode_lock:
        state.network_mode = "offline" if state.network_mode == "online" else "online"
        new_mode = state.network_mode
    return {"network_mode": new_mode}


@router.post("/routing-mode", tags=["System"])
async def toggle_routing_mode() -> dict:
    """Toggle between local-only and hybrid routing.

    In hybrid mode, the agent auto-escalates to a cloud LLM when
    its confidence in the local response is below threshold.
    """
    async with state._mode_lock:
        state.routing_mode = "hybrid" if state.routing_mode == "local-only" else "local-only"
        new_mode = state.routing_mode
    return {"routing_mode": new_mode}


@router.post("/models/swap", response_model=SwapResponse, tags=["System"])
async def swap_models(request: SwapRequest) -> SwapResponse:
    """Zero-downtime model swap via dual-port redirection (~100ms)."""
    if request.mode == state.model_mode:
        return SwapResponse(status="no_change", mode=request.mode, message="Already in this mode")

    target_ports = FT_PORTS if request.mode == "finetuned" else BASE_PORTS

    # Health-check target ports (all must be running)
    import httpx
    async with httpx.AsyncClient(timeout=3.0) as http:
        for role, port in target_ports.items():
            try:
                r = await http.get(f"http://localhost:{port}/health")
                if r.status_code != 200:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{role} server on port {port} not healthy. Start servers first.",
                    )
            except httpx.ConnectError:
                raise HTTPException(
                    status_code=400,
                    detail=f"{role} server on port {port} not running. "
                           f"{'Use --all flag or run training first.' if request.mode == 'finetuned' else 'Start base servers.'}",
                )

    # Swap: redirect client to target ports
    urls = {role: f"http://localhost:{port}/v1" for role, port in target_ports.items()}
    state.client.swap_urls(urls)
    state.model_mode = request.mode

    label = "fine-tuned" if request.mode == "finetuned" else "base"
    return SwapResponse(status="swapped", mode=request.mode, message=f"Switched to {label} models")


@router.get("/models/mode", tags=["System"])
async def get_model_mode() -> dict:
    """Return the current model mode (base or finetuned)."""
    return {"mode": state.model_mode}


# ---------------------------------------------------------------------------
# WebSocket — real-time GPU + energy push (replaces client-side polling)
# ---------------------------------------------------------------------------

@router.websocket("/ws/stats")
async def ws_stats(ws: WebSocket):
    """Push combined GPU + energy stats every ~3 s over a single WebSocket."""
    await ws.accept()
    import json as _json
    try:
        while True:
            gpu = await gpu_stats()           # reuses existing logic + energy accumulation
            energy = await energy_stats()
            payload = {
                "gpu": gpu.model_dump(),
                "energy": energy.model_dump(),
            }
            await ws.send_text(_json.dumps(payload))
            await asyncio.sleep(3)
    except (WebSocketDisconnect, Exception):
        pass
