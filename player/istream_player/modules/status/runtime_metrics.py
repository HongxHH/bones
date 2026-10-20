"""Lightweight host/process metrics for StatusExporter telemetry.

Failures are silent: missing psutil/pynvml/torch must never affect playback.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

log = logging.getLogger("RuntimeMetrics")


class RuntimeMetrics:
    def __init__(
        self,
        *,
        interval_s: float = 1.0,
        device_index: int = 0,
        enabled: bool = True,
    ) -> None:
        self._interval_s = max(0.25, float(interval_s))
        self._device_index = max(0, int(device_index))
        self._enabled = bool(enabled)
        self._last_t = 0.0
        self._last: dict[str, Any] = {}
        self._proc = None
        self._gpu = None
        self._pid = os.getpid()
        self._backends_ready = False
        if self._enabled:
            self._ensure_backends()

    def set_enabled(self, enabled: bool) -> None:
        """Toggle host CPU/GPU/RSS sampling (ExperimentSpec collect.system_metrics)."""
        self._enabled = bool(enabled)
        if self._enabled and not self._backends_ready:
            self._ensure_backends()
        if not self._enabled:
            self._last = {}
            self._last_t = 0.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _ensure_backends(self) -> None:
        if self._backends_ready:
            return
        self._backends_ready = True
        try:
            import psutil

            self._proc = psutil.Process()
            self._proc.cpu_percent(None)  # prime; first real sample needs a gap
        except Exception as exc:
            log.debug("psutil unavailable: %s", exc)
            self._proc = None
        try:
            from istream_player.utils.gpu_monitor import GPUMonitor

            mon = GPUMonitor(
                device_index=self._device_index,
                sample_interval=self._interval_s,
            )
            if mon.is_available:
                self._gpu = mon
        except Exception as exc:
            log.debug("GPUMonitor unavailable: %s", exc)
            self._gpu = None

    def sample(self) -> dict[str, Any]:
        if not self._enabled:
            return {}
        self._ensure_backends()
        now = time.monotonic()
        if self._last and (now - self._last_t) < self._interval_s:
            return dict(self._last)
        out: dict[str, Any] = {}
        if self._proc is not None:
            try:
                import psutil

                # Process.cpu_percent is relative to one logical core and can exceed 100%
                # on multi-core hosts. Normalize to whole-machine share for Lab/HUD.
                raw = float(self._proc.cpu_percent(None))
                ncpu = max(1, int(psutil.cpu_count(logical=True) or 1))
                out["cpu_pct_raw"] = round(raw, 1)
                out["cpu_pct"] = round(min(100.0, raw / ncpu), 1)
                out["cpu_logical"] = ncpu
                try:
                    out["cpu_system_pct"] = round(float(psutil.cpu_percent(None)), 1)
                except Exception:
                    pass
                rss_mb = float(self._proc.memory_info().rss) / (1024 * 1024)
                out["rss_mb"] = round(rss_mb, 1)
                try:
                    total_mb = float(psutil.virtual_memory().total) / (1024 * 1024)
                    out["mem_total_mb"] = round(total_mb, 1)
                    if total_mb > 0:
                        out["mem_pct"] = round(min(100.0, 100.0 * rss_mb / total_mb), 1)
                except Exception:
                    pass
            except Exception as exc:
                log.debug("process sample failed: %s", exc)
        if self._gpu is not None:
            try:
                st = self._gpu.get_state()
            except Exception as exc:
                log.debug("gpu sample failed: %s", exc)
                st = None
            if st:
                util = st.get("util_pct")
                if util is not None:
                    out["gpu_util_pct"] = round(float(util), 1)
                used = st.get("mem_used_mb")
                total = st.get("mem_total_mb")
                if used is not None:
                    out["gpu_mem_used_mb"] = round(float(used), 1)
                if total is not None:
                    out["gpu_mem_total_mb"] = round(float(total), 1)
                if used is not None and total is not None and float(total) > 0:
                    out["gpu_mem_pct"] = round(min(100.0, 100.0 * float(used) / float(total)), 1)
        # Process VRAM is independent of device-level NVML (torch fallback on WDDM).
        proc_mb, proc_src = self._process_gpu_mem()
        # Ignore near-zero torch reserved readings from idle/non-CUDA contexts.
        if proc_mb is not None and proc_mb > 1.0 and proc_src:
            out["gpu_proc_mem_mb"] = round(proc_mb, 1)
            out["gpu_proc_mem_src"] = proc_src
            total = out.get("gpu_mem_total_mb")
            if total and float(total) > 0:
                out["gpu_proc_mem_pct"] = round(min(100.0, 100.0 * proc_mb / float(total)), 1)
        self._last = out
        self._last_t = now
        return dict(out)

    def _process_gpu_mem(self) -> tuple[Optional[float], Optional[str]]:
        """This process's GPU memory + source label.

        Prefer NVML per-PID (driver-reported). Fall back to
        ``torch.cuda.memory_reserved`` (CUDA allocator only; excludes
        non-torch / driver-side allocations such as PyNvCodec).
        """
        mb = self._nvml_process_mem_mb()
        if mb is not None:
            return mb, "nvml"
        mb = self._torch_process_mem_mb()
        if mb is not None:
            return mb, "torch"
        return None, None

    def _nvml_process_mem_mb(self) -> Optional[float]:
        if self._gpu is None or not getattr(self._gpu, "_handle", None):
            return None
        try:
            import pynvml

            handle = self._gpu._handle
            procs = []
            for getter in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
                "nvmlDeviceGetGraphicsRunningProcesses_v3",
                "nvmlDeviceGetGraphicsRunningProcesses_v2",
                "nvmlDeviceGetGraphicsRunningProcesses",
            ):
                fn = getattr(pynvml, getter, None)
                if fn is None:
                    continue
                try:
                    procs = list(fn(handle) or [])
                    if procs:
                        break
                except Exception:
                    continue
            total = 0.0
            found = False
            for proc in procs:
                if int(getattr(proc, "pid", -1)) != self._pid:
                    continue
                used = getattr(proc, "usedGpuMemory", None)
                if used is None:
                    continue
                # NVML uses a sentinel when memory is unavailable.
                if int(used) < 0 or int(used) >= 2**63:
                    continue
                total += float(used) / (1024 * 1024)
                found = True
            return total if found else None
        except Exception as exc:
            log.debug("nvml process mem failed: %s", exc)
            return None

    def _torch_process_mem_mb(self) -> Optional[float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            # reserved ≈ allocator footprint for this process (more stable than allocated)
            return float(torch.cuda.memory_reserved(self._device_index)) / (1024 * 1024)
        except Exception as exc:
            log.debug("torch process mem failed: %s", exc)
            return None
