"""RuntimeMetrics process-VRAM source selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from istream_player.modules.status.runtime_metrics import RuntimeMetrics


def _bare() -> RuntimeMetrics:
    m = RuntimeMetrics.__new__(RuntimeMetrics)
    m._interval_s = 1.0
    m._device_index = 0
    m._enabled = True
    m._backends_ready = True
    m._last_t = 0.0
    m._last = {}
    m._proc = None
    m._gpu = None
    m._pid = 1
    return m


def test_process_gpu_mem_prefers_nvml():
    m = _bare()
    m._nvml_process_mem_mb = lambda: 256.0  # type: ignore[method-assign]
    m._torch_process_mem_mb = lambda: 128.0  # type: ignore[method-assign]
    mb, src = m._process_gpu_mem()
    assert mb == 256.0
    assert src == "nvml"


def test_process_gpu_mem_falls_back_to_torch():
    m = _bare()
    m._nvml_process_mem_mb = lambda: None  # type: ignore[method-assign]
    m._torch_process_mem_mb = lambda: 512.5  # type: ignore[method-assign]
    mb, src = m._process_gpu_mem()
    assert mb == 512.5
    assert src == "torch"


def test_process_gpu_mem_none_when_both_missing():
    m = _bare()
    m._nvml_process_mem_mb = lambda: None  # type: ignore[method-assign]
    m._torch_process_mem_mb = lambda: None  # type: ignore[method-assign]
    mb, src = m._process_gpu_mem()
    assert mb is None
    assert src is None


def test_sample_emits_src_without_device_monitor():
    """Torch fallback must work even when NVML/GPUMonitor is unavailable."""
    m = _bare()
    m._nvml_process_mem_mb = lambda: None  # type: ignore[method-assign]
    m._torch_process_mem_mb = lambda: 128.0  # type: ignore[method-assign]
    out = m.sample()
    assert out["gpu_proc_mem_mb"] == 128.0
    assert out["gpu_proc_mem_src"] == "torch"
    assert "gpu_util_pct" not in out


def test_sample_drops_near_zero_proc_mem():
    m = _bare()
    m._nvml_process_mem_mb = lambda: None  # type: ignore[method-assign]
    m._torch_process_mem_mb = lambda: 0.4  # type: ignore[method-assign]
    out = m.sample()
    assert "gpu_proc_mem_mb" not in out
    assert "gpu_proc_mem_src" not in out


def test_disabled_runtime_returns_empty():
    m = _bare()
    m._enabled = True
    m._backends_ready = True
    m._nvml_process_mem_mb = lambda: 256.0  # type: ignore[method-assign]
    m._torch_process_mem_mb = lambda: None  # type: ignore[method-assign]
    m.set_enabled(False)
    assert m.sample() == {}
    assert m.enabled is False
