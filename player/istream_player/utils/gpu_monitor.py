"""
GPU 功耗与温度监控模块
使用 pynvml 库在后台线程中定期采样 GPU 功耗和温度。
若 pynvml 不可用或非 NVIDIA GPU，所有方法静默返回 None。
"""
import logging
import threading
import time
from typing import Any, Dict, Optional

log = logging.getLogger("GPUMonitor")


class GPUMonitor:
    """GPU 功耗与温度监控器（后台线程采样）"""

    def __init__(self, device_index: int = 0, sample_interval: float = 0.5):
        """
        Args:
            device_index: GPU 设备索引
            sample_interval: 采样间隔（秒）
        """
        self.device_index = device_index
        self.sample_interval = sample_interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False

        # 采样数据
        self._power_samples: list = []
        self._temp_samples: list = []
        self._util_samples: list = []
        self._mem_samples: list = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        # pynvml 句柄
        self._nvml_available = False
        self._handle = None
        self._init_nvml()

    def _init_nvml(self):
        """初始化 pynvml"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._nvml_available = True
            log.info(f"GPUMonitor initialized for device {self.device_index}")
        except Exception as e:
            self._nvml_available = False
            log.debug(f"pynvml not available: {e}")

    def start(self):
        """启动后台监控线程"""
        if not self._nvml_available:
            return
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._start_time = time.perf_counter()
        self._power_samples = []
        self._temp_samples = []
        self._util_samples = []
        self._mem_samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        log.info("GPUMonitor started")

    def stop(self) -> Optional[Dict[str, Any]]:
        """停止监控并返回统计结果"""
        if not self._nvml_available or not self._running:
            return None
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        self._end_time = time.perf_counter()
        result = self._compute_stats()
        log.info(f"GPUMonitor stopped: {result}")
        return result

    def _sample_loop(self):
        """后台采样循环"""
        try:
            import pynvml
            while not self._stop_event.is_set():
                try:
                    import pynvml
                    # 功耗（毫瓦）
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                    self._power_samples.append(power_mw / 1000.0)  # 转换为瓦
                    # 温度（摄氏度）
                    temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                    self._temp_samples.append(float(temp))
                    # 利用率（GPU核心 %）
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self._util_samples.append(float(util.gpu))
                    # 显存（字节 -> MB）
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                    self._mem_samples.append({
                        'used_mb': mem.used / 1024 / 1024,
                        'total_mb': mem.total / 1024 / 1024,
                        'free_mb': mem.free / 1024 / 1024,
                    })
                except Exception as e:
                    log.debug(f"GPU sample error: {e}")
                self._stop_event.wait(self.sample_interval)
        except Exception as e:
            log.error(f"GPUMonitor sample loop failed: {e}")

    def _compute_stats(self) -> Optional[Dict[str, Any]]:
        """计算统计结果"""
        if not self._power_samples:
            return None

        duration = (self._end_time - self._start_time) if (self._end_time and self._start_time) else 0
        avg_power = sum(self._power_samples) / len(self._power_samples)
        peak_power = max(self._power_samples)
        # 总能耗（千焦）= 平均功率(W) × 时间(s) / 1000
        total_energy_kj = (avg_power * duration) / 1000.0 if duration > 0 else 0

        avg_temp = sum(self._temp_samples) / len(self._temp_samples) if self._temp_samples else None
        peak_temp = max(self._temp_samples) if self._temp_samples else None

        avg_util = sum(self._util_samples) / len(self._util_samples) if self._util_samples else None
        peak_util = max(self._util_samples) if self._util_samples else None

        if self._mem_samples:
            avg_mem_used = sum(s['used_mb'] for s in self._mem_samples) / len(self._mem_samples)
            peak_mem_used = max(s['used_mb'] for s in self._mem_samples)
            total_mem = self._mem_samples[0]['total_mb']
        else:
            avg_mem_used = peak_mem_used = total_mem = None

        return {
            "avg_power_w": round(avg_power, 2),
            "peak_power_w": round(peak_power, 2),
            "total_energy_kj": round(total_energy_kj, 4),
            "avg_temp_c": round(avg_temp, 2) if avg_temp is not None else None,
            "peak_temp_c": round(peak_temp, 2) if peak_temp is not None else None,
            "avg_util_pct": round(avg_util, 1) if avg_util is not None else None,
            "peak_util_pct": round(peak_util, 1) if peak_util is not None else None,
            "avg_mem_used_mb": round(avg_mem_used, 1) if avg_mem_used is not None else None,
            "peak_mem_used_mb": round(peak_mem_used, 1) if peak_mem_used is not None else None,
            "total_mem_mb": round(total_mem, 1) if total_mem is not None else None,
            "sample_count": len(self._power_samples),
            "duration_seconds": round(duration, 2),
        }

    def get_state(self) -> Optional[Dict[str, Any]]:
        """实时查询当前 GPU 状态（单次采样，不依赖后台线程）"""
        if not self._nvml_available:
            return None
        try:
            import pynvml
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return {
                "power_w": round(power_mw / 1000.0, 2),
                "temp_c": float(temp),
                "util_pct": float(util.gpu),
                "mem_used_mb": round(mem.used / 1024 / 1024, 1),
                "mem_total_mb": round(mem.total / 1024 / 1024, 1),
                "mem_free_mb": round(mem.free / 1024 / 1024, 1),
            }
        except Exception as e:
            log.debug(f"get_state error: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """pynvml 是否可用"""
        return self._nvml_available
