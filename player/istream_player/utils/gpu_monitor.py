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
                    # 功耗（毫瓦）
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                    self._power_samples.append(power_mw / 1000.0)  # 转换为瓦
                    # 温度（摄氏度）
                    temp = pynvml.nvmlDeviceGetTemperature(self._handle, pynvml.NVML_TEMPERATURE_GPU)
                    self._temp_samples.append(float(temp))
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

        return {
            "avg_power_w": round(avg_power, 2),
            "peak_power_w": round(peak_power, 2),
            "total_energy_kj": round(total_energy_kj, 4),
            "avg_temp_c": round(avg_temp, 2) if avg_temp is not None else None,
            "peak_temp_c": round(peak_temp, 2) if peak_temp is not None else None,
            "sample_count": len(self._power_samples),
            "duration_seconds": round(duration, 2),
        }

    @property
    def is_available(self) -> bool:
        """pynvml 是否可用"""
        return self._nvml_available
