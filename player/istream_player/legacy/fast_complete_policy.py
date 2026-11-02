from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque


@dataclass
class UCBConfig:
    """
    风险约束（chance constraint）相关配置：
    - delta: 允许"错过deadline"的概率上界（越小越保守）
    - waste_budget_s: 基础waste_budget，表示允许的最大"额外开销"时间（秒）
    - waste_budget_cap_s: waste_budget的上限保护值（秒）
    - check_interval_min: 检查间隔的最小帧数
    - check_interval_max: 检查间隔的最大帧数
    """
    delta: float = 0.2
    waste_budget_s: float = 0.2
    waste_budget_cap_s: float = 0.3
    check_interval_min: int = 10
    check_interval_max: int = 30


class SlidingWindowStats:
    """
    简单滑动窗口统计：均值、方差（Welford 适合在线，但这里窗口化更稳）。
    优化：添加缓存机制避免重复计算
    """
    def __init__(self, window_size: int = 30):
        self._window_size = max(int(window_size), 1)
        self._buf: Deque[float] = deque(maxlen=self._window_size)

        # 缓存机制：避免重复计算统计量
        self._cache_valid = False
        self._cached_mean: Optional[float] = None
        self._cached_std: Optional[float] = None
        self._cached_sorted: Optional[List[float]] = None

    def add(self, x: float) -> None:
        if x is None:
            return
        if x < 0:
            return
        self._buf.append(float(x))
        self._cache_valid = False  # 新数据加入，缓存失效

    @property
    def n(self) -> int:
        return len(self._buf)

    def _update_cache(self) -> None:
        """更新所有缓存的统计量"""
        if not self._buf:
            self._cached_mean = None
            self._cached_std = None
            self._cached_sorted = None
            self._cache_valid = True
            return

        n = len(self._buf)

        # 计算均值
        self._cached_mean = sum(self._buf) / n

        # 计算标准差
        if n < 2:
            self._cached_std = 0.0
        else:
            var = sum((x - self._cached_mean) ** 2 for x in self._buf) / (n - 1)
            self._cached_std = math.sqrt(max(var, 0.0))

        # 计算排序结果（用于分位数）
        self._cached_sorted = sorted(self._buf)

        self._cache_valid = True

    def mean_std(self) -> Tuple[Optional[float], Optional[float]]:
        if not self._cache_valid:
            self._update_cache()
        return self._cached_mean, self._cached_std

    def quantile(self, q: float) -> Optional[float]:
        """
        经验分位数（nearest-rank，不做插值；窗口较小时也够用）
        优化：使用缓存的排序结果
        q in [0, 1]
        """
        if not self._cache_valid:
            self._update_cache()

        if self._cached_sorted is None or not self._cached_sorted:
            return None

        q = min(max(float(q), 0.0), 1.0)
        xs = self._cached_sorted  # 使用缓存的排序结果
        k = int(math.ceil(q * len(xs))) - 1
        k = max(0, min(len(xs) - 1, k))
        return float(xs[k])


class RiskAwareFastCompletePolicy:
    """
    风险约束 fast-complete 策略：
    - 用在线观测的帧级耗时构造上置信界（UCB），近似控制 miss-deadline 概率 <= delta
    - 用 waste_budget 推导自适应检查间隔，waste_budget 会根据实际检查开销动态调整
    """
    def __init__(
        self,
        config: UCBConfig,
        default_enhance_time_s: float,
        stats_window: int = 30,
    ):
        self.cfg = config
        self.enh_stats = SlidingWindowStats(window_size=stats_window)

        # 当统计还不稳定时的先验（来自 latency_table/经验值）
        self.default_enhance_time_s = max(float(default_enhance_time_s), 1e-6)
        # 检查时间测量：跟踪每次检查的实际计算开销
        self.check_time_stats = SlidingWindowStats(window_size=10)
        self.measured_check_cost_s = 0.005  # 初始估计：每次检查约0.00005秒（基于测量学习）

        # 最近一次推导结果，便于日志/可视化
        self.last: Dict[str, float] = {}

    def _beta(self) -> float:
        # 正态/次高斯下常见的 UCB 系数（论文里可由 Hoeffding/Bernstein 推导）
        d = float(self.cfg.delta)
        d = min(max(d, 1e-6), 0.5)
        return math.sqrt(2.0 * math.log(1.0 / d))

    def update_observation(self, *, enhanced_dt_s: Optional[float] = None) -> None:
        self.enh_stats.add(enhanced_dt_s)

    def update_check_cost_measurement(self, check_time_s: float) -> None:
        """
        更新检查时间测量数据，并缓慢调整估计值
        """
        if check_time_s is None or check_time_s <= 0:
            return

        self.check_time_stats.add(check_time_s)

        # 当有足够测量数据时，更新测量值（指数移动平均）
        if self.check_time_stats.n >= 3:
            recent_mean = self.check_time_stats.mean_std()[0]
            if recent_mean is not None and recent_mean > 0:
                # 缓慢更新：90% 历史测量 + 10% 新测量
                self.measured_check_cost_s = 0.9 * self.measured_check_cost_s + 0.1 * recent_mean

    def _ucb_per_frame(self, mean: Optional[float], std: Optional[float], default: float) -> float:
        if mean is None:
            return default
        if std is None:
            std = 0.0
        # 经验上给一个很小的 std floor，避免早期 std=0 导致过于激进
        std_floor = 0.05 * mean
        std = max(std, std_floor)
        return max(mean + self._beta() * std, 0.0)

    def _ucb_per_frame_empirical(self, stats: SlidingWindowStats, default: float) -> float:
        """
        计算单帧耗时上界
        
        用经验分位数作为 per-frame 上界：
        - 样本足够时（>=10）使用 q=1-delta
        - 否则回退到均值+beta*std
        """
        if stats.n >= 10:
            q = 1.0 - float(self.cfg.delta)
            v = stats.quantile(q)
            if v is not None:
                return max(v, 0.0)
        mean, std = stats.mean_std()
        return self._ucb_per_frame(mean, std, default)


    def compute_all_stats(self, *, remaining_enhanced_frames: int, slack_s: float) -> Dict[str, float]:
        """
        批量计算所有需要的统计量，避免重复计算
        返回包含所有决策所需统计的字典
        """
        slack_s = max(float(slack_s), 0.0)
        rem_enh = int(remaining_enhanced_frames)

        # 一次性计算 u_enh（这是最耗时的操作）
        u_enh = self._ucb_per_frame_empirical(self.enh_stats, self.default_enhance_time_s)


        # 计算检查间隔
        rem_total = max(rem_enh, 1)
        u_avg = (rem_enh * u_enh) / rem_total
        u_avg = max(u_avg, 1e-6)

        # ========== 计算检查间隔 ==============
        # waste_budget = float(self.cfg.waste_budget_s)

        # # # 如果有足够的检查时间测量数据，用实际开销调整
        # # if self.check_time_stats.n >= 3:
        # #     min_waste_budget = self.measured_check_cost_s * 4  # 支持4次检查
        # #     waste_budget = max(waste_budget, min_waste_budget)

        # # 上限保护
        # waste_budget = min(waste_budget, float(self.cfg.waste_budget_cap_s))


        # k = int(waste_budget / u_avg)
        # k = max(self.cfg.check_interval_min, min(self.cfg.check_interval_max, k))
        k = 15

        # 计算剩余时间上界
        u_remain = rem_enh * u_enh
        # 判断是否需要触发
        triggered = u_remain >= slack_s

        # 更新缓存
        self.last.update({
            "slack_s": slack_s,
            "u_enh": u_enh,
            "u_remain": u_remain,
            "k": float(k),
            "n_enh_obs": float(self.enh_stats.n),
        })

        return {
            "slack_s": slack_s,
            "u_enh": u_enh,
            "u_remain": u_remain,
            "check_k": k,
            "triggered": triggered,
            "n_enh_obs": float(self.enh_stats.n),
        }


