from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
from collections import deque


@dataclass
class UCBConfig:
    """
    风险约束（chance constraint）相关配置：
    - delta: 允许“错过deadline”的概率上界（越小越保守）
    - waste_budget_s: 单次检查间隔允许的“额外浪费计算时间”上界，具体来说是指，在slack_s的范围内，最多可以浪费的计算时间
    - slack_waste_ratio: slack_s的范围内，最多可以浪费的计算时间与slack_s的比值
    - waste_budget_cap_s: 单次检查间隔允许的“额外浪费计算时间”上界的最大值
    - check_interval_min: 单次检查间隔的最小值
    - check_interval_max: 单次检查间隔的最大值
    """
    delta: float = 0.2
    waste_budget_s: float = 0.08
    # slack 相关的动态放宽：slack 越大，允许更大的检查间隔浪费预算（避免 check_k 过短）
    slack_waste_ratio: float = 0.05
    waste_budget_cap_s: float = 0.5
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
    - 用 waste_budget 推导自适应检查间隔，给出“额外浪费计算时间”上界
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

        # 最近一次推导结果，便于日志/可视化
        self.last: Dict[str, float] = {}

    def _beta(self) -> float:
        # 正态/次高斯下常见的 UCB 系数（论文里可由 Hoeffding/Bernstein 推导）
        d = float(self.cfg.delta)
        d = min(max(d, 1e-6), 0.5)
        return math.sqrt(2.0 * math.log(1.0 / d))

    def update_observation(self, *, enhanced_dt_s: Optional[float] = None) -> None:
        self.enh_stats.add(enhanced_dt_s)

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


    def estimate_remaining_ucb(self, *, remaining_enhanced_frames: int, ) -> float:
        """
        估计剩余增强时间上界
        """
        u_enh = self._ucb_per_frame_empirical(self.enh_stats, self.default_enhance_time_s)
        u_rem = remaining_enhanced_frames * u_enh 
        self.last.update(
            {
                "u_enh": u_enh,
                "u_remain": u_rem,
                "n_enh_obs": float(self.enh_stats.n),
            }
        )
        return u_rem

    def should_fast_complete(self, *, slack_s: float, remaining_enhanced_frames: int) -> bool:
        """
        根据 slack 和剩余增强帧数，判断是否需要快速完成
        """
        slack_s = max(float(slack_s), 0.0)
        u_rem = self.estimate_remaining_ucb(
            remaining_enhanced_frames=remaining_enhanced_frames,
        )
        self.last.update({"slack_s": slack_s})
        return u_rem >= slack_s

    def suggest_check_interval_frames(self, *, remaining_enhanced_frames: int, slack_s: Optional[float] = None) -> int:
        """
        根据 waste_budget 推导检查间隔：保证“下次检查前最多浪费的计算时间”<= waste_budget_s。
        """
        # 用“未来每一帧平均最坏代价”近似：按剩余帧组成做一次加权
        u_enh = self._ucb_per_frame_empirical(self.enh_stats, self.default_enhance_time_s)
        rem_total = max(int(remaining_enhanced_frames), 1)
        u_avg = (remaining_enhanced_frames * u_enh ) / rem_total
        u_avg = max(u_avg, 1e-6)

        # 根据 slack 动态放宽 waste budget：slack 越大，允许更长的检查间隔（上限封顶）
        waste_budget = float(self.cfg.waste_budget_s)
        if slack_s is not None:
            slack_s = max(float(slack_s), 0.0)
            waste_budget = max(waste_budget, slack_s * float(self.cfg.slack_waste_ratio))
        waste_budget = min(waste_budget, float(self.cfg.waste_budget_cap_s))

        k = int(waste_budget / u_avg)
        k = max(self.cfg.check_interval_min, min(self.cfg.check_interval_max, k))
        self.last.update({"u_avg": u_avg, "k": float(k), "waste_budget_s": float(waste_budget)})
        return k

    def compute_all_stats(self, *, remaining_enhanced_frames: int, slack_s: float) -> Dict[str, float]:
        """
        批量计算所有需要的统计量，避免重复计算
        返回包含所有决策所需统计的字典
        """
        slack_s = max(float(slack_s), 0.0)
        rem_enh = int(remaining_enhanced_frames)

        # 一次性计算 u_enh（这是最耗时的操作）
        u_enh = self._ucb_per_frame_empirical(self.enh_stats, self.default_enhance_time_s)

        # 计算剩余时间上界
        u_remain = rem_enh * u_enh

        # 计算检查间隔
        rem_total = max(rem_enh, 1)
        u_avg = (rem_enh * u_enh) / rem_total
        u_avg = max(u_avg, 1e-6)

        waste_budget = float(self.cfg.waste_budget_s)
        waste_budget = max(waste_budget, slack_s * float(self.cfg.slack_waste_ratio))
        waste_budget = min(waste_budget, float(self.cfg.waste_budget_cap_s))

        k = int(waste_budget / u_avg)
        k = max(self.cfg.check_interval_min, min(self.cfg.check_interval_max, k))

        # 判断是否需要触发
        triggered = u_remain >= slack_s

        # 更新缓存
        self.last.update({
            "slack_s": slack_s,
            "u_enh": u_enh,
            "u_remain": u_remain,
            "u_avg": u_avg,
            "k": float(k),
            "waste_budget_s": waste_budget,
            "n_enh_obs": float(self.enh_stats.n),
        })

        return {
            "slack_s": slack_s,
            "u_enh": u_enh,
            "u_remain": u_remain,
            "u_avg": u_avg,
            "check_k": k,
            "waste_budget_s": waste_budget,
            "triggered": 1.0 if triggered else 0.0,
            "n_enh_obs": float(self.enh_stats.n),
        }


