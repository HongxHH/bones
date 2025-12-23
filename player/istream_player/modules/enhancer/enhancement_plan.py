"""
帧增强计划模块
提供可扩展的帧增强计划生成系统，支持基于间隔和基于复杂度的增强策略
"""
from dataclasses import dataclass
from typing import List, Optional, Iterator, Protocol
import numpy as np


@dataclass
class EnhancementDecision:
    """
    单帧增强决策数据类
    表示对某一帧的增强处理决策
    
    Attributes:
        should_enhance: 是否需要进行模型增强（True=增强，False=跳过或插值）
        enhance_level: 增强级别（1=低，2=中，3=高），仅在should_enhance=True时有效
        use_interpolation: 是否使用插值（True=插值，False=跳过处理），仅在should_enhance=False时有效
        frame_index: 帧索引（可选，用于调试和日志）
    """
    should_enhance: bool
    enhance_level: int = 1  # 默认低级别增强
    use_interpolation: bool = True  # 默认使用插值
    frame_index: Optional[int] = None  # 可选，用于调试
    
    def __post_init__(self):
        """验证决策的有效性"""
        if self.should_enhance and self.enhance_level not in [1, 2, 3]:
            raise ValueError(f"Invalid enhance_level: {self.enhance_level}, must be 1, 2, or 3")


class FrameEnhancementPlan(Protocol):
    """
    帧增强计划协议接口
    定义增强计划生成器的标准接口，便于未来扩展不同的计划策略
    """
    
    def generate_plan(self, num_frames: int, **kwargs) -> Iterator[EnhancementDecision]:
        """
        生成增强计划
        
        Args:
            num_frames: 段的总帧数
            **kwargs: 额外的策略参数（如复杂度列表、间隔配置等）
            
        Yields:
            EnhancementDecision: 每帧的增强决策
        """
        ...
    
    def get_plan_list(self, num_frames: int, **kwargs) -> List[EnhancementDecision]:
        """
        获取完整的增强计划列表（非生成器版本）
        
        Args:
            num_frames: 段的总帧数
            **kwargs: 额外的策略参数
            
        Returns:
            List[EnhancementDecision]: 完整的增强决策列表
        """
        return list(self.generate_plan(num_frames, **kwargs))


class IntervalBasedPlan:
    """
    基于间隔的增强计划生成器
    根据固定的帧间隔来决定哪些帧需要增强
    
    例如：interval=1 表示每隔1帧增强一次（即每2帧增强1帧）
         interval=0 表示所有帧都增强
    """
    
    def __init__(self, interval: int, default_enhance_level: int = 1, use_interpolation: bool = True):
        """
        初始化基于间隔的计划生成器
        
        Args:
            interval: 增强间隔（0=全帧增强，1=隔1帧，2=隔2帧...）
            default_enhance_level: 默认增强级别
            use_interpolation: 跳过帧是否使用插值
        """
        self.interval = interval
        self.default_enhance_level = default_enhance_level
        self.use_interpolation = use_interpolation
    
    def generate_plan(self, num_frames: int, **kwargs) -> Iterator[EnhancementDecision]:
        """
        生成基于间隔的增强计划
        
        Args:
            num_frames: 段的总帧数
            **kwargs: 忽略（为接口兼容性保留）
            
        Yields:
            EnhancementDecision: 每帧的增强决策
        """
        # 如果间隔为0或负数，所有帧都增强
        if self.interval <= 0:
            for i in range(num_frames):
                yield EnhancementDecision(
                    should_enhance=True,
                    enhance_level=self.default_enhance_level,
                    frame_index=i
                )
        else:
            # 根据间隔决定哪些帧增强
            for i in range(num_frames):
                should_enhance = (i % (self.interval + 1)) == 0
                yield EnhancementDecision(
                    should_enhance=should_enhance,
                    enhance_level=self.default_enhance_level if should_enhance else 1,
                    use_interpolation=self.use_interpolation if not should_enhance else True,
                    frame_index=i
                )
    
    def get_plan_list(self, num_frames: int, **kwargs) -> List[EnhancementDecision]:
        """获取完整的增强计划列表"""
        return list(self.generate_plan(num_frames, **kwargs))


class ComplexityBasedPlan:
    """
    基于复杂度的增强计划生成器（未来扩展）
    根据每帧的复杂度来决定增强策略
    
    复杂度高的帧使用高级别模型，复杂度低的帧使用低级别模型或跳过
    """
    
    def __init__(self, 
                 complexity_threshold_low: float = 0.3,
                 complexity_threshold_high: float = 0.7,
                 low_level: int = 1,
                 medium_level: int = 2,
                 high_level: int = 3,
                 use_interpolation: bool = True):
        """
        初始化基于复杂度的计划生成器
        
        Args:
            complexity_threshold_low: 低复杂度阈值，低于此值跳过增强
            complexity_threshold_high: 高复杂度阈值，高于此值使用高级别模型
            low_level: 低级别增强级别
            medium_level: 中级别增强级别
            high_level: 高级别增强级别
            use_interpolation: 跳过帧是否使用插值
        """
        self.complexity_threshold_low = complexity_threshold_low
        self.complexity_threshold_high = complexity_threshold_high
        self.low_level = low_level
        self.medium_level = medium_level
        self.high_level = high_level
        self.use_interpolation = use_interpolation
    
    def generate_plan(self, num_frames: int, complexity_list: Optional[List[float]] = None, **kwargs) -> Iterator[EnhancementDecision]:
        """
        生成基于复杂度的增强计划
        
        Args:
            num_frames: 段的总帧数
            complexity_list: 每帧的复杂度列表（0.0-1.0），长度应与num_frames一致
            **kwargs: 其他参数
            
        Yields:
            EnhancementDecision: 每帧的增强决策
            
        Raises:
            ValueError: 如果complexity_list长度与num_frames不匹配
        """
        for i, complexity in enumerate(complexity_list):
            if complexity < self.complexity_threshold_low:
                # 低复杂度：跳过增强
                yield EnhancementDecision(
                    should_enhance=False,
                    use_interpolation=self.use_interpolation,
                    frame_index=i
                )
            elif complexity < self.complexity_threshold_high:
                # 中等复杂度：使用中级别模型
                yield EnhancementDecision(
                    should_enhance=True,
                    enhance_level=self.medium_level,
                    frame_index=i
                )
            else:
                # 高复杂度：使用高级别模型
                yield EnhancementDecision(
                    should_enhance=True,
                    enhance_level=self.high_level,
                    frame_index=i
                )
    
    def get_plan_list(self, num_frames: int, complexity_list: Optional[List[float]] = None, **kwargs) -> List[EnhancementDecision]:
        """获取完整的增强计划列表"""
        return list(self.generate_plan(num_frames, complexity_list=complexity_list, **kwargs))

