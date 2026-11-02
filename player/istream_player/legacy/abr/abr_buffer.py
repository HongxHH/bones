from collections import OrderedDict
from typing import Dict, Optional

from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet


# 基于缓冲区的自适应比特率控制器实现类 - 继承自Module和ABRController接口
# 负责根据播放缓冲区水位选择最优的视频质量表示，实现基于缓冲区的自适应比特率控制
@ModuleOption("buffer", requires=[BufferManager])
class BufferABRController(Module, ABRController):
    def __init__(self):
        # 速率映射表，用于存储缓冲区水位与比特率的对应关系
        # 使用OrderedDict保持插入顺序，便于线性插值计算
        self.rate_map = None

        # 缓冲区低水位阈值（10%），当缓冲区水位低于此值时选择最低比特率
        # 这个阈值确保在缓冲区不足时优先保证播放连续性
        self.RESERVOIR = 0.1
        # 缓冲区高水位阈值（90%），当缓冲区水位高于此值时选择最高比特率
        # 这个阈值确保在缓冲区充足时充分利用可用带宽
        self.UPPER_RESERVOIR = 0.9

    # 设置方法 - 初始化ABR控制器的配置参数和依赖组件
    async def setup(self, config: PlayerConfig, buffer_manager: BufferManager, **kwargs):
        # 从配置中获取缓冲区大小（秒），用于计算缓冲区水位百分比
        self.buffer_size = config.buffer_duration
        # 保存缓冲区管理器实例的引用，用于获取当前缓冲区水位
        self.buffer_manager = buffer_manager

    # 更新表示选择的方法 - ABR控制器的核心方法，根据缓冲区水位选择最优质量
    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Dict[int, int]:
        # 初始化最终选择结果字典，存储每个自适应集的选择结果
        final_selections = dict()

        # 遍历所有自适应集，为每个自适应集选择最优表示
        for adaptation_set in adaptation_sets.values():
            # 调用基于缓冲区的选择方法，为当前自适应集选择最优表示
            final_selections[adaptation_set.id] = self.choose_ideal_selection_buffer_based(adaptation_set)

        # 返回所有自适应集的选择结果
        return final_selections

    # 基于缓冲区选择理想表示的方法 - 为单个自适应集选择最优质量
    def choose_ideal_selection_buffer_based(self, adaptation_set) -> Optional[int]:
        """
        基于速率映射估计下一个比特率的模块
        速率映射：缓冲区占用率 vs 比特率：
            如果缓冲区占用率 < 低水位阈值（10%）：
                选择最低比特率
            如果低水位阈值 < 缓冲区占用率 < 高水位阈值（90%）：
                基于速率映射的线性函数
            如果缓冲区占用率 > 高水位阈值：
                选择最高比特率
        参考论文[1]的图6
        参数
        ----------
        adaptation_set: AdaptationSet
            要选择表示的自适应集
        返回
        -------
        representation_id: int
            下一个段的比特率对应的表示ID
        """
        # 初始化下一个比特率变量
        next_bitrate = None

        # 提取所有可用的比特率，并转换为列表
        # 从自适应集的所有表示中提取带宽信息
        bitrates = [representation.bandwidth for representation in adaptation_set.representations.values()]
        # 对比特率进行升序排序，便于后续的线性插值计算
        bitrates.sort()

        # 计算当前缓冲区占用率百分比
        # 获取当前缓冲区水位（秒）
        current_buffer_occupancy = self.buffer_manager.buffer_level
        # 计算缓冲区占用率百分比（0-1之间）
        buffer_percentage = current_buffer_occupancy / self.buffer_size

        # 基于速率映射选择下一个比特率
        # 如果速率映射表未初始化，则创建速率映射表
        if self.rate_map is None:
            self.rate_map = self.get_rate_map(bitrates)

        # 根据缓冲区水位百分比选择比特率
        if buffer_percentage <= self.RESERVOIR:
            # 如果缓冲区水位低于低水位阈值，选择最低比特率
            # 这确保在缓冲区不足时优先保证播放连续性
            next_bitrate = bitrates[0]
        elif buffer_percentage >= self.UPPER_RESERVOIR:
            # 如果缓冲区水位高于高水位阈值，选择最高比特率
            # 这确保在缓冲区充足时充分利用可用带宽
            next_bitrate = bitrates[-1]
        else:
            # 如果缓冲区水位在低水位和高水位之间，使用线性插值
            # 从高到低遍历速率映射表的标记点
            for marker in reversed(self.rate_map.keys()):
                # 找到第一个小于当前缓冲区百分比的标记点
                if marker < buffer_percentage:
                    break
                # 更新下一个比特率为当前标记点对应的比特率
                next_bitrate = self.rate_map[marker]

        # 根据选择的比特率查找对应的表示ID
        representation_id = None
        # 遍历所有表示，查找匹配的比特率
        for representation in adaptation_set.representations.values():
            # 如果找到匹配的比特率，记录对应的表示ID
            if representation.bandwidth == next_bitrate:
                representation_id = representation.id

        # 返回找到的表示ID
        return representation_id

    # 生成速率映射表的方法 - 为比特率、低水位和高水位生成映射关系
    def get_rate_map(self, bitrates):
        """
        为比特率、低水位和高水位生成速率映射的模块
        参数
        ----------
        bitrates: List[int]
            排序后的比特率列表
        返回
        -------
        rate_map: OrderedDict
            缓冲区水位百分比到比特率的映射表
        """
        # 创建有序字典，用于存储速率映射关系
        rate_map = OrderedDict()
        # 在低水位阈值处映射到最低比特率
        rate_map[self.RESERVOIR] = bitrates[0]
        # 获取中间级别的比特率（排除最低和最高比特率）
        intermediate_levels = bitrates[1:-1]
        # 计算标记点之间的间距，用于线性分布中间比特率
        marker_length = (self.UPPER_RESERVOIR - self.RESERVOIR) / (len(intermediate_levels) + 1)
        # 初始化当前标记点位置，从低水位阈值开始
        current_marker = self.RESERVOIR + marker_length
        # 为每个中间比特率创建映射点
        for bitrate in intermediate_levels:
            # 在当前标记点位置映射到当前比特率
            rate_map[current_marker] = bitrate
            # 移动到下一个标记点位置
            current_marker += marker_length
        # 在高水位阈值处映射到最高比特率
        rate_map[self.UPPER_RESERVOIR] = bitrates[-1]
        # 返回完整的速率映射表
        return rate_map
