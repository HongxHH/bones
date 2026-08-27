from typing import Dict

from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.module import Module, ModuleOption
from istream_player.models.mpd_objects import AdaptationSet


# 基于带宽的自适应比特率控制器实现类 - 继承自Module和ABRController接口
# 负责根据网络带宽测量结果选择最优的视频质量表示，实现自适应比特率控制
@ModuleOption("bandwidth", requires=[BandwidthMeter])
class BandwidthABRController(Module, ABRController):
    def __init__(self):
        # 调用父类构造函数，初始化模块基础功能
        super().__init__()

    # 设置方法 - 初始化ABR控制器的配置参数和依赖组件
    async def setup(self, config: PlayerConfig, bandwidth_meter: BandwidthMeter):
        # 保存带宽测量器实例的引用，用于获取当前网络带宽信息
        self.bandwidth_meter = bandwidth_meter

    # 更新表示选择的方法 - ABR控制器的核心方法，根据带宽选择最优质量
    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Dict[int, int]:
        # 注释掉的保守带宽使用策略 - 只使用70%的测量带宽
        # 这种策略可以避免网络波动导致的卡顿，但可能无法充分利用可用带宽
        # available_bandwidth = int(self.bandwidth_meter.bandwidth * 0.7)
        # 使用100%的测量带宽，最大化利用可用网络资源
        # 将浮点数带宽转换为整数，用于后续计算
        available_bandwidth = int(self.bandwidth_meter.bandwidth)

        # 统计视频和音频自适应集的数量，用于带宽分配
        # 视频自适应集数量，用于计算视频流分配的带宽
        num_videos = 0
        # 音频自适应集数量，用于计算音频流分配的带宽
        num_audios = 0
        # 遍历所有自适应集，统计不同类型的内容
        for adaptation_set in adaptation_sets.values():
            # 检查内容类型是否为视频
            if adaptation_set.content_type == "video":
                # 增加视频自适应集计数
                num_videos += 1
            else:
                # 增加音频自适应集计数（假设非视频即为音频）
                num_audios += 1

        # 计算理想的选择结果 - 根据内容类型进行不同的带宽分配策略
        # 检查是否存在视频或音频自适应集
        if num_videos == 0 or num_audios == 0:
            # 如果只有一种类型的内容，平均分配带宽
            # 计算每个自适应集可分配的带宽
            bw_per_adaptation_set = available_bandwidth / (num_videos + num_audios)
            # 初始化理想选择结果字典
            ideal_selection: Dict[int, int] = dict()

            # 为每个自适应集选择最优表示
            for adaptation_set in adaptation_sets.values():
                # 调用带宽选择方法，为当前自适应集选择最优表示
                ideal_selection[adaptation_set.id] = self.choose_ideal_selection_bandwidth_based(
                    adaptation_set, bw_per_adaptation_set
                )
        else:
            # 如果同时存在视频和音频，使用不同的带宽分配比例
            # 视频流分配80%的带宽，因为视频通常需要更多带宽
            bw_per_video = (available_bandwidth * 0.8) / num_videos
            # 音频流分配20%的带宽，因为音频相对需要较少带宽
            bw_per_audio = (available_bandwidth * 0.2) / num_audios
            # 初始化理想选择结果字典
            ideal_selection: Dict[int, int] = dict()
            # 为每个自适应集选择最优表示
            for adaptation_set in adaptation_sets.values():
                # 根据内容类型选择相应的带宽分配
                if adaptation_set.content_type == "video":
                    # 为视频自适应集分配视频带宽
                    ideal_selection[adaptation_set.id] = self.choose_ideal_selection_bandwidth_based(adaptation_set, bw_per_video)
                else:
                    # 为音频自适应集分配音频带宽
                    ideal_selection[adaptation_set.id] = self.choose_ideal_selection_bandwidth_based(adaptation_set, bw_per_audio)

        # 返回理想的选择结果
        return ideal_selection

    # 基于带宽选择理想表示的静态方法 - 为单个自适应集选择最优质量
    @staticmethod
    def choose_ideal_selection_bandwidth_based(adaptation_set: AdaptationSet, bw) -> int:
        """
        为单个自适应集选择理想的比特率表示，不考虑缓冲水位或其他因素
        参数
        ----------
        adaptation_set
            要选择表示的自适应集
        bw
            可分配给此自适应集的带宽
        返回
        -------
        id: int
            表示ID
        """
        # 按带宽从高到低排序所有表示，便于选择最优质量
        # 使用lambda函数按bandwidth字段降序排序
        representations = sorted(adaptation_set.representations.values(), key=lambda x: x.bandwidth, reverse=True)

        # 遍历排序后的表示列表，选择第一个带宽小于可用带宽的表示
        for representation in representations:
            # 检查当前表示的带宽是否小于可用带宽
            if representation.bandwidth < bw:
                # 返回第一个满足条件的表示ID
                return representation.id
        # 如果没有找到带宽小于可用带宽的表示，返回最低质量的表示
        # 这确保即使在带宽不足的情况下也能选择到可用的表示
        return representations[-1].id
