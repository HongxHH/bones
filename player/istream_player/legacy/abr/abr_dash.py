import logging
from typing import Dict, Optional

from istream_player.config.config import PlayerConfig
from istream_player.core.abr import ABRController
from istream_player.core.buffer import BufferManager
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.models.mpd_objects import AdaptationSet


# DASH自适应比特率控制器实现类 - 继承自Module和ABRController接口
# 负责实现DASH标准的自适应比特率控制，结合带宽测量和缓冲区管理进行智能质量选择
@ModuleOption("dash", default=True, requires=[BandwidthMeter, BufferManager, MPDProvider])
class DashABRController(Module, ABRController):
    # 获取日志记录器实例，用于记录DASH ABR控制相关的日志信息
    log = logging.getLogger("DashABRController")

    # 设置方法 - 初始化DASH ABR控制器的配置参数和依赖组件
    async def setup(
        self,
        config: PlayerConfig,
        bandwidth_meter: BandwidthMeter,
        buffer_manager: BufferManager,
        mpd_provider: MPDProvider,
        **kwargs,
    ):
        # 保存缓冲区管理器实例的引用，用于获取当前缓冲区状态
        self.buffer_manager = buffer_manager
        # 保存MPD提供器实例的引用，用于获取MPD文件信息和段持续时间
        self.mpd_provider = mpd_provider
        # 从配置中获取恐慌缓冲区水位，当缓冲区低于此值时采用保守策略
        self.panic_buffer = config.panic_buffer_level
        # 从配置中获取安全缓冲区水位，当缓冲区高于此值时可以采用更激进的策略
        self.safe_buffer = config.safe_buffer_level
        # 保存带宽测量器实例的引用，用于获取当前网络带宽信息
        self.bandwidth_meter = bandwidth_meter

    def __init__(self):
        # 上次选择的表示ID字典，用于跟踪历史选择状态
        # 键为自适应集ID，值为选择的表示ID
        self._last_selections: Optional[Dict[int, int]] = None

    # 选择理想表示的静态方法 - 基于带宽为单个自适应集选择最优质量
    @staticmethod
    def choose_ideal_selection(adaptation_set, bw) -> int:
        """
        为单个自适应集选择理想的比特率表示，不考虑缓冲区水位或其他因素

        参数
        ----------
        adaptation_set: AdaptationSet
            要选择表示的自适应集
        bw: int
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

    # 更新表示选择的方法 - DASH ABR控制器的核心方法，结合带宽和缓冲区进行智能选择
    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Dict[int, int]:
        # 断言确保MPD文件已下载，这是进行质量选择的前提条件
        assert self.mpd_provider.mpd is not None, "MPD File not downloaded"

        # 使用70%的测量带宽，采用保守策略避免网络波动导致的卡顿
        # 这种策略在DASH标准中很常见，可以平衡带宽利用率和播放稳定性
        available_bandwidth = int(self.bandwidth_meter.bandwidth * 0.7)

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
                # 调用理想选择方法，为当前自适应集选择最优表示
                ideal_selection[adaptation_set.id] = self.choose_ideal_selection(adaptation_set, bw_per_adaptation_set)
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
                    ideal_selection[adaptation_set.id] = self.choose_ideal_selection(adaptation_set, bw_per_video)
                else:
                    # 为音频自适应集分配音频带宽
                    ideal_selection[adaptation_set.id] = self.choose_ideal_selection(adaptation_set, bw_per_audio)

        # 获取当前缓冲区水位，用于后续的缓冲区感知选择
        buffer_level = self.buffer_manager.buffer_level
        # 初始化最终选择结果字典
        final_selections = dict()

        # 将缓冲区水位纳入考虑范围 - 这是DASH ABR的核心特性
        # 检查是否存在上次选择的历史记录
        if self._last_selections is not None:
            # 为每个自适应集进行缓冲区感知的选择
            for id_, adaptation_set in adaptation_sets.items():
                # 获取当前自适应集的所有表示
                representations = adaptation_set.representations
                # 获取上次选择的表示对象
                last_repr = representations[self._last_selections.get(id_, 0)]
                # 获取理想选择的表示对象
                ideal_repr = representations[ideal_selection.get(id_, 0)]
                # 记录缓冲区状态信息到日志，用于调试和分析
                self.log.info(f"buffer_level={buffer_level}, panic_buffer={self.panic_buffer}")
                
                # 根据缓冲区水位采用不同的选择策略
                if buffer_level < self.panic_buffer:
                    # 如果缓冲区水位低于恐慌阈值，采用保守策略
                    # 选择上次选择和理想选择中带宽较低的那个，确保播放连续性
                    final_repr_id = last_repr.id if last_repr.bandwidth < ideal_repr.bandwidth else ideal_repr.id
                elif buffer_level > self.safe_buffer:
                    # 如果缓冲区水位高于安全阈值，可以考虑更激进的策略
                    # 检查上次选择的带宽是否高于理想选择
                    if last_repr.bandwidth > ideal_repr.bandwidth:
                        # 如果上次选择带宽更高，需要计算下载时间来决定是否降级
                        if adaptation_set.content_type == "video":
                            # 为视频流计算每视频的带宽分配
                            bw_per_video = (available_bandwidth * 0.8) / num_videos
                            # 计算下一个段的下载时间
                            # 公式：下载时间 = (当前带宽 + 理想带宽) * 段持续时间 / 可用带宽
                            next_segment_download_time = (last_repr.bandwidth + ideal_repr.bandwidth) * (
                                self.mpd_provider.mpd.max_segment_duration / bw_per_video
                            )
                            # 记录详细的下载时间计算信息到日志
                            self.log.info(
                                f"bw_per_video={bw_per_video}, last_repr.bandwidth={last_repr.bandwidth}, "
                                + f"next_segment_download_time={next_segment_download_time}, buffer_level={buffer_level}"
                            )
                        else:
                            # 为音频流计算每音频的带宽分配
                            bw_per_audio = (available_bandwidth * 0.2) / num_audios
                            # 计算下一个段的下载时间
                            next_segment_download_time = (last_repr.bandwidth + ideal_repr.bandwidth) * (
                                self.mpd_provider.mpd.max_segment_duration / bw_per_audio
                            )
                        # 如果计算的下载时间小于等于当前缓冲区水位，保持上次选择
                        if next_segment_download_time <= buffer_level:
                            final_repr_id = last_repr.id
                        else:
                            # 否则降级到理想选择
                            final_repr_id = ideal_repr.id
                    else:
                        # 如果上次选择带宽不高于理想选择，直接使用理想选择
                        final_repr_id = ideal_repr.id
                else:
                    # 如果缓冲区水位在恐慌阈值和安全阈值之间，使用理想选择
                    final_repr_id = ideal_repr.id
                # 将最终选择结果添加到结果字典
                final_selections[id_] = final_repr_id
        else:
            # 如果没有上次选择的历史记录，直接使用理想选择
            final_selections = ideal_selection
        
        # 更新上次选择记录，为下次选择提供历史信息
        self._last_selections = final_selections
        # 记录最终选择结果到日志，包含当前带宽信息
        self.log.info(f"Final selection at {self.bandwidth_meter.bandwidth} is {final_selections}")
        # 返回最终选择结果
        return final_selections
