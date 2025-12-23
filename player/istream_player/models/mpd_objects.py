from dataclasses import dataclass
from typing import Dict, Literal, Optional
import torch


class MPD(object):
    """
    MPD（Media Presentation Description）类 - DASH流媒体的核心描述文件对象
    
    该类封装了DASH流媒体的所有元数据信息，包括媒体内容的基本信息、
    自适应集集合以及播放相关的配置参数。MPD文件是DASH标准的核心，
    播放器通过解析MPD文件来了解可用的媒体内容和质量级别。
    """
    def __init__(
        self,
        content: str,  # MPD文件的原始XML内容
        url: str,  # MPD文件的下载URL
        type_: Literal["static", "dynamic"],  # MPD类型：静态（VOD）或动态（Live）
        media_presentation_duration: float,  # 媒体演示总时长（秒）
        max_segment_duration: float,  # 单个段的最大持续时间（秒）
        min_buffer_time: float,  # 推荐的最小缓冲时间（秒）
        adaptation_sets: Dict[int, "AdaptationSet"],  # 自适应集字典，键为自适应集ID
        attrib: Dict[str, str]  # 从XML中提取的所有属性
    ):
        # 存储MPD文件的原始XML内容
        # 用于调试、重新解析或传递给其他组件
        self.content = content
        """
        MPD文件的原始XML内容字符串
        """

        # 存储MPD文件的下载URL
        # 用于相对路径解析和重新下载
        self.url = url
        """
        MPD文件的下载URL地址
        """

        # 设置MPD类型，影响播放器的行为策略
        # static: 点播内容，所有段都可用，可以随机访问
        # dynamic: 直播内容，段会持续生成，需要定期更新MPD
        self.type: Literal["static", "dynamic"] = type_
        """
        MPD类型标识：static表示点播内容，dynamic表示直播内容
        """

        # 媒体演示的总时长
        # 对于静态内容，这是固定的总时长
        # 对于动态内容，这表示当前可用的时长
        self.media_presentation_duration = media_presentation_duration
        """
        媒体演示的总时长，单位为秒
        """

        # 推荐的最小缓冲时间
        # 播放器应保持的最小缓冲量，用于平滑播放
        self.min_buffer_time = min_buffer_time
        """
        推荐的最小缓冲时间，单位为秒
        """

        # 单个段的最大持续时间
        # 用于计算缓冲水位和调度决策
        self.max_segment_duration = max_segment_duration
        """
        单个段的最大持续时间，单位为秒
        """

        # 所有自适应集的集合
        # 这是MPD的核心数据结构，包含了所有可用的媒体流
        # 键为自适应集ID，值为AdaptationSet对象
        self.adaptation_sets: Dict[int, AdaptationSet] = adaptation_sets
        """
        所有自适应集的字典，键为自适应集ID，值为AdaptationSet对象
        """

        # 从XML根元素提取的所有属性
        # 包含MPD文件的元数据和配置信息
        self.attrib = attrib
        """
        从XML根元素提取的所有属性字典
        """


class AdaptationSet(object):
    """
    自适应集类 - 代表一组具有相同内容但不同质量级别的媒体流
    
    自适应集是DASH标准中的核心概念，它将相同内容的不同质量级别组织在一起。
    例如，一个视频自适应集可能包含1080p、720p、480p等不同分辨率的视频流。
    播放器可以根据网络条件和设备能力选择合适的表示进行播放。
    """
    def __init__(
        self,
        adaptation_set_id: int,  # 自适应集的唯一标识符
        content_type: Literal["video", "audio"],  # 内容类型：视频或音频
        frame_rate: Optional[str],  # 帧率字符串，可能为None
        max_width: int,  # 该自适应集中所有表示的最大宽度
        max_height: int,  # 该自适应集中所有表示的最大高度
        par: Optional[str],  # 像素宽高比（Pixel Aspect Ratio）
        representations: Dict[int, "Representation"],  # 表示集合，键为表示ID
        attrib: Dict[str, str]  # 从XML中提取的所有属性
    ):
        # 设置自适应集的唯一标识符
        # 在MPD文件中，每个自适应集都有唯一的ID
        self.id = adaptation_set_id
        """
        自适应集的唯一标识符
        """

        # 设置内容类型，决定播放器的处理方式
        # video: 视频内容，需要视频解码器和渲染器
        # audio: 音频内容，需要音频解码器和播放器
        self.content_type: str = content_type
        """
        自适应集的内容类型，只能是"video"或"audio"
        """

        # 设置帧率信息
        # 对于视频内容，这表示播放帧率
        # 对于音频内容，通常为None
        self.frame_rate: Optional[str] = frame_rate
        """
        帧率字符串，对于视频内容表示播放帧率
        """

        # 设置该自适应集中所有表示的最大宽度
        # 用于UI显示和渲染器配置
        self.max_width = max_width
        """
        该自适应集中所有表示的最大宽度（像素）
        """

        # 设置该自适应集中所有表示的最大高度
        # 用于UI显示和渲染器配置
        self.max_height = max_height
        """
        该自适应集中所有表示的最大高度（像素）
        """

        # 设置像素宽高比
        # 用于正确的视频显示比例
        self.par: Optional[str] = par
        """
        像素宽高比，用于正确的视频显示比例
        """

        # 设置表示集合
        # 这是自适应集的核心，包含所有不同质量级别的表示
        # 键为表示ID，值为Representation对象
        self.representations: Dict[int, Representation] = representations
        """
        该自适应集下的所有表示集合，键为表示ID，值为Representation对象
        """

        # 存储从XML中提取的所有属性
        # 包含自适应集的元数据和配置信息
        self.attrib = attrib
        """
        从XML自适应集元素提取的所有属性字典
        """


class Representation(object):
    """
    表示类 - 代表特定质量级别的媒体流
    
    表示是DASH标准中的质量级别概念，每个表示代表一种特定的编码配置，
    包括分辨率、码率、编解码器等。播放器根据网络条件和设备能力
    选择合适的表示进行下载和播放。
    """
    def __init__(
        self,
        id_: int,  # 表示的唯一标识符
        mime_type: str,  # MIME类型，如"video/mp4"
        codecs: str,  # 编解码器字符串，如"avc1.42E01E"
        bandwidth: int,  # 平均码率，单位为bps
        width: int,  # 视频宽度（像素）
        height: int,  # 视频高度（像素）
        initialization: str,  # 初始化段的URL
        segments: Dict[int, "Segment"],  # 段集合，键为段索引
        attrib: Dict[str, str]  # 从XML中提取的所有属性
    ):
        # 设置表示的唯一标识符
        # 在自适应集内，每个表示都有唯一的ID
        self.id = id_
        """
        表示的唯一标识符
        """

        # 设置MIME类型
        # 用于确定媒体格式和编解码器
        self.mime_type = mime_type
        """
        表示的MIME类型，如"video/mp4"或"audio/mp4"
        """

        # 设置编解码器字符串
        # 用于确定具体的编解码器版本和配置
        self.codecs: str = codecs
        """
        编解码器字符串，如"avc1.42E01E"表示H.264编码
        """

        # 设置平均码率
        # 这是ABR算法选择表示的主要依据
        # 单位为比特每秒（bps）
        self.bandwidth: int = bandwidth
        """
        该表示的平均码率，单位为比特每秒（bps）
        """

        # 设置视频宽度
        # 用于渲染器配置和UI显示
        self.width = width
        """
        视频的宽度，单位为像素
        """

        # 设置视频高度
        # 用于渲染器配置和UI显示
        self.height = height
        """
        视频的高度，单位为像素
        """

        # 设置初始化段URL
        # 初始化段包含了解码所需的头信息
        # 在播放任何媒体段之前必须先下载初始化段
        self.initialization: str = initialization
        """
        初始化段的URL地址
        """

        # 设置段集合
        # 这是表示的核心，包含所有媒体段
        # 键为段索引，值为Segment对象
        self.segments: Dict[int, Segment] = segments
        """
        该表示的所有视频段集合，键为段索引，值为Segment对象
        """

        # 存储从XML中提取的所有属性
        # 包含表示的元数据和配置信息
        self.attrib = attrib
        """
        从XML表示元素提取的所有属性字典
        """


class Segment(object):
    """
    段类 - 代表媒体内容的一个时间片段
    
    段是DASH流媒体的基本播放单元，每个段包含一段时间的媒体内容。
    播放器通过下载和播放连续的段来实现流媒体播放。段对象不仅包含
    媒体数据的位置信息，还包含播放器调度和增强处理的相关信息。
    """
    def __init__(
            self,
            url: str,  # 段的下载URL
            init_url: str,  # 初始化段的URL
            duration: float,  # 段的播放持续时间（秒）
            start_time: float,  # 段在媒体时间轴上的开始时间（秒）
            as_id: int,  # 所属自适应集的ID
            repr_id: int,  # 所属表示的ID
            path: Optional[str] = None,  # 本地文件路径
            init_path: Optional[str] = None,  # 初始化段的本地文件路径
            decode_data: Optional[torch.Tensor] = None,  # 解码后的数据
            download_action: Optional[int] = 0,  # 下载动作（表示索引）
            enhance_action: Optional[int] = 0 , # 增强动作（模型索引）
            enhance_scale: Optional[int] = 0, # 增强放大倍数
            is_enhance: bool = None,  # 是否真正执行了增强动作 而没有abort
            enhance_latency: Optional[float] = None,  # 若执行了增强动作，则记录该动作的延迟时间
            decision_time_buffer_level: Optional[float] = None, # abr算法做出决策时的缓冲区剩余时长
            enhance_end_to_play_time: Optional[float] = None, 
    ):
        # 设置段的下载URL
        # 这是段在服务器上的位置，用于下载媒体数据
        self.url = url
        """
        段的下载URL地址
        """

        # 设置初始化段的URL
        # 初始化段包含了解码该表示所需的所有头信息
        self.init_url = init_url
        """
        流初始化段的URL地址
        """

        # 设置段的播放持续时间
        # 这是段在播放时占用的时间长度
        # 用于播放器的时间管理和缓冲计算
        self.duration = duration
        """
        段的播放持续时间，单位为秒
        """

        # 设置段在媒体时间轴上的开始时间
        # 用于播放器的时间同步和段排序
        self.start_time = start_time
        """
        段在媒体时间轴上的开始时间，单位为秒
        """

        # 设置段所属的自适应集ID
        # 用于标识段属于哪个自适应集
        self.as_id = as_id
        """
        段所属的自适应集ID
        """

        # 设置段所属的表示ID
        # 用于标识段属于哪个表示（质量级别）
        self.repr_id = repr_id
        """
        段所属的表示ID
        """

        # 设置段的本地文件路径
        # 下载完成后，段数据会保存到本地文件
        # 这个路径用于后续的解码和播放
        self.path = path
        """
        段下载后的本地文件路径
        """

        # 设置初始化段的本地文件路径
        # 初始化段下载后会保存到本地
        # 用于解码器的初始化
        self.init_path = init_path
        """
        初始化段下载后的本地文件路径
        """

        # 设置解码后的数据
        # 如果段已经被解码，这里存储解码后的张量数据
        # 用于预解码的段或增强处理后的段
        self.decode_data = decode_data
        """
        解码后的数据，以PyTorch张量形式存储
        """

        # 设置下载动作
        # 记录调度器选择下载哪个表示（质量级别）
        # 用于ABR算法的历史记录和调试
        self.download_action = download_action
        """
        下载动作，记录选择的表示索引
        """

        # 设置增强动作
        self.enhance_action = enhance_action
        """
        增强动作，记录选择的增强模型索引
        """

        self.enhance_scale = enhance_scale
        """
        增强放大倍率
        """

        self.is_enhance = is_enhance
        """
        是否真正执行了增强动作 而没有abort
        """
        
        self.enhance_start_time = None
        """
        若执行了增强动作，则记录该动作的开始时间戳
        """

        self.enhance_end_time = None
        """
        若执行了增强动作，则记录该动作的结束时间戳
        """

        self.enhance_latency = enhance_latency
        """
        若执行了增强动作，则记录该动作的延迟时间
        """

        self.decision_time_buffer_level = decision_time_buffer_level
        """
         abr算法做出决策时的缓冲区剩余时长
        """

        self.enhanced_cnt = None
        """
        使用模型增强的帧数
        """
        self.interpolated_cnt = None
        """
        插值 / 复制的帧数
        """

        self.enhance_fps = None
        """
        段增强 FPS
        """

        self.enhance_frame_interval = None
        """
        超分间隔帧数（后续可变）
        """

        self.fast_complete_threshold = None
        """
        fast-complete 阈值（秒）
        """

        self.fast_complete_triggered = None
        """
        fast-complete 是否触发
        """

        self.complexity_list = None
        """
        段复杂度列表
        """

        self.vmaf = None
        """
        段增强后的离线 VMAF（来自质量表）
        """

        self.wasted_enhanced_cnt = None
        """
        段增强后的浪费帧数  若为None表明没有进入增强  若为0表明还没正式开始进行增强就被丢弃，没有算力浪费
        """

        self.abort_reason = None
        """
        若被中止，记录中止原因
        """

        self.enhance_end_to_play_time = None
        """
        若执行了增强动作，增强完成后该增强段距离播放开始的时间 (渲染器剩余时长 + 中间隔得段数*段时长)
        """

    def __str__(self):
        """
        返回段的字符串表示
        用于调试和日志记录，包含段的所有关键信息
        """
        result = f"Segment: url={self.url}, init_url={self.init_url}, duration={self.duration}, start_time={self.start_time}, as_id={self.as_id}, repr_id={self.repr_id}, path={self.path}, init_path={self.init_path}, download_action={self.download_action}, enhance_action={self.enhance_action}, is_enhance={self.is_enhance}"
        return result