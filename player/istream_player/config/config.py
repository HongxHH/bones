# 导入数据类相关的装饰器和字段工厂函数，用于创建配置类
from dataclasses import dataclass, field
# 导入类型提示，用于可选类型的声明
from typing import Optional


# 静态配置类 - 存放播放器系统的全局常量配置参数
class StaticConfig(object):
    # 最大初始比特率 (bps) - 播放器启动时的比特率上限，设为1Mbps
    max_initial_bitrate = 1000000

    # 平滑因子 - 用于带宽估算的移动平均算法
    # 算法: averageSpeed = SMOOTHING_FACTOR * lastSpeed + (1-SMOOTHING_FACTOR) * averageSpeed
    smoothing_factor = 0.5

    # 最小帧块大小比例 - 视频段中I、P、B帧的大小比例下限
    # 用于确定视频分块的最小单位比例
    min_frame_chunk_ratio = 0.6

    # 视频质量阈值 - 用于视频质量评估的门限值
    vq_threshold = 0.8

    # [未使用] 基于大小比例的视频质量阈值 - 根据帧块比例计算的复合阈值
    # 计算公式结合了最小帧块比例和质量阈值
    vq_threshold_size_ratio = min_frame_chunk_ratio * (min_frame_chunk_ratio + (1 - min_frame_chunk_ratio) * vq_threshold)

    # 更新间隔 (秒) - 系统状态更新的时间间隔，50毫秒
    update_interval = 0.05

    # [未使用] 数据块大小 - 网络传输时的分块大小，40KB
    chunk_size = 40960

    # [未使用] 超时最大比例 - 网络超时的倍数上限
    timeout_max_ratio = 2

    # [未使用] 质量提升的最小持续时间 (毫秒) - 提升视频质量前需要等待的时间
    min_duration_for_quality_increase_ms = 6000

    # [未使用] 质量降低的最大持续时间 (毫秒) - 降低视频质量的最大等待时间
    max_duration_for_quality_decrease_ms = 8000

    # [未使用] 丢弃后重训练的最小持续时间 (毫秒) - 丢弃数据后重新训练的等待时间
    min_duration_to_retrain_after_discard_ms = 8000

    # [未使用] 带宽使用分数 - 可用带宽的使用比例，75%
    bandwidth_fraction = 0.75

    # 数据包最大延迟 (秒) - 超过此延迟的数据包不参与带宽估算
    # 用于过滤网络抖动对带宽测量的影响
    max_packet_delay = 2

    # 连续带宽估算窗口 (秒) - 带宽测量的时间窗口大小
    cont_bw_window = 1


# 播放器配置类 - 使用数据类装饰器，包含播放器的所有可配置参数
@dataclass
class PlayerConfig:
    # 静态配置引用 - 包含上述静态配置类的所有参数
    static = StaticConfig

    # ======================
    # 必需配置参数
    # ======================
    # 输入源 - 视频文件路径或流媒体URL
    input: str = ""
    # 网络追踪文件 - 用于网络模拟的追踪数据文件路径
    trace: str = ""
    # 运行目录 - 程序运行时的工作目录
    run_dir: str = "output/istream_run_dir"
    # 日志文件路径 - 记录播放器运行日志的文件位置
    log: str = "output/logs"
    # 指标输出文件路径 - 播放器运行过程中生成的指标数据文件位置
    metric_output: str = "output/data"
    # 图表输出文件路径 - 播放器运行过程中生成的图表数据文件位置
    plots_dir: str = "output/plots"
    # 时间因子 - 播放速度的倍数，1.0表示正常速度
    time_factor: float = 1

    # ======================
    # 模块配置 - 定义播放器各个功能模块的实现类型
    # ======================
    # MPD解析器模块 - 处理DASH清单文件解析
    mod_mpd: str = "mpd"
    # 下载器模块 - 处理视频段下载，"auto"表示自动选择 TCP下载器 或 本地文件下载器
    mod_downloader: str = "auto"
    # 带宽测量模块 - 负责网络带宽的实时测量
    mod_bw: str = "bw_meter"
    # 网络感知调度器模块 - 基于网络状态的智能调度算法
    mod_nes: str = "bones" # bones dynamic_greedy bola_greedy buffer_greedy tput_greedy nas
    # 任务调度器模块 - 管理各种任务的调度执行
    mod_scheduler: str = "scheduler"
    # 下载缓冲区模块 - 管理下载数据的缓冲
    mod_download_buffer: str = "download_buffer"
    # 增强缓冲区模块 - 管理视频增强处理的缓冲
    mod_enhance_buffer: str = "enhance_buffer"
    # 播放器模块 - 核心播放引擎，支持DASH协议
    mod_player: str = "dash"
    # 视频增强器模块 - 使用IMDN进行视频超分辨率增强
    mod_enhancer: str = "imdn_always"
    # 渲染器模块 - 视频显示渲染，"headless"表示无头模式
    mod_renderer: str = "opencv"  # 可选的OpenCV渲染器 opencv、opengl、headless
    # 分析器模块列表 - 数据收集和分析组件，使用工厂函数创建默认列表
    mod_analyzer: list[str] = field(default_factory=lambda: ["data_collector"])

    # ======================
    # 缓冲区配置
    # ======================
    # 最大缓冲持续时间 (秒) - 缓冲区能存储的最大视频时长
    buffer_duration: float = 30  # maximum buffer level (s)
    # 安全缓冲水位 (秒) - 保证流畅播放的安全缓冲时长
    safe_buffer_level: float = 6
    # 紧急缓冲水位 (秒) - 触发紧急下载的最低缓冲时长
    panic_buffer_level: float = 2.5

    # 以下配置项已被注释，保留作为备用参数
    # min_rebuffer_duration: float = 1  # 最小重缓冲持续时间
    # min_start_duration: float = 1     # 最小启动缓冲时间

    # 选择器配置 - 用于特定选择算法的参数
    select_as: str = "-"

    # SSL密钥日志文件 - 用于网络调试的SSL/TLS密钥记录文件
    ssl_keylog_file: Optional[str] = None

    # 实时事件日志文件路径 - 记录播放过程中的实时事件
    live_log: Optional[str] = None

    # ======================
    # 显示配置
    # ======================
    # 可选的显示分辨率配置
    display_width = 1920   # 1080p分辨率宽度
    display_height = 1080  # 1080p分辨率高度
    # 当前使用的显示分辨率 - 720p高清
    # display_width = 1280   # 显示宽度：1280像素
    # display_height = 720   # 显示高度：720像素
    # display_width = 640    # 备用的360p分辨率宽度
    # display_height = 360   # 备用的360p分辨率高度
    # 目标显示帧率 - 30FPS，注意：这是目标值，不保证实际达到
    display_fps = 30  # target display fps, cannot guarantee

    # ======================
    # 视频保存配置
    # ======================
    save_rendered_video: bool = False  # 是否保存渲染后的画面
    save_video_path: Optional[str] = "output/video"  # 保存文件路径，默认使用运行目录
    save_file_name:  str = "test"
    save_video_codec: str = "mp4v"  # 'XVID','MJPG','mp4v','H264'
    save_video_max_queue: int = 120  # 写盘线程缓冲帧数

    # ======================
    # 设备配置
    # ======================
    # 视频增强器运算设备 - 当前使用CPU进行视频增强计算
    enhancer_device = "cuda"  # cpu/gpu
    # 渲染器运算设备 - 使用CUDA GPU进行渲染加速
    renderer_device = "cuda" # cpu/gpu

    # 控制视频增强帧间隔
    enhance_frame_interval = 0 # 0:每帧都增强
    # 是否启用提前检测
    use_fast_complete = True

    # ======================
    # 视频增强元数据配置
    # ======================
    # 内容感知标志 - 是否启用针对特定内容的增强算法
    content_aware = False
    # 根据内容感知标志选择相应的增强元数据文件
    if content_aware:
        # 针对BBB视频内容优化的IMDN模型元数据
        enhance_metadata = "imdn_bbb_quality.json"
    else:
        # 使用通用DIV2K数据集训练的IMDN模型元数据
        enhance_metadata = "imdn_div2k_quality.json"


    def validate(self) -> None:
        """
        配置验证方法 - 确保配置参数设置正确
        检查必需参数是否已正确设置，如果验证失败则抛出断言错误
        """
        # 验证输入源参数不能为空 - 这是播放器运行的必需参数
        assert bool(self.input), "A non-empty '--input' arg or 'input' config is required"