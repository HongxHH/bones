import argparse
import asyncio
import logging
from collections import defaultdict
from pprint import pformat
from typing import Any, Callable, Dict, Optional, Type, TypedDict

from istream_player.config.config import PlayerConfig
from istream_player.core.module import Module, ModuleInterface
from istream_player.modules.nes import (BONESController,
                                        DynamicGreedyController,
                                        BOLAGreedyController,
                                        BufferGreedyController,
                                        ThroughputGreedyController,
                                        NASController,
                                        AlwaysGreedyController)

from istream_player.modules.analyzer.analyzer import PlaybackAnalyzer
from istream_player.modules.analyzer.event_logger import EventLogger
from istream_player.modules.analyzer.file_content_listener import \
    FileContentListener
from istream_player.modules.analyzer.playback import Playback
# from istream_player.modules.buffer.buffer_manager import BufferManagerImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.download_buffer import DownloadBufferImpl


from istream_player.modules.bw_meter.bandwidth import BandwidthMeterImpl
from istream_player.modules.downloader.quic.client import QuicClientImpl
from istream_player.modules.downloader import LocalClient, TCPClientImpl, TraceClient
from istream_player.modules.mpd.mpd_provider_impl import MPDProviderImpl
from istream_player.modules.player.player_dash import DASHPlayer
from istream_player.modules.scheduler.scheduler import SchedulerImpl

from istream_player.modules.renderer import OpenGLRenderer, HeadlessRenderer, OpenCVRenderer
from istream_player.modules.enhancer import IMDNEnhancer,IMDNEnhancerAlways

# 模块初始化函数类型定义 - 用于创建模块实例的函数签名
ModInitFnType = Callable[[str, Any, Any], Dict[str, Module]]


def first_non_none(*args):
    """
    返回第一个非None值的工具函数
    用于在多个可选值中选择第一个有效的值
    """
    # 遍历所有传入的参数
    for arg in args:
        # 如果参数不是None，则返回该参数
        if arg is not None:
            return arg
    # 如果所有参数都是None，则返回None
    return None


def get_mod_name(val: str):
    """
    从模块配置字符串中提取模块名称
    模块配置格式: "mod_name:prop1=val1,prop2=val2"
    返回冒号前的部分并转换为小写
    """
    # 按冒号分割，取第一部分并转换为小写
    return val.split(":", 1)[0].lower()


def get_mod_props(val: str):
    """
    从模块配置字符串中提取模块属性
    解析格式: "mod_name:prop1=val1,prop2=val2"
    返回属性字典
    """
    # 按冒号分割，获取属性部分
    all_props = val.split(":", 1)
    # 如果没有属性部分，返回空字典
    if len(all_props) == 1:
        return {}

    def prop_key(prop):
        """提取属性名 - 等号前的部分"""
        return prop.split("=", 1)[0]

    def prop_val(prop):
        """提取属性值 - 等号后的部分，如果没有等号则默认为True"""
        return [*prop.split("=", 1), True][1]

    # 解析所有属性并构建字典
    ret = {prop_key(prop): prop_val(prop) for prop in all_props[1].split(",")}
    # print(all_props[1].split(",")[0].split("=", 1), ret)  # 调试输出
    return ret


class PlayerContext:
    """
    播放器上下文类 - 管理播放器运行时的所有模块和配置
    提供异步上下文管理器接口，负责模块的初始化、运行和清理
    """
    # 日志记录器 - 用于记录播放器上下文的操作日志
    log = logging.getLogger("PlayerContext")

    def __init__(self, config: PlayerConfig, modules: Dict[str, Dict[str, Module]], composer) -> None:
        """
        初始化播放器上下文
        
        Args:
            config: 播放器配置对象
            modules: 模块字典，按类型分组的模块实例
            composer: 模块组合器引用，用于依赖解析
        """
        # 存储所有模块实例的字典
        self.modules = modules
        # 播放器配置对象
        self.config = config
        # 模块组合器引用，用于获取模块依赖关系
        self.composer = composer

    async def __aenter__(self):
        """
        异步上下文管理器入口 - 设置所有模块
        按照依赖关系顺序初始化所有注册的模块
        """
        # 记录模块设置开始
        self.log.info("\tSetting up modules")
        # 调试输出：显示所有模块类型和名称
        for mod_type, mods in self.modules.items():
            self.log.debug(f"\t\t{mod_type} : {mods}")
        # 遍历所有模块类型和实例
        for mod_type, mods in self.modules.items():
            for mod_name, mod in mods.items():
                # 获取当前模块的依赖关系
                deps = self.composer.get_deps(mod.__class__.__mod_requires__)
                # print(f"Dependencies for {mod_name}")  # 调试输出
                # pprint(deps)  # 调试输出依赖关系
                # 调用模块的setup方法，传入配置和依赖
                await mod.setup(self.config, *deps)
        # 返回自身，供上下文管理器使用
        return self

    async def __aexit__(self, *args):
        """
        异步上下文管理器出口 - 清理所有模块
        在播放器关闭时调用所有模块的cleanup方法
        """
        # 遍历所有模块并调用清理方法
        for mods in self.modules.values():
            for mod in mods.values():
                await mod.cleanup()

    async def run(self):
        """
        运行播放器 - 启动所有模块的异步任务
        为每个模块创建独立的异步任务并等待完成
        """
        # 创建任务列表
        tasks = []
        # 为每个模块创建异步运行任务
        for mods in self.modules.values():
            for mod in mods.values():
                # 创建任务并指定任务名称用于调试
                tasks.append(asyncio.create_task(mod.run(), name=f"TASK_MOD_{mod.__mod_name__}_RUN"))

        # 等待所有任务完成
        for task in tasks:
            await task


class ModuleCliConfig(TypedDict):
    """
    模块命令行配置类型定义
    定义模块在命令行参数中的配置选项
    """
    # 帮助信息 - 模块在命令行中的描述
    help: Optional[str]
    # 是否必需 - 该模块是否必须在命令行中指定
    required: Optional[bool]
    # 默认值 - 模块的默认配置值，可以是字符串或字符串列表
    default: Optional[list[str] | str]
    # choices: list[str]  # 可选值列表（已注释）
    # 是否允许多个 - 是否允许同时指定多个该类型的模块
    allow_multi: Optional[bool]


class PlayerComposer:
    """
    播放器组合器类 - 核心模块管理系统
    负责模块注册、依赖解析、实例创建和播放器构建
    """
    # 日志记录器 - 用于记录组合器操作
    log = logging.getLogger("PlayerComposer")

    # 模块命令行配置字典 - 存储每个模块类型的CLI配置
    module_cli: Dict[str, ModuleCliConfig]
    # 模块初始化函数字典 - 存储每个模块类型的初始化函数
    module_init_fn: Dict[str, ModInitFnType]
    # 模块选项字典 - 存储每个模块类型可用的模块类
    module_options: Dict[str, Dict[str, Type[Module]]]
    # 模块实例字典 - 存储已创建的模块实例
    modules: Dict[str, Dict[str, Module]]

    def __init__(self) -> None:
        """
        初始化播放器组合器
        创建所有必要的字典和默认配置
        """
        # 初始化模块选项字典
        self.module_options = {}
        # 初始化模块实例字典，使用defaultdict自动创建空字典
        self.modules = defaultdict(dict)
        # 初始化模块初始化函数字典
        self.module_init_fn = {}
        # 初始化模块CLI配置，设置默认值
        self.module_cli = defaultdict(lambda: ModuleCliConfig(help="[Module]", allow_multi=False, default="", required=False))

    def get_deps(self, reqs: list[str | Type[ModuleInterface]]):
        """
        获取模块依赖关系
        根据模块的依赖要求，从已注册的模块中找到对应的依赖实例
        
        Args:
            reqs: 依赖要求列表，可以是模块名称字符串或接口类型
            
        Returns:
            依赖模块实例列表
            
        Raises:
            Exception: 当找不到必需的依赖模块时
        """
        # 依赖列表
        deps = []
        # 遍历每个依赖要求
        for req in reqs:
            dep = {}
            # 如果是字符串类型的依赖（模块名称）
            if isinstance(req, str):
                # 在所有模块中查找匹配的模块名称
                for mods in self.modules.values():
                    for mod_name, mod in mods.items():
                        if mod_name == req:
                            dep[mod_name] = mod
            else:
                # 如果是类型依赖（接口类型），查找实现了该接口的模块
                for mods in self.modules.values():
                    for mod_name, mod in mods.items():
                        if issubclass(mod.__class__, req):
                            dep[mod_name] = mod
            # 检查是否找到依赖
            if len(dep) == 0:
                # 如果没找到依赖，抛出异常
                raise Exception(f"Module dependency not found : {req}")
            else:
                # 如果找到依赖，添加到依赖列表
                # 如果找到多个依赖，返回所有依赖；否则返回单个依赖
                deps.append(dep.values() if len(dep) > 1 else list(dep.values())[0])
        return deps

    def create_arg_parser(self, config: Optional[PlayerConfig] = None):
        """
        创建命令行参数解析器
        根据注册的模块自动生成命令行参数选项
        
        Args:
            config: 可选的PlayerConfig实例（保留用于向后兼容，当前未使用）
        
        Returns:
            argparse.ArgumentParser: 配置好的参数解析器
            
        Note:
            模块参数的默认值不再从register_module时设置的默认值读取，
            而是从PlayerConfig类的默认值读取。如果用户未提供命令行参数，
            argparse会返回None，load_from_dict会跳过None值，保留config.py中的默认值。
        """
        # 创建主参数解析器
        parser = argparse.ArgumentParser(description="IStream DASH Player")

        class ModuleChoices(list[str]):
            """
            模块选择列表类 - 自定义列表类，支持模块名称的模糊匹配
            在检查包含关系时使用小写比较
            """
            # 重写__contains__方法，使用小写比较
            def __contains__(self, option: str):
                return super().__contains__(option.split(":", 1)[0].lower())

        # 添加基本命令行参数
        parser.add_argument("--config", help="Configure using yaml/json", required=False)
        parser.add_argument("-i", "--input", help="Manifest (MPD) file location", type=str, required=True)
        parser.add_argument("-t", "--trace", help="Network trace location", type=str, required=False)
        parser.add_argument("-v", "--verbose", help="Enable debug level output", action="store_true", required=False)
        parser.add_argument("-l", "--log", help="Log file path", type=str, required=False)
        parser.add_argument('-r', "--run_dir",  help="Run directory, will be automatically OVERWRITTEN", default="output/istream_run_dir")
        parser.add_argument('-aw', "--content_aware", help="Content-aware enhancement", action="store_true", required=False)
        # pprint(self.module_cli)  # 调试输出模块CLI配置
        
        # 为每个模块类型添加命令行参数
        for mod_type, mods in self.module_options.items():
            # 获取该模块类型的CLI配置
            cli_opt = self.module_cli[mod_type]
            
            # 添加模块参数
            parser.add_argument(
                f"--mod_{mod_type}",  # 参数名称格式：--mod_模块类型
                help=cli_opt["help"],  # 帮助信息
                required=bool(cli_opt["required"]),  # 是否必需
                choices=list(self.module_options[mod_type].keys()),  # 可选值列表
                action=("append" if cli_opt["allow_multi"] else "store"),  # 动作类型
            )

        return parser

    async def run(self, config: PlayerConfig):
        """
        运行播放器
        使用给定的配置创建播放器上下文并运行
        
        Args:
            config: 播放器配置对象
        """
        # 创建播放器上下文并运行
        async with self.make_player(config) as player:
            await player.run()

    def register_module(
        self,
        mod_type: str,
        mod_class: Type[Module] | list[Type[Module]],
        init_fn: ModInitFnType | None,
        # 可选参数
        mod_help: Optional[str] = None,
        mod_required: Optional[bool] = None,
        mod_default: Optional[str | list[str]] = None,
        mod_allow_multi: Optional[bool] = None,
    ):
        """
        注册模块类型
        将模块类注册到组合器中，并配置其CLI选项
        
        Args:
            mod_type: 模块类型名称
            mod_class: 模块类或模块类列表
            init_fn: 模块初始化函数
            mod_help: CLI帮助信息
            mod_required: 是否必需
            mod_default: 默认值
            mod_allow_multi: 是否允许多个实例
        """
        # 设置或更新初始化函数
        if init_fn is not None:
            self.module_init_fn[mod_type] = init_fn

        # 设置或更新模块CLI配置
        prev_cli = self.module_cli[mod_type]
        new_cli = ModuleCliConfig(help=mod_help, required=mod_required, default=mod_default, allow_multi=mod_allow_multi)
        # 使用first_non_none函数合并配置，优先使用新值
        new_cli["help"] = first_non_none(mod_help, prev_cli["help"])
        new_cli["required"] = first_non_none(mod_required, prev_cli["required"])
        new_cli["default"] = first_non_none(mod_default, prev_cli["default"])
        new_cli["allow_multi"] = first_non_none(mod_allow_multi, prev_cli["allow_multi"])
        # 更新CLI配置
        prev_cli.update(new_cli)

        # 如果模块类型不存在，创建新的模块选项字典
        if mod_type not in self.module_options:
            self.module_options[mod_type] = {}

        # 支持一次注册多个模块
        if isinstance(mod_class, list):
            # 遍历模块类列表
            for _mod_class in mod_class:
                # 检查模块名称是否已存在
                if _mod_class.__mod_name__ in self.module_options[mod_type]:
                    raise Exception(f"Module with name {_mod_class.__mod_name__} alerady registerd under {mod_type}.")
                # 注册模块类
                self.module_options[mod_type][_mod_class.__mod_name__] = _mod_class
        else:
            # 注册单个模块类
            if mod_class.__mod_name__ in self.module_options[mod_type]:
                raise Exception(f"Module with name {mod_class.__mod_name__} alerady registerd under {mod_type}.")
            self.module_options[mod_type][mod_class.__mod_name__] = mod_class

    def make_player(self, config: PlayerConfig):
        """
        创建播放器实例
        根据配置创建并初始化所有模块，构建播放器上下文
        
        Args:
            config: 播放器配置对象
            
        Returns:
            PlayerContext: 配置好的播放器上下文
            
        Raises:
            Exception: 当模块初始化函数未提供时
        """
        # 自动选择下载器类型
        if config.mod_downloader == "auto":
            # 如果提供了trace文件，优先使用trace下载器
            if config.trace:
                config.mod_downloader = "trace"
            # 如果输入是HTTP/HTTPS URL，使用TCP下载器
            elif config.input.lower().startswith("http://") or config.input.lower().startswith("https://"):
                config.mod_downloader = "tcp"
            else:
                # 否则使用本地文件下载器
                config.mod_downloader = "local"

        # 调试输出：打印配置信息
        list(map(self.log.debug, pformat(config).splitlines()))

        # 遍历配置中的所有属性
        for attr_name, val in config.__dict__.items():
            # 只处理以"mod_"开头的模块配置
            if not attr_name.startswith("mod_"):
                continue
            # 提取模块类型名称（去掉"mod_"前缀,如"mod_mpd" -> "mpd"）
            mod_type_name = attr_name[4:]
            # 检查是否有对应的初始化函数
            if self.module_init_fn.get(mod_type_name) is None:
                raise Exception(f"Module init function not provided for module {mod_type_name}")
            # 使用初始化函数创建模块实例
            self.modules[mod_type_name].update(self.module_init_fn[mod_type_name](mod_type_name, val, self))
        # 返回播放器上下文
        return PlayerContext(config, self.modules, self)

    def register_core_modules(self):
        """
        注册核心模块
        注册播放器运行所需的所有核心模块类型
        """
        # 注册MPD提供器模块
        self.register_module("mpd", [MPDProviderImpl], single_initializer, "MPD Provider", False, "mpd")
        # 注册下载器模块（支持多种下载方式）
        self.register_module(
            "downloader", [LocalClient, TCPClientImpl, QuicClientImpl, TraceClient], downloader_initializer, "Downloader", False, "trace"
        )
        # 注册带宽测量模块
        self.register_module("bw", [BandwidthMeterImpl], single_initializer, "Bandwidth Estimation", False, "bw_meter")
        # 注册网络感知调度器模块（支持多种算法）
        self.register_module(
            "nes",
            [BONESController, DynamicGreedyController, BOLAGreedyController, BufferGreedyController, ThroughputGreedyController, NASController, AlwaysGreedyController],
            single_initializer,
            "Neural-Enhanced Streaming Controller",
            False,
            "buffer_greedy",

        )
        # 注册任务调度器模块
        self.register_module("scheduler", [SchedulerImpl], single_initializer, "Segment download scheduler", False, "scheduler")
        # self.register_module("buffer", [BufferManagerImpl], single_initializer, "Buffer manager", False, "buffer_manager")  # 已注释的缓冲管理器
        # 注册下载缓冲区模块
        self.register_module("download_buffer", [DownloadBufferImpl], single_initializer, "Download Buffer", False,
                             "download_buffer")
        # 注册增强缓冲区模块
        self.register_module("enhance_buffer", [EnhanceBufferImpl], single_initializer, "Enhance Buffer", False,
                             "enhance_buffer")

        # 注册视频增强器模块
        self.register_module("enhancer",
                             [IMDNEnhancer,IMDNEnhancerAlways],
                             single_initializer,
                             "Enhancer",
                             False,
                             "imdn_always"
                             )

        # 注册DASH播放器模块
        self.register_module("player", [DASHPlayer], single_initializer, "DASH Player", False, "dash")

        # 注册分析器模块（支持多个分析器）
        self.register_module(
            "analyzer",
            [PlaybackAnalyzer, FileContentListener, Playback, EventLogger],
            multi_initializer,
            "Analyzers",
            False,
            mod_default=[],
            mod_allow_multi=True,
        )
        # 注册渲染器模块（支持多种渲染方式）
        self.register_module("renderer", [OpenGLRenderer, HeadlessRenderer, OpenCVRenderer], single_initializer, "Renderer", False, "headless")


def single_initializer(mod_type, mod_name, composer: PlayerComposer) -> Dict[str, Module]:
    """
    单模块初始化器
    用于创建单个模块实例的初始化函数
    
    Args:
        mod_type: 模块类型名称
        mod_name: 模块名称（可能包含属性配置）
        composer: 播放器组合器引用
        
    Returns:
        包含单个模块实例的字典
        
    Raises:
        Exception: 当模块类型不支持单模块时
    """
    # 检查模块名称是否为字符串类型
    if not isinstance(mod_name, str):
        raise Exception(f"Module type {mod_type} only supports single module. Provided {mod_name}")

    # 获取模块类并创建实例
    return {get_mod_name(mod_name): composer.module_options[mod_type][get_mod_name(mod_name)](**get_mod_props(mod_name))}


def multi_initializer(mod_type, val, composer: PlayerComposer) -> Dict[str, Module]:
    """
    多模块初始化器
    用于创建多个模块实例的初始化函数，支持字符串、列表和字典格式的输入
    
    Args:
        mod_type: 模块类型名称
        val: 模块配置值（字符串、列表或字典）
        composer: 播放器组合器引用
        
    Returns:
        包含多个模块实例的字典
        
    Raises:
        Exception: 当模块值格式无效时
    """
    # 如果是字符串，使用单模块初始化器
    if isinstance(val, str):
        return single_initializer(mod_type, val, composer)
    elif isinstance(val, list):
        # 如果是列表，为每个模块创建实例
        # print(mod_type, [get_mod_name(mod) for mod in val])  # 调试输出
        return {get_mod_name(mod): composer.module_options[mod_type][get_mod_name(mod)](**get_mod_props(mod)) for mod in val}
    elif isinstance(val, dict):
        # 如果是字典，使用字典的键值对创建模块
        return {
            mod_key: composer.module_options[mod_type][get_mod_name(mod)](**get_mod_props(mod)) for mod_key, mod in val.items()
        }
    else:
        # 如果格式不支持，抛出异常
        raise Exception(f"Invalid mod value '{val}' received for mod '{mod_type}'")


def downloader_initializer(mod_type, val, composer: PlayerComposer) -> Dict[str, Module]:
    """
    下载器专用初始化器
    为下载器模块创建多个实例（MPD下载器、段下载器、模型下载器）
    
    Args:
        mod_type: 模块类型名称
        val: 下载器配置值
        composer: 播放器组合器引用
        
    Returns:
        包含三个下载器实例的字典
        
    Raises:
        Exception: 当模块类型不支持单模块时
    """
    # 检查配置值是否为字符串类型
    if not isinstance(val, str):
        raise Exception(f"Module type {mod_type} only supports single module. Provided {val}")

    # 获取下载器类
    _cl = composer.module_options[mod_type][get_mod_name(val)]
    # 创建三个下载器实例：MPD下载器、段下载器、模型下载器
    return {"mpd_downloader": _cl(**get_mod_props(val)), "segment_downloader": _cl(**get_mod_props(val)), "model_downloader": _cl(**get_mod_props(val))}

