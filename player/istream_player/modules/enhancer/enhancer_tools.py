# -*- coding: utf-8 -*-
"""Alias: mod_enhancer=imdn_tools → MultiTaskEnhancer（探索线）。

正式名称请用 multitask。保留 imdn_tools 以免旧命令行失效。
"""
from istream_player.core.module import ModuleOption
from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.scheduler import Scheduler
from istream_player.core.renderer import Renderer
from istream_player.core.player import Player

from .enhancer_multitask import MultiTaskEnhancer


@ModuleOption(
    "imdn_tools",
    default=False,
    requires=[
        "model_downloader",
        DownloadBufferImpl,
        EnhanceBufferImpl,
        Scheduler,
        Renderer,
        Player,
    ],
)
class ToolEnhancer(MultiTaskEnhancer):
    """Deprecated alias of MultiTaskEnhancer."""
    pass
