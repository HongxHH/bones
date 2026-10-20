from __future__ import annotations

from ..config import ServerConfig
from ..registry import PluginRegistry
from .delivery import SessionFilterDeliveryAdapter, StaticDeliveryAdapter
from .hooks import ArchiveExperimentHook
from .measurers import NoopMeasurer, VmafMeasurer
from .packagers import IdentityPackager
from .recipes import ImportDashRecipe


def build_registry(config: ServerConfig) -> PluginRegistry:
    registry = PluginRegistry()
    registry.register_recipe(ImportDashRecipe())
    registry.register_packager(IdentityPackager())
    registry.register_delivery(StaticDeliveryAdapter())
    registry.register_delivery(SessionFilterDeliveryAdapter())
    registry.register_measurer(NoopMeasurer())
    registry.register_measurer(VmafMeasurer(config.ffmpeg_bin))
    registry.register_hook(ArchiveExperimentHook())
    return registry
