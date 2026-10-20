from __future__ import annotations

from typing import Any, Protocol

from .schemas import JobKind, PackageResult, RecipeOutput


class TranscodeRecipe(Protocol):
    name: str
    labels: list[str]
    steps: list[str]
    default_packager: str

    async def run(self, *, asset_path: str, staging_dir: str, params: dict[str, Any]) -> RecipeOutput:
        ...


class Packager(Protocol):
    name: str
    labels: list[str]

    async def publish(self, *, artifact: RecipeOutput, output_dir: str, params: dict[str, Any]) -> PackageResult:
        ...


class DeliveryAdapter(Protocol):
    name: str
    labels: list[str]
    steps: list[str]

    def resolve_mpd(self, *, mpd_xml: str, policy: dict[str, Any], public_base: str, session_id: str, publication_id: str) -> str:
        ...


class Measurer(Protocol):
    name: str
    labels: list[str]
    steps: list[str]

    async def measure(self, *, publication_dir: str, params: dict[str, Any]) -> dict[str, Any]:
        ...


class ExperimentHook(Protocol):
    name: str

    async def on_job_finished(self, *, job: dict[str, Any], artifacts: dict[str, Any], experiment_dir: str) -> None:
        ...


class PluginRegistry:
    def __init__(self) -> None:
        self.recipes: dict[str, TranscodeRecipe] = {}
        self.packagers: dict[str, Packager] = {}
        self.delivery: dict[str, DeliveryAdapter] = {}
        self.measurers: dict[str, Measurer] = {}
        self.hooks: dict[str, ExperimentHook] = {}

    def register_recipe(self, plugin: TranscodeRecipe) -> None:
        self.recipes[plugin.name] = plugin

    def register_packager(self, plugin: Packager) -> None:
        self.packagers[plugin.name] = plugin

    def register_delivery(self, plugin: DeliveryAdapter) -> None:
        self.delivery[plugin.name] = plugin

    def register_measurer(self, plugin: Measurer) -> None:
        self.measurers[plugin.name] = plugin

    def register_hook(self, plugin: ExperimentHook) -> None:
        self.hooks[plugin.name] = plugin

    def recipe(self, name: str) -> TranscodeRecipe:
        if name not in self.recipes:
            raise KeyError(f"Unknown transcode recipe: {name}")
        return self.recipes[name]

    def packager(self, name: str) -> Packager:
        if name not in self.packagers:
            raise KeyError(f"Unknown packager: {name}")
        return self.packagers[name]

    def delivery_adapter(self, name: str) -> DeliveryAdapter:
        if name not in self.delivery:
            raise KeyError(f"Unknown delivery adapter: {name}")
        return self.delivery[name]

    def measurer(self, name: str) -> Measurer:
        if name not in self.measurers:
            raise KeyError(f"Unknown measurer: {name}")
        return self.measurers[name]

    def pipeline_for(self, plugin: str, kind: JobKind) -> tuple[list[str], list[str]]:
        """Steps and labels come from the plugin, not from orchestrator if-tables."""
        if kind == JobKind.SESSION:
            adapter = self.delivery.get(plugin) or self.delivery["session_filter"]
            return list(adapter.steps), list(adapter.labels)
        if plugin in self.recipes:
            spec = self.recipes[plugin]
            return list(spec.steps), list(spec.labels)
        if plugin in self.measurers:
            spec = self.measurers[plugin]
            return list(spec.steps), list(spec.labels)
        if plugin in self.delivery:
            spec = self.delivery[plugin]
            return list(spec.steps), list(spec.labels)
        raise KeyError(f"Unknown plugin: {plugin}")

    def packager_name_for(self, plugin: str, params: dict[str, Any]) -> str:
        if params.get("packager"):
            return str(params["packager"])
        recipe = self.recipes.get(plugin)
        if recipe is not None:
            return recipe.default_packager
        return "identity"

    def labels_for_step(self, plugin: str, step: str, params: dict[str, Any]) -> list[str]:
        if step == "package":
            return list(self.packager(self.packager_name_for(plugin, params)).labels)
        if step == "measure":
            return list(self.measurer(plugin).labels)
        if step == "apply_policy":
            name = params.get("delivery") or plugin
            adapter = self.delivery.get(name) or self.delivery["session_filter"]
            return list(adapter.labels)
        if plugin in self.recipes:
            return list(self.recipes[plugin].labels)
        if plugin in self.measurers:
            return list(self.measurers[plugin].labels)
        return ["cpu"]

    def catalog(self) -> dict[str, list[str]]:
        return {
            "recipes": sorted(self.recipes),
            "packagers": sorted(self.packagers),
            "delivery": sorted(self.delivery),
            "measurers": sorted(self.measurers),
            "hooks": sorted(self.hooks),
        }
