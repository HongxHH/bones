from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from ..config import ServerConfig

T = TypeVar("T", bound=BaseModel)


class LocalAssetStore:
    """Filesystem object store + per-entity JSON metadata."""

    def __init__(self, config: ServerConfig):
        self.config = config
        config.ensure_dirs()

    def asset_dir(self, asset_id: str) -> Path:
        path = self.config.assets_dir / asset_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def publication_dir(self, publication_id: str) -> Path:
        path = self.config.publications_dir / publication_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def experiment_dir(self, experiment_id: str) -> Path:
        path = self.config.experiments_dir / experiment_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def staging_dir(self, job_id: str) -> Path:
        path = self.config.staging_dir / job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def copy_tree(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    def copy_file(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def write_text(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def meta_path(self, name: str) -> Path:
        return self.config.meta_dir / f"{name}.json"

    def collection_dir(self, name: str) -> Path:
        path = self.config.meta_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def entity_path(self, collection: str, entity_id: str) -> Path:
        return self.collection_dir(collection) / f"{entity_id}.json"

    def save_entity(self, collection: str, item: BaseModel) -> None:
        entity_id = getattr(item, "id", None)
        if not entity_id:
            raise ValueError(f"entity in {collection} missing id")
        path = self.entity_path(collection, entity_id)
        path.write_text(json.dumps(item.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")

    def delete_entity(self, collection: str, entity_id: str) -> None:
        path = self.entity_path(collection, entity_id)
        if path.exists():
            path.unlink()

    def load_collection(self, name: str, model: type[T]) -> list[T]:
        """Load per-entity files; migrate legacy single-list JSON if present."""
        directory = self.collection_dir(name)
        items: list[T] = []
        for path in sorted(directory.glob("*.json")):
            try:
                items.append(model.model_validate(json.loads(path.read_text(encoding="utf-8-sig"))))
            except Exception:
                continue
        if items:
            return items
        legacy = self.meta_path(name)
        if legacy.exists():
            raw = json.loads(legacy.read_text(encoding="utf-8-sig"))
            migrated = [model.model_validate(item) for item in raw]
            for item in migrated:
                self.save_entity(name, item)
            return migrated
        return []

    def load_list(self, name: str, model: type[T]) -> list[T]:
        return self.load_collection(name, model)

    def save_list(self, name: str, items: list[BaseModel]) -> None:
        """Compatibility: rewrite collection as per-entity files."""
        directory = self.collection_dir(name)
        keep = set()
        for item in items:
            self.save_entity(name, item)
            keep.add(getattr(item, "id"))
        for path in directory.glob("*.json"):
            if path.stem not in keep:
                path.unlink()
