"""Reusable quality assets and the local lookup half of QualityResolver.

Playback facts are not stored here. A cell is keyed by content, representation,
and enhancement semantics so IMDN L2 and another model never share a score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .frame_quality import FrameQualityIndex


def normalize_model(name: str) -> str:
    text = (name or "").strip().lower()
    if "pac" in text:
        return "pacnet"
    if "naf" in text:
        return "nafnet"
    if not text or "imdn" in text:
        return "imdn"
    return text


def resolve_content_id(source: str, catalog_ref: str = "") -> str:
    ref = (catalog_ref or "").strip()
    if ref.startswith("content:"):
        token = ref.split(":", 1)[1].strip()
        return token or "unknown"
    blob = f"{source} {ref}".lower()
    for needle, content_id in (
        ("big_buck", "bbb"),
        ("video_mp4box", "bbb"),
        ("bbb", "bbb"),
        ("tears", "tos"),
        ("div2k", "div2k"),
        ("div_quality", "div2k"),
    ):
        if needle in blob:
            return content_id
    return "unknown"


@dataclass(frozen=True)
class QualityKey:
    content_id: str
    metric: str = "vmaf"
    granularity: str = "segment"
    repr_index: int = 0
    model: str = "imdn"
    model_ver: str = ""
    level: int = 0
    scale: str = ""
    partial: str = "none"

    def slug(self) -> str:
        raw = "|".join(
            [
                self.metric,
                self.granularity,
                str(self.repr_index),
                self.model,
                self.model_ver,
                str(self.level),
                self.scale,
                self.partial,
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


@dataclass
class QualityHit:
    value: Optional[float]
    source: str
    degraded: bool = False
    reason: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)
    key: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "degraded": self.degraded,
            "reason": self.reason,
            "provenance": self.provenance,
            "key": self.key,
        }


@dataclass
class QualityAsset:
    key: QualityKey
    scalar: Optional[float] = None
    by_segment: Optional[list[Optional[float]]] = None
    by_frame: Optional[dict[str, list[Optional[float]]]] = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": asdict(self.key),
            "scalar": self.scalar,
            "by_segment": self.by_segment,
            "by_frame": self.by_frame,
            "provenance": self.provenance,
        }


# Scoring tables match the player analyzer (one VMAF per bitrate index × level).
_STATIC_TABLES: tuple[tuple[str, str, str], ...] = (
    ("bbb", "imdn", "player/imdn_bbb_quality.json"),
    ("div2k", "imdn", "player/imdn_div2k_quality.json"),
    ("tos", "imdn", "player/imdn_tos_quality.json"),
)

# Per-segment measurements. Used when the static cell is missing.
_SEGMENT_PACKS: tuple[tuple[str, str, str], ...] = (
    ("bbb", "imdn", "data/bbb/imdn_bbb_3060ti.json"),
)


def _valid(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number < 0:
        return None
    return number


class QualityCatalog:
    def __init__(self, repo_root: Path, store_dir: Path):
        self.repo_root = Path(repo_root)
        self.store_dir = Path(store_dir)
        self._static: dict[tuple[str, str], Optional[list]] = {}
        self._packs: dict[tuple[str, str], Optional[dict]] = {}
        self._assets: dict[tuple, QualityAsset] = {}
        self._store_loaded = False
        self._remote_pulled: set[tuple[str, str]] = set()
        self._compute_attempted: set[str] = set()
        self._explicit: dict[tuple[str, str], Any] = {}
        self._frame_index: Optional[FrameQualityIndex] = None

    def _frames(self) -> FrameQualityIndex:
        if self._frame_index is None:
            self._frame_index = FrameQualityIndex(self.repo_root)
        return self._frame_index

    def static_matrix(self, content_id: str, model: str) -> Optional[list]:
        model = normalize_model(model)
        cache_key = (content_id, model)
        if cache_key in self._static:
            return self._static[cache_key]
        matrix = self._frames().static_matrix(content_id, model)
        if matrix is not None:
            explicit = self._explicit.get(cache_key)
            if isinstance(explicit, list):
                matrix = explicit
            self._static[cache_key] = matrix
            return matrix
        for cid, mid, rel in _STATIC_TABLES:
            if cid == content_id and mid == model:
                path = self.repo_root / rel
                matrix = _load_matrix(path)
                break
        explicit = self._explicit.get(cache_key)
        if isinstance(explicit, list):
            matrix = explicit
        self._static[cache_key] = matrix
        return matrix

    def _from_frame_baseline(
        self,
        content_id: str,
        model: str,
        repr_index: int,
        level: int,
        segment_index: Optional[int],
        frame_in_segment: Optional[int],
        granularity: str,
        key: QualityKey,
    ) -> Optional[QualityHit]:
        index = self._frames()
        if not index.has_content(content_id, model):
            return None
        value: Optional[float] = None
        if granularity == "frame" and segment_index is not None and frame_in_segment is not None:
            value = index.frame_score(
                content_id, repr_index, level, int(segment_index), int(frame_in_segment), model
            )
        elif segment_index is not None:
            value = index.segment_mean(content_id, repr_index, level, int(segment_index), model)
        else:
            matrix = index.static_matrix(content_id, model)
            if matrix and 0 <= repr_index < len(matrix):
                row = matrix[repr_index]
                if 0 <= level < len(row):
                    value = _valid(row[level])
        if value is None:
            return None
        return QualityHit(
            value,
            "frame_baseline",
            provenance={"source": "frame_baseline", "purpose": "quality_baseline"},
            key=asdict(key),
        )

    def segment_pack(self, content_id: str, model: str) -> Optional[dict]:
        model = normalize_model(model)
        cache_key = (content_id, model)
        if cache_key in self._packs:
            return self._packs[cache_key]
        pack = None
        for cid, mid, rel in _SEGMENT_PACKS:
            if cid == content_id and mid == model:
                path = self.repo_root / rel
                if path.is_file():
                    try:
                        pack = json.loads(path.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        pack = None
                break
        explicit = self._explicit.get(cache_key)
        if isinstance(explicit, dict) and "base_quality_vmaf" in explicit:
            pack = explicit
        self._packs[cache_key] = pack
        return pack

    def register_document(self, content_id: str, model: str, document: Any) -> None:
        """Install a table, segment pack, or remote entry list for this process."""
        model = normalize_model(model)
        self._explicit[(content_id, model)] = document
        self._static.pop((content_id, model), None)
        self._packs.pop((content_id, model), None)
        if isinstance(document, dict) and "entries" in document:
            self.import_entries(content_id, document, provenance={"source": "remote"})

    def lookup(
        self,
        *,
        content_id: str,
        repr_index: int,
        model: str,
        level: int,
        segment_index: Optional[int] = None,
        frame_in_segment: Optional[int] = None,
        granularity: str = "segment",
        scale: str = "",
        partial: str = "none",
        model_ver: str = "",
        sources: tuple[str, ...] = ("store", "frame_baseline", "static", "segment_pack"),
    ) -> Optional[QualityHit]:
        model = normalize_model(model)
        key = QualityKey(
            content_id=content_id,
            granularity="frame" if frame_in_segment is not None and granularity == "frame" else granularity,
            repr_index=int(repr_index),
            model=model,
            model_ver=model_ver or "",
            level=int(level),
            scale=scale or "",
            partial=partial or "none",
        )
        if "store" in sources:
            hit = self._from_store(key, segment_index, frame_in_segment)
            if hit is None and (key.scale or key.partial != "none" or key.model_ver):
                relaxed = QualityKey(
                    content_id=key.content_id,
                    metric=key.metric,
                    granularity=key.granularity,
                    repr_index=key.repr_index,
                    model=key.model,
                    level=key.level,
                )
                hit = self._from_store(relaxed, segment_index, frame_in_segment)
            if hit is not None and hit.value is not None:
                return hit
        if granularity == "frame" and "store" in sources:
            # Frame series may live on the segment asset and be aggregated later.
            seg_key = QualityKey(
                content_id=content_id,
                granularity="segment",
                repr_index=int(repr_index),
                model=model,
                level=int(level),
            )
            stored = self._asset(seg_key)
            if stored and stored.by_frame and segment_index is not None:
                frames = stored.by_frame.get(str(segment_index)) or []
                if frame_in_segment is not None and 0 <= frame_in_segment < len(frames):
                    value = _valid(frames[frame_in_segment])
                    if value is not None:
                        return QualityHit(value, stored.provenance.get("source") or "local_store", provenance=stored.provenance, key=asdict(seg_key))
        if "frame_baseline" in sources:
            hit = self._from_frame_baseline(
                content_id,
                model,
                int(repr_index),
                int(level),
                segment_index,
                frame_in_segment,
                granularity,
                key,
            )
            if hit is not None:
                return hit
        if "static" in sources:
            matrix = self.static_matrix(content_id, model)
            if matrix and 0 <= int(repr_index) < len(matrix) and isinstance(matrix[int(repr_index)], list):
                row = matrix[int(repr_index)]
                if 0 <= int(level) < len(row):
                    value = _valid(row[int(level)])
                    if value is None:
                        return None
                    degraded = granularity == "frame"
                    return QualityHit(
                        value,
                        "static_table",
                        degraded=degraded,
                        reason="static table has no per-frame scores" if degraded else "",
                        provenance={"source": "local_table", "purpose": "scoring_table"},
                        key=asdict(key),
                    )
        if "segment_pack" in sources and segment_index is not None:
            value = self._pack_cell(content_id, model, int(repr_index), int(level), int(segment_index))
            if value is not None:
                degraded = granularity == "frame"
                return QualityHit(
                    value,
                    "segment_pack",
                    degraded=degraded,
                    reason="segment measurement reused for a frame" if degraded else "",
                    provenance={"source": "local_table", "purpose": "segment_measurement"},
                    key=asdict(key),
                )
        return None

    def write_asset(self, asset: QualityAsset) -> Path:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        folder = self.store_dir / _safe(asset.key.content_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{asset.key.slug()}.json"
        path.write_text(json.dumps(asset.to_json(), indent=2), encoding="utf-8")
        self._assets[_asset_id(asset.key)] = asset
        self._store_loaded = True
        return path

    def import_entries(self, content_id: str, document: dict, *, provenance: dict) -> int:
        written = 0
        for entry in document.get("entries") or []:
            if not isinstance(entry, dict):
                continue
            model = normalize_model(str(entry.get("model") or "imdn"))
            level = int(entry.get("level") or 0)
            repr_index = int(entry.get("repr_index") or entry.get("repr") or 0)
            granularity = str(entry.get("granularity") or "segment")
            values = entry.get("values")
            key = QualityKey(
                content_id=content_id,
                metric=str(entry.get("metric") or "vmaf"),
                granularity="frame" if granularity == "frame" else "segment",
                repr_index=repr_index,
                model=model,
                model_ver=str(entry.get("model_ver") or ""),
                level=level,
                scale=str(entry.get("scale") or ""),
                partial=str(entry.get("partial") or "none"),
            )
            prov = dict(provenance)
            prov.update(entry.get("provenance") or {})
            prov.setdefault("source", provenance.get("source") or "local_store")
            prov.setdefault("created_at", _now())
            if granularity == "frame" and isinstance(values, dict):
                asset = QualityAsset(key=key, by_frame={str(k): list(v) for k, v in values.items()}, provenance=prov)
            elif isinstance(values, list):
                if granularity == "frame":
                    asset = QualityAsset(key=key, by_frame={"0": [_valid(v) for v in values]}, provenance=prov)
                else:
                    asset = QualityAsset(key=key, by_segment=[_valid(v) for v in values], provenance=prov)
            elif _valid(entry.get("scalar")) is not None or _valid(values) is not None:
                asset = QualityAsset(key=key, scalar=_valid(entry.get("scalar", values)), provenance=prov)
            else:
                continue
            self.write_asset(asset)
            written += 1
        return written

    def _from_store(
        self,
        key: QualityKey,
        segment_index: Optional[int],
        frame_in_segment: Optional[int],
    ) -> Optional[QualityHit]:
        asset = self._asset(key)
        if asset is None and key.granularity == "frame":
            asset = self._asset(
                QualityKey(
                    content_id=key.content_id,
                    metric=key.metric,
                    granularity="segment",
                    repr_index=key.repr_index,
                    model=key.model,
                    model_ver=key.model_ver,
                    level=key.level,
                    scale=key.scale,
                    partial=key.partial,
                )
            )
        if asset is None:
            return None
        source = str(asset.provenance.get("source") or "local_store")
        if frame_in_segment is not None and asset.by_frame and segment_index is not None:
            frames = asset.by_frame.get(str(segment_index)) or []
            if 0 <= frame_in_segment < len(frames):
                value = _valid(frames[frame_in_segment])
                if value is not None:
                    return QualityHit(value, source, provenance=asset.provenance, key=asdict(key))
        if key.granularity == "segment" and asset.by_frame and segment_index is not None:
            frames = [v for v in (asset.by_frame.get(str(segment_index)) or []) if _valid(v) is not None]
            if frames:
                return QualityHit(
                    sum(frames) / len(frames),
                    "aggregated",
                    provenance=asset.provenance,
                    key=asdict(key),
                )
        if asset.by_segment is not None and segment_index is not None and 0 <= segment_index < len(asset.by_segment):
            value = _valid(asset.by_segment[segment_index])
            if value is not None:
                degraded = key.granularity == "frame"
                return QualityHit(
                    value,
                    source,
                    degraded=degraded,
                    reason="segment series reused for a frame" if degraded else "",
                    provenance=asset.provenance,
                    key=asdict(key),
                )
        if asset.scalar is not None:
            degraded = key.granularity == "frame"
            return QualityHit(
                asset.scalar,
                source,
                degraded=degraded,
                reason="scalar quality reused for a frame" if degraded else "",
                provenance=asset.provenance,
                key=asdict(key),
            )
        return None

    def _asset(self, key: QualityKey) -> Optional[QualityAsset]:
        self._load_store()
        return self._assets.get(_asset_id(key))

    def _load_store(self) -> None:
        if self._store_loaded:
            return
        self._store_loaded = True
        if not self.store_dir.is_dir():
            return
        for path in self.store_dir.glob("*/*.json"):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            key_raw = raw.get("key") or {}
            try:
                key = QualityKey(
                    content_id=str(key_raw.get("content_id") or path.parent.name),
                    metric=str(key_raw.get("metric") or "vmaf"),
                    granularity=str(key_raw.get("granularity") or "segment"),
                    repr_index=int(key_raw.get("repr_index") or 0),
                    model=normalize_model(str(key_raw.get("model") or "imdn")),
                    model_ver=str(key_raw.get("model_ver") or ""),
                    level=int(key_raw.get("level") or 0),
                    scale=str(key_raw.get("scale") or ""),
                    partial=str(key_raw.get("partial") or "none"),
                )
            except (TypeError, ValueError):
                continue
            self._assets[_asset_id(key)] = QualityAsset(
                key=key,
                scalar=_valid(raw.get("scalar")),
                by_segment=raw.get("by_segment"),
                by_frame=raw.get("by_frame"),
                provenance=dict(raw.get("provenance") or {}),
            )

    def _pack_cell(self, content_id: str, model: str, repr_index: int, level: int, segment_index: int) -> Optional[float]:
        pack = self.segment_pack(content_id, model)
        if not pack:
            return None
        base = pack.get("base_quality_vmaf") or []
        enh = pack.get("enhanced_quality_vmaf") or []
        if not base or segment_index < 0:
            return None
        n_low = len(base)
        n_seg = len(base[0]) if isinstance(base[0], list) else 0
        if segment_index >= n_seg:
            return None
        if level <= 0:
            if repr_index < n_low and isinstance(base[repr_index], list):
                return _valid(base[repr_index][segment_index])
            if repr_index == n_low:
                return 100.0
            return None
        method = level - 1
        if repr_index < 0 or repr_index >= len(enh):
            return None
        methods = enh[repr_index]
        if not isinstance(methods, list) or method < 0 or method >= len(methods):
            return None
        series = methods[method]
        if not isinstance(series, list) or segment_index >= len(series):
            return None
        return _valid(series[segment_index])


def _asset_id(key: QualityKey) -> tuple:
    return (
        key.content_id,
        key.metric,
        key.granularity,
        key.repr_index,
        key.model,
        key.model_ver,
        key.level,
        key.scale,
        key.partial,
    )


def _safe(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text.strip())
    return cleaned or "unknown"


def _load_matrix(path: Path) -> Optional[list]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(data, list):
        return data
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


_CATALOGS: dict[tuple[str, str], QualityCatalog] = {}


def catalog_for(repo_root: Path, data_dir: Path) -> QualityCatalog:
    key = (str(repo_root), str(Path(data_dir) / "quality_catalog"))
    found = _CATALOGS.get(key)
    if found is None:
        found = QualityCatalog(Path(repo_root), Path(data_dir) / "quality_catalog")
        _CATALOGS[key] = found
    return found
