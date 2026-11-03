"""Simulation loader and the optional BONES frame-catalog table."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "player"))
sys.path.insert(0, str(ROOT / "honghong"))

from control.frame_quality_results import catalog_decision_table, load_frame_results, load_per_frame_data
from istream_player.modules.enhancer.enhancer_imdn_always import IMDNEnhancerAlways
from prism.control.metrics.bundle import _score_observation
from prism.control.metrics.catalog import QualityCatalog
from prism.control.metrics.frame_quality import FrameQualityIndex, quality_segment_index


def test_loader_matches_segment_means_and_skips_illegal_sr():
    index = FrameQualityIndex(ROOT)
    packed = load_frame_results(ROOT, "bbb")
    results = packed["results"]
    for name, level in (("base", 0), ("sr_medium", 2)):
        frames = results[f"240_{name}_0"]
        assert len(frames) == 120
        assert abs(sum(frames) / len(frames) - index.segment_mean("bbb", 0, level, 0)) < 1e-9
    assert "720_base_0" in results
    assert "720_sr_medium_0" not in results
    assert load_per_frame_data(SimpleNamespace(per_frame_path=None), ROOT)["results"]


def test_quality_source_static_keeps_cached_table_and_catalog_uses_segment():
    cached = np.full((5, 5), 1.0)
    stub = SimpleNamespace(
        quality_table=cached,
        video_dataset="bbb",
        content_aware=False,
        config=SimpleNamespace(quality_source="static"),
        _frame_quality_index=None,
    )
    assert IMDNEnhancerAlways.get_quality_table(stub, segment_index=0) is cached

    stub.config.quality_source = "frame_catalog"
    table = IMDNEnhancerAlways.get_quality_table(stub, segment_index=0)
    expected = FrameQualityIndex(ROOT).segment_mean("bbb", 0, 2, 0)
    div2k = 46.495838515723264
    assert abs(float(table[0, 2]) - expected) < 1e-6
    assert abs(float(table[0, 2]) - div2k) > 1
    assert not np.isfinite(table[3, 2])
    assert table[4, 0] == 100.0


def test_mpd_segment_numbers_map_onto_frame_zero():
    assert quality_segment_index(1, 1) == 0
    assert quality_segment_index(2, 1) == 1
    index = FrameQualityIndex(ROOT)
    catalog = QualityCatalog(ROOT, Path(tempfile.mkdtemp()))
    scored = _score_observation(
        catalog,
        "bbb",
        {
            "index": 1,
            "quality_index": quality_segment_index(1, 1),
            "repr_index": 0,
            "model": "imdn",
            "level": 2,
            "partial": "none",
        },
    )
    assert abs(scored["vmaf"] - index.segment_mean("bbb", 0, 2, 0)) < 1e-6


def test_missing_segment_falls_back_to_clip_mean():
    index = FrameQualityIndex(ROOT)
    clip = index.static_matrix("bbb")
    missing = index.decision_matrix("bbb", 999)
    assert abs(missing[0][0] - clip[0][0]) < 1e-9
    assert abs(missing[0][2] - clip[0][2]) < 1e-9
    assert missing[3][2] is None
    present = index.decision_matrix("bbb", 0)
    assert abs(present[0][2] - index.segment_mean("bbb", 0, 2, 0)) < 1e-9
    assert abs(present[0][2] - clip[0][2]) > 0.05

    table = catalog_decision_table(999, ROOT)
    assert abs(float(table[0, 0]) - clip[0][0]) < 1e-6
    assert not np.isfinite(table[3, 2])

    played = index.played_score("bbb", 0, 2, 0, enhanced_frames=40, interval=0)
    prefix = index.playback_mean("bbb", 0, 2, 0, 40)
    assert abs(played["real"] - prefix) < 1e-9
    assert abs(played["full"] - played["real"]) > 0.05
