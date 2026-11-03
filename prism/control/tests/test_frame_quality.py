"""Frame-baseline adapter and Lab scoring priority."""

from __future__ import annotations

import tempfile
from pathlib import Path

from prism.control.metrics.bundle import (
    _FRAME_BASELINE_REASON,
    _override_fc_quality_drop,
    _score_observation,
    _status,
)
from prism.control.metrics.catalog import QualityCatalog
from prism.control.metrics.frame_quality import FrameQualityIndex

ROOT = Path(__file__).resolve().parents[3]


def test_bbb_240_clip_and_segment_means_match_measured_baseline():
    index = FrameQualityIndex(ROOT)
    matrix = index.static_matrix("bbb", "imdn")
    assert matrix is not None
    bicubic, low, medium, high, ultra = matrix[0]
    assert abs(bicubic - 39.296) < 0.01
    assert abs(low - 51.736) < 0.01
    assert abs(medium - 56.079) < 0.01
    assert abs(high - 61.493) < 0.01
    assert ultra < 0
    # Smoke segment (first 120 frames) from the quality_baseline README.
    assert abs(index.segment_mean("bbb", 0, 0, 0) - 51.479) < 0.01
    assert abs(index.segment_mean("bbb", 0, 2, 0) - 57.081) < 0.01
    assert index.frame_score("bbb", 0, 2, 0, 0) is not None
    # Tail segment uses the leftover frames, not a full 120.
    assert index.segment_mean("bbb", 0, 0, 158) is not None
    assert index.segment_mean("bbb", 0, 0, 159) is None
    # 720 has no SR; 1080 reference matches the legacy static cell.
    assert index.segment_mean("bbb", 3, 3, 0) is None
    assert index.segment_mean("bbb", 4, 0, 0) == 100.0
    assert index.segment_mean("bbb", 4, 1, 0) is None


def test_scoring_prefers_frame_baseline_over_analyzer_real_vmaf():
    catalog = QualityCatalog(ROOT, Path(tempfile.mkdtemp()))
    full = _score_observation(
        catalog,
        "bbb",
        {
            "index": 0,
            "repr_index": 0,
            "model": "imdn",
            "level": 2,
            "partial": "none",
            "outcome_source": "analyzer",
            "real_vmaf": 1.0,
        },
    )
    assert full["source"] == "frame_baseline"
    assert abs(full["vmaf"] - 57.081) < 0.01

    mixed = _score_observation(
        catalog,
        "bbb",
        {
            "index": 0,
            "repr_index": 0,
            "model": "imdn",
            "level": 2,
            "partial": "fast_complete",
            "ratio": 0.5,
            "outcome_source": "analyzer",
            "real_vmaf": 1.0,
        },
    )
    assert mixed["source"] == "mixed"
    assert mixed["degraded"] is True
    assert abs(mixed["vmaf"] - 1.0) > 10

    fallback = _score_observation(
        catalog,
        "unknown-content",
        {
            "index": 0,
            "repr_index": 0,
            "model": "imdn",
            "level": 2,
            "partial": "none",
            "outcome_source": "analyzer",
            "real_vmaf": 12.0,
        },
    )
    assert fallback["source"] == "analyzer_fallback"
    assert fallback["vmaf"] == 12.0
    assert fallback["degraded"] is True


def test_fast_complete_prefix_is_not_a_segment_mean_blend():
    index = FrameQualityIndex(ROOT)
    catalog = QualityCatalog(ROOT, Path(tempfile.mkdtemp()))
    enhanced = 40
    prefix = index.playback_mean("bbb", 0, 2, 0, enhanced)
    full = index.segment_mean("bbb", 0, 2, 0)
    plain = index.segment_mean("bbb", 0, 0, 0)
    assert prefix is not None and full is not None and plain is not None
    blended = plain * (1 - enhanced / 120) + full * (enhanced / 120)
    assert abs(prefix - blended) > 0.05

    scored = _score_observation(
        catalog,
        "bbb",
        {
            "index": 0,
            "repr_index": 0,
            "model": "imdn",
            "level": 2,
            "partial": "fast_complete",
            "ratio": enhanced / 120,
            "enhanced_cnt": enhanced,
            "played_frames": 120,
            "enhance_frame_interval": 0,
            "outcome_source": "analyzer",
            "real_vmaf": 1.0,
        },
    )
    assert scored["source"] == "frame_baseline"
    assert abs(scored["vmaf"] - prefix) < 1e-6
    assert abs(scored["quality_drop"] - (full - prefix)) < 1e-6

    derived = {"fast_complete_v1": {"inputs": {"segments": 1, "saved_frames": 80, "quality_drop_avg": 9.0}}}
    _override_fc_quality_drop(derived, {"segments": [scored]})
    assert abs(derived["fast_complete_v1"]["inputs"]["quality_drop_avg"] - scored["quality_drop"]) < 1e-6
    assert derived["fast_complete_v1"]["inputs"]["saved_frames"] == 80


def test_illegal_sr_uses_bicubic_and_status_ignores_mixed():
    index = FrameQualityIndex(ROOT)
    catalog = QualityCatalog(ROOT, Path(tempfile.mkdtemp()))
    plain = index.segment_mean("bbb", 3, 0, 0)
    scored = _score_observation(
        catalog,
        "bbb",
        {
            "index": 0,
            "repr_index": 3,
            "model": "imdn",
            "level": 2,
            "partial": "none",
            "outcome_source": "analyzer",
            "real_vmaf": 1.0,
        },
    )
    assert scored["source"] == "frame_baseline"
    assert abs(scored["vmaf"] - plain) < 1e-6
    assert scored["vmaf"] != 1.0

    status, reasons = _status(
        [{}],
        {"segment_vmafs": [1.0], "segments": [{"vmaf": 1.0, "source": "mixed"}], "degraded": True},
        {},
    )
    assert status == "partial"
    assert _FRAME_BASELINE_REASON not in reasons

    status, reasons = _status(
        [{}],
        {
            "segment_vmafs": [plain],
            "segments": [{"vmaf": plain, "source": "frame_baseline"}],
            "degraded": False,
        },
        {},
    )
    assert _FRAME_BASELINE_REASON in reasons
