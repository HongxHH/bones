"""Original-vs-enhanced baseline: off by default, LR bicubic only when asked."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "player"))

from istream_player.config.config import PlayerConfig
from istream_player.modules.enhancer.compare_view import (
    drop_segment,
    enabled,
    keep_input,
    keep_same,
    pad_same,
    start_segment,
    take,
    to_display_bgr,
)
from istream_player.modules.renderer.frame_archive import FrameArchive
from prism.frame_sink import BASE_NAME, NAME


class _Seg:
    pass


class _Cfg:
    compare_view = False


def test_compare_defaults_off():
    assert PlayerConfig().compare_view is False
    assert enabled(PlayerConfig()) is False
    assert BASE_NAME == NAME + "Base"


def test_segment_keeps_nothing_until_enabled():
    segment = _Seg()
    start_segment(segment, _Cfg())
    keep_same(segment)
    pad_same(segment, 4)
    assert segment.compare_lr is None
    assert take(segment, 0) is None


def test_enabled_segment_aligns_and_drops():
    import torch

    cfg = _Cfg()
    cfg.compare_view = True
    segment = _Seg()
    start_segment(segment, cfg)
    keep_same(segment)
    lr = torch.zeros(1, 3, 2, 2)
    lr[0, 0] = 1
    keep_input(segment, lr)
    pad_same(segment, 1)
    assert take(segment, 0) is None
    stored = take(segment, 1)
    assert stored is not None
    assert stored.dtype == torch.uint8
    assert tuple(stored.shape) == (3, 2, 2)
    assert int(stored[0, 0, 0]) == 255
    assert take(segment, 2) is None
    drop_segment(segment)
    assert segment.compare_lr is None


def test_bicubic_baseline_matches_display_size():
    import torch

    lr = torch.zeros(3, 4, 4, dtype=torch.uint8)
    lr[0] = 255
    bgr = to_display_bgr(lr, 8, 6)
    assert bgr.shape == (6, 8, 3)
    assert bgr.dtype == np.uint8
    # Stored plane 0 is RGB red, which is BGR channel 2.
    assert int(bgr[0, 0, 2]) > 200
    assert int(bgr[0, 0, 0]) < 40


def test_archive_writes_baseline_jpeg(tmp_path: Path):
    archive = FrameArchive(str(tmp_path / "arm"))
    picture = np.zeros((6, 8, 3), np.uint8)
    picture[:, :] = (0, 0, 255)
    baseline = np.zeros((6, 8, 3), np.uint8)
    baseline[:, :] = (255, 0, 0)
    assert archive.submit(picture, {"segment": 1, "enhanced": True}, baseline) == 0
    archive.close()
    row = (tmp_path / "arm" / "frames.jsonl").read_text(encoding="utf-8")
    assert '"has_baseline": true' in row
    assert (tmp_path / "arm" / "frames" / "000000.base.jpg").read_bytes().startswith(b"\xff\xd8")
