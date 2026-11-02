from pathlib import Path

from istream_player.modules.renderer.frame_clock import FrameClock


def test_frame_clock_writes_rows(tmp_path: Path):
    path = tmp_path / "frame_times.jsonl"
    clock = FrameClock(path)
    clock.write({"kind": "frame", "seg": 3, "gap_ms": 33.2})
    clock.close()
    text = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 1
    assert '"kind":"frame"' in text[0]
    assert '"seg":3' in text[0]
