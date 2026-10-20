import numpy as np

from prism.frame_sink import FrameSink
from prism.control.attach import player_cmd


def test_frame_roundtrip():
    sink = FrameSink("PrismFrameTest", create=True)
    try:
        sink.open()
        rgb = np.zeros((8, 8, 3), np.uint8)
        rgb[0, 0] = (9, 8, 7)
        assert sink.write(rgb)
        got = sink.read()
        assert got is not None
        seq, frame = got
        assert seq > 0 and seq % 2 == 0
        assert tuple(int(v) for v in frame[0, 0]) == (9, 8, 7)
    finally:
        sink.close()


def test_player_cmd_includes_frame_shm():
    cmd = player_cmd(
        "http://127.0.0.1:8080/manifest.mpd",
        {"mod_nes": "always_greedy"},
        8765,
        frame_shm="PrismFrame",
    )
    assert "--frame_shm" in cmd
    assert cmd[cmd.index("--frame_shm") + 1] == "PrismFrame"


def test_player_cmd_collect_system_metrics_flag():
    on = player_cmd("http://x/m.mpd", {"collect_system_metrics": True}, 8765)
    assert "--collect_system_metrics" in on
    assert on[on.index("--collect_system_metrics") + 1] == "true"
    off = player_cmd("http://x/m.mpd", {"collect_system_metrics": False}, 8765)
    assert off[off.index("--collect_system_metrics") + 1] == "false"
