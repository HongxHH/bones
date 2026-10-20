from istream_player.models import State
from istream_player.modules.player.player_dash import DASHPlayer


def test_pause_resume_stop_flags():
    player = DASHPlayer()
    player._state = State.READY
    player.pause()
    assert player._paused is True
    assert not player._pause_gate.is_set()
    player.resume()
    assert player._paused is False
    assert player._pause_gate.is_set()
    player.stop()
    assert player._stopped is True
    assert player._pause_gate.is_set()
