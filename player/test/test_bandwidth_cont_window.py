"""Sliding-window samples stay bounded; estimate matches the old reverse scan."""

from __future__ import annotations

import asyncio
from collections import deque

from istream_player.modules.bw_meter.bandwidth_cont import ContinuousBandwidthMeter


def _legacy_mean(samples, time_at: float, window: float, min_values: int = 2):
    window_start = time_at - window
    window_values = []
    for bw in samples[::-1]:
        if bw[1] < window_start and len(window_values) >= min_values:
            break
        window_values.append(bw)
    total_bytes = sum(bw[2] for bw in window_values)
    total_time = sum(bw[1] - bw[0] for bw in window_values)
    return int(8 * total_bytes / total_time)


class _Meter(ContinuousBandwidthMeter):
    def get_stats(self, url: str):
        return None


def test_cont_bw_trims_and_matches_legacy():
    meter = _Meter()
    meter.cont_bw_window = 1.0
    meter.listeners = []
    meter.first_byte_in_segment = True
    meter.last_byte_at = 0.0

    async def drive():
        # First byte only primes the clock.
        await meter.update_cont_bw(1000, 10.0)
        legacy = []
        for i in range(1, 200):
            t = 10.0 + i * 0.05
            nbytes = 1000 + i
            legacy.append((meter.last_byte_at, t, nbytes))
            await meter.update_cont_bw(nbytes, t)
            if len(legacy) < 2:
                continue
            expected = _legacy_mean(legacy, t, meter.cont_bw_window)
            assert meter.last_cont_bw == expected
            assert len(meter._cont_bw) <= 25
            assert isinstance(meter._cont_bw, deque)

    asyncio.run(drive())
