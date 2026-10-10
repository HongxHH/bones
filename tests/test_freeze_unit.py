# -*- coding: utf-8 -*-
"""
M0 验收：仿真器 QoE 卡顿惩罚单位统一为秒（与播放器 analyzer 同口径）。

背景：simulation/simulator.py 与 control/bones_*.py 的 report() 历史上直接把
BufferModel.idle_time（毫秒）除以段数后 /10，1 秒卡顿惩罚 100；播放器
player/istream_player/modules/analyzer/analyzer.py:801,806 用秒 /10，1 秒卡顿
惩罚 0.1，两边尺度差 1000 倍（档案 md_qoe C1b）。

运行（仓库根目录）：
  E:\\dev\\env\\miniconda3\\envs\\SR\\python.exe tests\\test_freeze_unit.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulation.simulator import Simulator
from simulation.buffer_model import BufferModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeLogger:
    def is_verbose(self):
        return False

    def summarize(self, *args, **kwargs):
        pass


def make_sim(idle_ms, num_seg=2):
    """构造最小填充的 Simulator（不走 __init__ 的文件解析），只喂 report() 需要的字段。"""
    sim = Simulator.__new__(Simulator)
    sim.global_time = 0
    sim.seg_time = 4000
    n_bitrate, n_method = 3, 4
    sim.seg_size = np.ones((num_seg, n_bitrate), dtype=np.int64) * 1000
    sim.vmaf_enh_seg = np.full((num_seg, n_bitrate, n_method), 80.0)
    sim.ssim_enh_seg = np.ones((num_seg, n_bitrate, n_method)) * 0.9
    dl = BufferModel(FakeLogger(), 10 ** 9, num_seg, "download")
    eh = BufferModel(FakeLogger(), 10 ** 9, num_seg, "enhance")
    dl.history[:] = 1
    eh.history[:] = 0
    dl.idle_time = idle_ms
    sim.buffer_download = dl
    sim.buffer_enhance = eh
    sim.logger = FakeLogger()
    return sim


def test_report_freeze_in_seconds():
    # 2 段共卡顿 2000ms = 平均每段 1 秒；质量 80、振荡 0；QoE 应为 80 - 1/10 = 79.9
    sim = make_sim(idle_ms=2000, num_seg=2)
    rep = sim.report()
    assert rep["freeze time"] == 2000, rep["freeze time"]
    assert abs(rep["freeze time sec"] - 2.0) < 1e-9, rep["freeze time sec"]
    assert abs(rep["qoe"] - 79.9) < 1e-9, rep["qoe"]
    # 旧 ms/10 口径会给出 80 - 1000/10 = -20，这里同时守住"不得退回旧口径"
    assert rep["qoe"] > 0


def test_zero_freeze():
    sim = make_sim(idle_ms=0, num_seg=2)
    rep = sim.report()
    assert abs(rep["qoe"] - 80.0) < 1e-9, rep["qoe"]


def test_all_four_report_sites_divide_ms_by_1000():
    """静态守卫：四处 report 的 avg_freeze 必须先 /1000 再 /段数，禁止裸 ms /10。"""
    targets = [
        (os.path.join(REPO, "simulation", "simulator.py"),
         "/ 1000.0 / self.seg_size.shape[0]"),
        (os.path.join(REPO, "control", "bones_frame_level.py"),
         "/ 1000.0 / num_seg"),
        (os.path.join(REPO, "control", "bones_incremental.py"),
         "/ 1000.0 / num_seg"),
        (os.path.join(REPO, "control", "bones_combined.py"),
         "/ 1000.0 / num_seg"),
    ]
    for path, needle in targets:
        with open(path, "r", encoding="utf-8") as f:
            src = f.read()
        assert needle in src, f"{path} 缺少秒口径换算 {needle}"
        # 不应再存在"毫秒直接除段数"的 avg_freeze 行
        for line in src.splitlines():
            if "avg_freeze =" in line:
                assert "/ 1000.0" in line, f"{path} 仍存在 ms 口径 avg_freeze: {line.strip()}"


if __name__ == "__main__":
    test_report_freeze_in_seconds()
    test_zero_freeze()
    test_all_four_report_sites_divide_ms_by_1000()
    print("M0 freeze-unit tests PASSED")
