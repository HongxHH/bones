# -*- coding: utf-8 -*-
"""幂等补丁：给 pacnet_enhancer.py 加强制出口对照 + 可配 halt 阈值。
通过环境变量切换，不改 yaml、跑完不留痕。
- PACNET_FORCE_EXIT: 0/1/2 强制 e1/e2/e3；-1(默认) 正常 halt
- PACNET_E3_THRESH: e3 阈值倍数（默认1.5，离线最优1.2）
- PACNET_E2_THRESH: e2 阈值倍数（默认1.3，离线最优1.6）
TOS 无 per_seg_gain(gain=1.0)，内容感知中间档恒为简单段，故 _halt_decide 简化为纯预算三档。
"""
import io, sys

p = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\pacnet_enhancer.py"
s = io.open(p, "r", encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"

# 1) __init__ 加环境变量字段（幂等：已存在则跳过）
old_init = (
    '        self._halt_safe_factor = 1.5  # 预算充足判定安全因子（更保守，避免e2/e3积压）' + nl +
    '        self._halt_gain_thresh = 1.5  # 内容难度阈值（gain>此值视为困难段）'
)
new_init = (
    '        self._halt_safe_factor = 1.5  # 预算充足判定安全因子（更保守，避免e2/e3积压）' + nl +
    '        self._halt_gain_thresh = 1.5  # 内容难度阈值（gain>此值视为困难段）' + nl +
    '        # 阶段二实验开关：环境变量控制，跑完不留痕' + nl +
    '        self._force_exit = int(os.environ.get("PACNET_FORCE_EXIT", "-1"))  # -1=正常halt, 0/1/2=强制出口' + nl +
    '        self._halt_e3_thresh = float(os.environ.get("PACNET_E3_THRESH", "1.5"))' + nl +
    '        self._halt_e2_thresh = float(os.environ.get("PACNET_E2_THRESH", "1.3"))'
)
if "PACNET_FORCE_EXIT" not in s:
    assert old_init in s, "init anchor not found"
    s = s.replace(old_init, new_init)
    print("[ok] init: added force_exit/threshold env fields")
else:
    print("[skip] init already patched")

# 2) _halt_decide 简化为纯预算三档（TOS gain=1.0），阈值用可配字段
old_decide = (
    '    def _halt_decide(self, remaining_time, remaining_frames, segment_index):' + nl +
    '        """帧级 halt 控制器：根据剩余时间预算 + 段内容难度决定出口。' + nl +
    '        返回 exit_idx (0=e1, 1=e2, 2=e3)。' + nl +
    '        策略：预算充足全 e3；预算紧张时内容感知（困难段深算、简单段浅算）；预算极少保命 e1。' + nl +
    '        """' + nl +
    '        if remaining_frames <= 0:' + nl +
    '            return 0' + nl +
    '        budget_per_frame = remaining_time / remaining_frames' + nl +
    '        l1, l2, l3 = self.pacnet_latency' + nl +
    '        gain = self.per_seg_gain.get(segment_index, 1.0)' + nl +
    '        is_hard = gain >= self._halt_gain_thresh' + nl +
    nl +
    '        # 预算充足：全 e3（和 greedy 一样）' + nl +
    '        if budget_per_frame >= l3 * self._halt_safe_factor:' + nl +
    '            return 2' + nl +
    '        # 预算够 e3 但不充足：困难段 e3，简单段 e2' + nl +
    '        if budget_per_frame >= l3 * 1.2:' + nl +
    '            return 2 if is_hard else 1' + nl +
    '        # 预算够 e2（带安全因子）：困难段 e2，简单段 e1' + nl +
    '        if budget_per_frame >= l2 * 1.3:' + nl +
    '            return 1 if is_hard else 0' + nl +
    '        # 预算极少：保命 e1' + nl +
    '        return 0'
)
new_decide = (
    '    def _halt_decide(self, remaining_time, remaining_frames, segment_index):' + nl +
    '        """帧级 halt：预算充足 e3，中等 e2，紧张 e1（纯预算三档，阈值环境变量可配）。' + nl +
    '        PACNET_FORCE_EXIT>=0 时强制该出口（全e3对照用）。"""' + nl +
    '        if self._force_exit >= 0:' + nl +
    '            return self._force_exit' + nl +
    '        if remaining_frames <= 0:' + nl +
    '            return 0' + nl +
    '        budget_per_frame = remaining_time / remaining_frames' + nl +
    '        l1, l2, l3 = self.pacnet_latency' + nl +
    '        if budget_per_frame >= l3 * self._halt_e3_thresh:' + nl +
    '            return 2   # e3' + nl +
    '        if budget_per_frame >= l2 * self._halt_e2_thresh:' + nl +
    '            return 1   # e2' + nl +
    '        return 0       # e1 保命'
)
if "PACNET_FORCE_EXIT>=0 时强制" not in s:
    assert old_decide in s, "halt_decide anchor not found"
    s = s.replace(old_decide, new_decide)
    print("[ok] _halt_decide: simplified to budget 3-tier, env-configurable thresholds")
else:
    print("[skip] _halt_decide already patched")

io.open(p, "w", encoding="utf-8", newline="").write(s)
print("done")
