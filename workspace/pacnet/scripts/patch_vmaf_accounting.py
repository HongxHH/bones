# -*- coding: utf-8 -*-
"""补丁：analyzer 按 PACNet 帧级出口占比计算真实 VMAF（幂等）。
保持 analyzer.py 原有 CRLF 行尾。"""
import io, os

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "player", "istream_player", "modules", "analyzer", "analyzer.py")

with io.open(PATH, "r", encoding="utf-8", newline="") as f:
    raw = f.read()
nl = "\r\n" if "\r\n" in raw else "\n"
s = raw.replace("\r\n", "\n")

ANCHOR = (
"                if analyzer_segment.is_enhance:\n"
"                    e_idx = int(analyzer_segment.enhance_action)\n"
"                    # 这里我们需要更加精细化的QOE,因为即使该段进行了增强，但是也可能触发了Fast模式，仅增强了部分，因此还要扣除没有增强的部分\n"
"                    if analyzer_segment.fast_complete:\n"
)

NEW = (
"                if analyzer_segment.is_enhance:\n"
"                    e_idx = int(analyzer_segment.enhance_action)\n"
"                    # PACNet 帧级 halt：按段内 [e1,e2,e3] 实际增强帧数精确计算 VMAF，\n"
"                    # 未增强帧（fast-complete 截断部分）按 base_vmaf 计，避免用下发动作 e_idx 虚高打分。\n"
"                    _efc = analyzer_segment.exit_frame_counts\n"
"                    if _efc is not None and sum(int(c) for c in _efc) > 0:\n"
"                        _cnts = [int(c) for c in _efc]\n"
"                        _n_enh = sum(_cnts)\n"
"                        _n_non = int(analyzer_segment.interpolated_cnt) if analyzer_segment.interpolated_cnt else 0\n"
"                        _vmaf_enh = sum(_cnts[i] * float(self.quality_table[q_idx, i + 1])\n"
"                                        for i in range(3)) / _n_enh\n"
"                        _tot = _n_enh + _n_non\n"
"                        analyzer_segment.enhance_vmaf = _vmaf_enh\n"
"                        analyzer_segment.enhance_ratio = _n_enh / _tot if _tot > 0 else 1.0\n"
"                        analyzer_segment.real_vmaf = (\n"
"                            (_vmaf_enh * _n_enh + analyzer_segment.base_vmaf * _n_non) / _tot\n"
"                            if _tot > 0 else _vmaf_enh)\n"
"                        analyzer_segment.enhanced_frame_vmaf = _vmaf_enh\n"
"                        analyzer_segment.non_enhanced_frame_vmaf = analyzer_segment.base_vmaf\n"
"                        analyzer_segment.quality_drop_due_to_fast = _vmaf_enh - analyzer_segment.real_vmaf\n"
"                        analyzer_segment.quality_switches_within_segment = 1 if _n_non > 0 else 0\n"
"                    # 这里我们需要更加精细化的QOE,因为即使该段进行了增强，但是也可能触发了Fast模式，仅增强了部分，因此还要扣除没有增强的部分\n"
"                    elif analyzer_segment.fast_complete:\n"
)

if "_efc = analyzer_segment.exit_frame_counts" in s:
    print("[skip] analyzer PACNet VMAF accounting already patched")
elif ANCHOR not in s:
    raise SystemExit("[error] anchor not found in analyzer.py")
else:
    s = s.replace(ANCHOR, NEW, 1)
    with io.open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(s.replace("\n", nl))
    print("[ok] analyzer.py patched (frame-level exit VMAF accounting), newline=%s" % repr(nl))
