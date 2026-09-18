# -*- coding: utf-8 -*-
"""halt 前瞻安全系数扫描汇总（K=1.5，TOS，headless）。
两条 trace（0921 最差 / 0914 居中），E3=E2 系数 1.0/1.1/1.2/1.3，另加 IMDN 基线行。"""
import glob, json, os, statistics

D = r"E:\dev\Code\Python\bones\player\output\data"
OUT = r"E:\dev\Code\Python\bones\workspace\pacnet\sweep_thresh\summary_sweep.json"

TRACE = ["0921", "0914"]
TNAME = {"0921": "report.2010-09-21_1001CEST", "0914": "report.2010-09-14_1038CEST"}
# (trace, label) -> 结果文件时间戳；s10=系数1.0 halt 基线（multi_k 已跑）
TS = {
    ("0921", "s10"): "20260918-121806",
    ("0921", "s11"): "20260918-144402",
    ("0921", "s12"): "20260918-145754",
    ("0921", "s13"): "20260918-151143",
    ("0921", "imdn"): "20260918-124549",
    ("0914", "s10"): "20260918-103522",
    ("0914", "s11"): "20260918-152411",
    ("0914", "s12"): "20260918-153644",
    ("0914", "s13"): "20260918-154912",
    ("0914", "imdn"): "20260918-110254",
}
ORDER = ["s10", "s11", "s12", "s13", "imdn"]
LABEL = {"s10": "halt 1.0", "s11": "halt 1.1", "s12": "halt 1.2", "s13": "halt 1.3", "imdn": "IMDN 基线"}


def load(trace, lab):
    ts = TS[(trace, lab)]
    hits = glob.glob(os.path.join(D, f"*_{ts}.json"))
    if not hits:
        raise FileNotFoundError(f"{trace}/{lab} ts={ts}")
    j = json.load(open(hits[0], encoding="utf-8"))
    rec = {
        "file": os.path.basename(hits[0]),
        "vmaf": j.get("avg_vmaf"), "qoe": j.get("QOE"),
        "num_stall": j.get("num_stall"), "dur_stall": j.get("dur_stall"),
        "fc": j.get("fast_complete_segments_count"),
        "level_count": j.get("enhance_level_count"),
    }
    tot = [0, 0, 0]
    for s in j.get("segments", []):
        efc = s.get("exit_frame_counts")
        if efc:
            for i in range(3):
                tot[i] += int(efc[i] or 0)
    if sum(tot) > 0:
        s_ = sum(tot)
        rec["exit_pct"] = [round(100 * tot[i] / s_, 1) for i in range(3)]
        rec["exit_frames"] = tot
    return rec


runs = {}
for tr in TRACE:
    for lab in ORDER:
        runs[(tr, lab)] = load(tr, lab)

for tr in TRACE:
    print("=" * 112)
    print(f"trace {tr}  ({TNAME[tr]})")
    print(f"{'strategy':<12}{'VMAF':>8}{'QoE':>8}{'stall#':>8}{'stall(s)':>10}{'fcSeg':>7}  出口 e1/e2/e3 %")
    print("-" * 112)
    im = runs[(tr, "imdn")]
    for lab in ORDER:
        r = runs[(tr, lab)]
        extra = ""
        if r.get("exit_pct"):
            extra = "/".join(f"{x:.0f}" for x in r["exit_pct"]) + "%"
        elif r.get("level_count"):
            extra = "l0-l3=%s" % r["level_count"]
        print(f"{LABEL[lab]:<12}{r['vmaf']:>8.2f}{r['qoe']:>8.2f}{str(r['num_stall']):>8}"
              f"{r['dur_stall']:>10.2f}{str(r['fc']):>7}  {extra}")
    print("-" * 112)
    print(f"{'Δ vs IMDN':<12}{'ΔVMAF':>8}{'ΔQoE':>8}{'':>8}{'Δstall(s)':>10}")
    for lab in ["s10", "s11", "s12", "s13"]:
        r = runs[(tr, lab)]
        print(f"{LABEL[lab]:<12}{r['vmaf']-im['vmaf']:>+8.2f}{r['qoe']-im['qoe']:>+8.2f}{'':>8}"
              f"{r['dur_stall']-im['dur_stall']:>+10.2f}")
    print()

# 拐点判定：跨两 trace 汇总每系数相对 IMDN
print("=" * 70)
print("跨两 trace：各系数相对 IMDN 的中位 Δ")
print(f"{'系数':<8}{'中位ΔVMAF':>12}{'中位ΔQoE':>12}{'中位Δstall(s)':>16}")
for lab in ["s10", "s11", "s12", "s13"]:
    dv = [runs[(tr, lab)]["vmaf"] - runs[(tr, "imdn")]["vmaf"] for tr in TRACE]
    dq = [runs[(tr, lab)]["qoe"] - runs[(tr, "imdn")]["qoe"] for tr in TRACE]
    ds = [runs[(tr, lab)]["dur_stall"] - runs[(tr, "imdn")]["dur_stall"] for tr in TRACE]
    print(f"{LABEL[lab]:<8}{statistics.median(dv):>+12.2f}{statistics.median(dq):>+12.2f}"
          f"{statistics.median(ds):>+16.2f}")

json.dump({"runs": {f"{k[0]}_{k[1]}": v for k, v in runs.items()}},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n[saved] {OUT}")
