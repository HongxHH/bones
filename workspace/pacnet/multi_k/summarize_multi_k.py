# -*- coding: utf-8 -*-
"""阶段二多轨迹 K=1.5 汇总：3 trace x (halt/force_e3/imdn)，算 Δ、胜率、中位数。"""
import glob, json, os, statistics

D = r"E:\dev\Code\Python\bones\player\output\data"
OUT = r"E:\dev\Code\Python\bones\workspace\pacnet\multi_k\summary_multi_k.json"

TRACE = {
    "0913": "report.2010-09-13_1003CEST",
    "0914": "report.2010-09-14_1038CEST",
    "0921": "report.2010-09-21_1001CEST",
}
# (trace, strategy) -> 结果文件时间戳
TS = {
    ("0913", "halt"): "20260918-113913",
    ("0913", "force_e3"): "20260918-115153",
    ("0913", "imdn"): "20260918-120420",
    ("0914", "halt"): "20260918-103522",
    ("0914", "force_e3"): "20260918-104856",
    ("0914", "imdn"): "20260918-110254",
    ("0921", "halt"): "20260918-121806",
    ("0921", "force_e3"): "20260918-123213",
    ("0921", "imdn"): "20260918-124549",
}
NES = {"halt": "bones_pacnet", "force_e3": "bones_pacnet", "imdn": "bones_imdn_always"}


def load(trace, strat):
    ts = TS[(trace, strat)]
    hits = glob.glob(os.path.join(D, f"*_{ts}.json"))
    if not hits:
        raise FileNotFoundError(f"{trace}/{strat} ts={ts}")
    j = json.load(open(hits[0], encoding="utf-8"))
    rec = {
        "file": os.path.basename(hits[0]),
        "vmaf": j.get("avg_vmaf"),
        "qoe": j.get("QOE"),
        "num_stall": j.get("num_stall"),
        "dur_stall": j.get("dur_stall"),
        "fc": j.get("fast_complete_segments_count"),
        "level_count": j.get("enhance_level_count"),
    }
    # PAC 出口帧分布
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
    for st in ("halt", "force_e3", "imdn"):
        runs[(tr, st)] = load(tr, st)

# 明细表
print("=" * 118)
print(f"{'trace':<6}{'strategy':<10}{'VMAF':>7}{'QoE':>8}{'stall#':>8}{'stall(s)':>9}{'fcSeg':>6}  出口分布 / level_count")
print("-" * 118)
for tr in TRACE:
    for st in ("halt", "force_e3", "imdn"):
        r = runs[(tr, st)]
        extra = ""
        if r.get("exit_pct"):
            extra = "e1/e2/e3=%s" % "/".join(f"{x:.0f}" for x in r["exit_pct"]) + "%"
        elif r.get("level_count"):
            extra = "l0-l3=%s" % r["level_count"]
        print(f"{tr:<6}{st:<10}{r['vmaf']:>7.2f}{r['qoe']:>8.2f}{str(r['num_stall']):>8}"
              f"{r['dur_stall']:>9.2f}{str(r['fc']):>6}  {extra}")
    print("-" * 118)

# Δ 与胜率
def collect(metric):
    return {tr: {st: runs[(tr, st)][metric] for st in ("halt", "force_e3", "imdn")} for tr in TRACE}


def report(metric, label, lower_better=False, unit=""):
    print(f"\n### {label}")
    d_fe, d_im = [], []
    for tr in TRACE:
        h = runs[(tr, "halt")][metric]
        fe = runs[(tr, "force_e3")][metric]
        im = runs[(tr, "imdn")][metric]
        a = h - fe
        b = h - im
        d_fe.append(a); d_im.append(b)
        print(f"  {tr}: halt={h:.2f} force_e3={fe:.2f} imdn={im:.2f} | Δ(halt-fe)={a:+.2f}  Δ(halt-imdn)={b:+.2f}{unit}")
    def wins(x):
        return sum(1 for v in x if v < 0) if lower_better else sum(1 for v in x if v > 0)
    print(f"  halt 对 force_e3：{'优' if not lower_better else '低/优'} {wins(d_fe)}/3，中位 Δ={statistics.median(d_fe):+.2f}{unit}，均值 Δ={statistics.mean(d_fe):+.2f}{unit}")
    print(f"  halt 对 imdn    ：{'优' if not lower_better else '低/优'} {wins(d_im)}/3，中位 Δ={statistics.median(d_im):+.2f}{unit}，均值 Δ={statistics.mean(d_im):+.2f}{unit}")


report("vmaf", "avg_vmaf（越高越好）")
report("qoe", "QoE（越高越好）")
report("dur_stall", "卡顿总时长（越低越好）", lower_better=True, unit="s")

# fc 对比
print("\n### fast_complete 段数（越低越好）")
for tr in TRACE:
    print(f"  {tr}: halt={runs[(tr,'halt')]['fc']}  force_e3={runs[(tr,'force_e3')]['fc']}  imdn={runs[(tr,'imdn')]['fc']}")

json.dump({"runs": {f"{k[0]}_{k[1]}": v for k, v in runs.items()}}, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\n[saved] {OUT}")
