"""M5 步骤3：六真实轨迹 × 四策略回归（每策略独立仿真器，卡顿向量独立，md_qoe C2）。

轨迹：3 条 3G（0913/0914/0921）+ 3 条 4G（bus/car/bicycle，编号 0001，
与 lookahead 多轨迹口径一致）。
策略：BONES 基线；FL/INC/COMB 默认 EDF 因果版；另跑 oracle 给真实轨迹头间。
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from control.bones import BONES
from control.bones_frame_level import BONESFrameLevel
from control.bones_incremental import BONESIncremental
from control.bones_combined import BONESCombined

ENHANCE_PATH = os.path.join(REPO, r"data\bbb\imdn_bbb_3060ti.json")
LOG_DIR = os.path.join(REPO, r"workspace\sim_logs_v3")

TRACES = [
    ("3g_0913", r"data\trace\3Glogs\report.2010-09-13_1003CEST.json"),
    ("3g_0914", r"data\trace\3Glogs\report.2010-09-14_1038CEST.json"),
    ("3g_0921", r"data\trace\3Glogs\report.2010-09-21_1001CEST.json"),
    ("4g_bus", r"data\trace\4Glogs\report_bus_0001.json"),
    ("4g_car", r"data\trace\4Glogs\report_car_0001.json"),
    ("4g_bicycle", r"data\trace\4Glogs\report_bicycle_0001.json"),
]
CONTROLLERS = [("fl", BONESFrameLevel), ("inc", BONESIncremental),
               ("comb", BONESCombined)]


def make_args(trace_path, oracle_rank):
    args = argparse.Namespace()
    args.log_path = os.path.join(LOG_DIR, "m5_regression.log")
    args.manifest_path = os.path.join(REPO, r"data\bbb\bbb_video_info.json")
    args.trace_path = trace_path
    args.enhance_path = ENHANCE_PATH
    args.video_length = None
    args.throughput = "ewma"
    args.download_buffer = 60
    args.enhance_buffer = 60
    args.bandwidth_multiplier = 1.0
    args.speed_multiplier = 1.0
    args.verbose = False
    args.gamma_p = 10.0
    args.V_multiplier = 1.0
    args.monitor = False
    args.autotune = False
    args.quick_start = False
    args.per_frame_path = None
    args.oracle_rank = oracle_rank
    return args


def run_one(cls, trace_path, oracle_rank):
    sim = cls(make_args(trace_path, oracle_rank))
    sim.run()
    r = sim.report()
    return {
        "vmaf": r["average quality vmaf"],
        "qoe": r["qoe"],
        "osc": r["average oscillation vmaf"],
        "freeze_s": r["freeze time sec"],
        "freeze_events": r["freeze event"],
        "enh_ratio": r.get("enhanced frame ratio", 0.0),
        "rank_mode": r.get("rank mode", "n/a"),
    }


def main():
    out = {}
    for tname, trel in TRACES:
        tpath = os.path.join(REPO, trel)
        print(f"--- {tname}")
        entry = {"bones": run_one(BONES, tpath, False)}
        for name, cls in CONTROLLERS:
            entry[f"{name}_edf"] = run_one(cls, tpath, False)
            entry[f"{name}_oracle"] = run_one(cls, tpath, True)
        out[tname] = entry

    # 对照表：comb(edf) 相对 BONES；oracle-edf
    print(f"\n{'trace':<12} {'BONES qoe':>9} {'FL':>7} {'INC':>7} {'COMB':>7} "
          f"{'COMB-B':>8} {'oracle-edf(comb)':>16} {'COMB freeze_s':>14} {'enh%':>6}")
    for tname, _ in TRACES:
        e = out[tname]
        c, o = e["comb_edf"], e["comb_oracle"]
        print(f"{tname:<12} {e['bones']['qoe']:>9.2f} "
              f"{e['fl_edf']['qoe']:>7.2f} {e['inc_edf']['qoe']:>7.2f} "
              f"{c['qoe']:>7.2f} {c['qoe']-e['bones']['qoe']:>+8.2f} "
              f"{o['qoe']-c['qoe']:>+16.3f} {c['freeze_s']:>14.2f} "
              f"{c['enh_ratio']*100:>5.1f}")

    # 胜率与中位数（comb edf vs bones）
    deltas = [out[t]["comb_edf"]["qoe"] - out[t]["bones"]["qoe"] for t, _ in TRACES]
    wins = sum(1 for d in deltas if d > 0)
    agg = {
        "comb_edf_vs_bones_qoe_deltas": [round(d, 3) for d in deltas],
        "wins": f"{wins}/{len(deltas)}",
        "median_delta": round(float(sorted(deltas)[len(deltas) // 2]), 4),
        "mean_delta": round(float(sum(deltas) / len(deltas)), 4),
        "mean_oracle_minus_edf_qoe_comb": round(float(sum(
            out[t]["comb_oracle"]["qoe"] - out[t]["comb_edf"]["qoe"]
            for t, _ in TRACES) / len(TRACES)), 4),
    }
    out["_aggregate"] = agg
    print("\naggregate:", json.dumps(agg, ensure_ascii=False))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "six_trace_regression.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
