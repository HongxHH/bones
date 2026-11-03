"""M5 步骤2：EDF 因果版 vs oracle 全知上界 端到端对比。

四合成轨迹 × (BONES 基线, FL/INC/COMB × edf/oracle)，每策略独立构造仿真器
（md_qoe C2 教训：禁共用卡顿向量）。

判决问题：
1. oracle 相对 EDF 的 VMAF/QoE 头间有多大（=帧级增益排序的可利用上限）；
2. EDF 相对 BONES 基线是否稳定占优。
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

TRACE_DIR = os.path.join(REPO, r"data\trace")
ENHANCE_PATH = os.path.join(REPO, r"data\bbb\imdn_bbb_3060ti.json")
LOG_DIR = os.path.join(REPO, r"workspace\sim_logs_v3")
TRACES = ["3g_low", "4g_medium", "wifi_high", "volatile"]
CONTROLLERS = [("fl", BONESFrameLevel), ("inc", BONESIncremental),
               ("comb", BONESCombined)]


def make_args(trace_path, oracle_rank):
    args = argparse.Namespace()
    args.log_path = os.path.join(LOG_DIR, "m5.log")
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
    for trace in TRACES:
        trace_path = os.path.join(TRACE_DIR, f"{trace}.json")
        print(f"--- {trace}")
        entry = {}
        entry["bones"] = run_one(BONES, trace_path, False)
        for name, cls in CONTROLLERS:
            entry[f"{name}_edf"] = run_one(cls, trace_path, False)
            entry[f"{name}_oracle"] = run_one(cls, trace_path, True)
        out[trace] = entry

    # 控制台对照表
    print(f"\n{'trace':<11} {'ctrl':<11} {'VMAF edf/oracle':>18} "
          f"{'QoE edf/oracle':>18}  ΔQoE oracle-edf  ΔQoE edf-BONES")
    for trace in TRACES:
        e = out[trace]
        bq = e["bones"]["qoe"]
        for name, _ in CONTROLLERS:
            ed, orc = e[f"{name}_edf"], e[f"{name}_oracle"]
            print(f"{trace:<11} {name:<11} {ed['vmaf']:>8.2f}/{orc['vmaf']:<8.2f} "
                  f"{ed['qoe']:>8.2f}/{orc['qoe']:<8.2f}  "
                  f"{orc['qoe']-ed['qoe']:>+15.3f} {ed['qoe']-bq:>+13.3f}")

    # 汇总：三控制器 EDF 与 oracle 的平均差距、EDF 对 BONES 平均差距
    agg = {"mean_oracle_minus_edf_qoe": {}, "mean_edf_minus_bones_qoe": {},
           "mean_oracle_minus_edf_vmaf": {}}
    for name, _ in CONTROLLERS:
        dq = [out[t][f"{name}_oracle"]["qoe"] - out[t][f"{name}_edf"]["qoe"]
              for t in TRACES]
        dv = [out[t][f"{name}_oracle"]["vmaf"] - out[t][f"{name}_edf"]["vmaf"]
              for t in TRACES]
        db = [out[t][f"{name}_edf"]["qoe"] - out[t]["bones"]["qoe"]
              for t in TRACES]
        agg["mean_oracle_minus_edf_qoe"][name] = round(sum(dq) / len(dq), 4)
        agg["mean_oracle_minus_edf_vmaf"][name] = round(sum(dv) / len(dv), 4)
        agg["mean_edf_minus_bones_qoe"][name] = round(sum(db) / len(db), 4)
    out["_aggregate"] = agg
    print("\naggregate:", json.dumps(agg, ensure_ascii=False))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "oracle_vs_edf.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
