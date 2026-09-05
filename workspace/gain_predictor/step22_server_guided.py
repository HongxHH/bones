"""
Step 22: 服务端引导神经增强 —— 方向生死证伪实验（纯仿真，秒级）。

核心假设：客户端单边内容感知失效的根因是"预测误差被 BONES 双缓冲耦合放大"。
若把增强层从码率决策中彻底解耦——码率/下载/卡顿完全冻结为原 BONES，增强在一条
独立、非阻塞（永不阻塞播放、永不反压码率）的 GPU 轨道上做 best-effort 执行——
那么"服务端用真值增益引导（可乱序重排增强任务 + 择优方法）"相对"FIFO 段序增强"
是否有稳定正收益？这是服务端协同方向的理论上限，决定该方向生死。

三条线（码率层 history_down 与 freeze 完全相同，差异只在增强 history_enh）：
  bones_orig : 原始 BONES 完整仿真的增强决策（联合 argmin + 双缓冲），现状基线
  fifo       : 独立非阻塞 GPU、严格按段序 FIFO、方法沿用 BONES 选择（解耦但无引导，
               用于校验独立 GPU 模型是否忠实——应与 bones_orig 接近）
  guide_eff  : 独立非阻塞 GPU、服务端按"单位算力真值增益"全局重排+择优（协同上限）
  guide_gain : 同上但按总真值增益排序（敏感性，检验结论是否依赖效率口径）

判定：
  - fifo ≈ bones_orig  => 独立非阻塞 GPU 模型忠实，对照公平；
  - guide 相对 fifo/bones 稳定正向(ΔQoE、胜率、或同质量省 GPU) => 协同方向成立；
  - guide 上限都不正向 => 服务端协同也无空间，方向止损。
"""
import os, sys, glob, json, argparse, time
import numpy as np

ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, "control"))
from control.bones import BONES

MANI = "../data/bbb/bbb_video_info.json"
ENH = "../data/bbb/imdn_bbb_2080ti.json"


class TimedBONES(BONES):
    """标准 BONES，额外记录每段下载完成时刻（global_time）。"""
    def __init__(self, args):
        super().__init__(args)
        self.dl_done = {}

    def warmup_stage(self):
        super().warmup_stage()
        self.dl_done[0] = self.global_time

    def control_download_finish(self):
        # simulator 在调用本钩子前已 idx_down+=1，刚完成段取 idx_enh（父类内才自增）
        seg = self.idx_enh
        super().control_download_finish()
        self.dl_done[seg] = self.global_time


def make_args(trace, sm=1.0):
    return argparse.Namespace(
        log_path=os.path.join(ROOT, "workspace", "gain_predictor", "tmp_log22.txt"),
        manifest_path=MANI, trace_path=trace, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=sm, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False, quick_start=False)


def build_playline(dl_done, seg_time, n):
    """由各段下载完成时刻重建播放时间线，返回 play[i](开始播放时刻) 与逐段 freeze。"""
    play = np.zeros(n)
    freeze = np.zeros(n)
    play[0] = dl_done[0]
    for i in range(1, n):
        scheduled = play[i - 1] + seg_time
        d = dl_done[i]
        if d > scheduled:
            play[i] = d
            freeze[i] = d - scheduled
        else:
            play[i] = scheduled
    return play, freeze


def best_effort_schedule(sim, hd, play, policy, he_bones):
    """
    独立非阻塞串行 GPU 调度，统一 EDF 保序（候选中选段序最小，保证可调度性最优），
    差别只在"该段用哪个增强方法"的策略 policy：
      max    : 选 deadline 内可行的最高级别方法（客户端简单规则，不依赖真值）
      oracle : 选可行中真值增益最高的方法（服务端真值选方法）
      smooth : 纯本地级别平滑，相邻段方法级别差不超过1（不看任何内容真值），压振荡
      bones  : 优先沿用 BONES 原选方法，否则取最高级别
    只在 t+enhance_time<=play[i] 时启动，故永不阻塞播放、永不反压码率。
    """
    n = len(hd)
    nM = sim.vmaf_enh_seg.shape[2]
    et = sim.enhance_time
    q = sim.vmaf_enh_seg
    he = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    done[0] = True  # warmup 段不增强
    t = 0.0
    prev_e = 0

    def feasible_methods(i, tnow):
        d = hd[i]
        out = []
        for e in range(1, nM):
            if not np.isfinite(q[i][d][e]):
                continue
            if tnow + et[d][e] <= play[i] + 1e-6:
                gain = q[i][d][e] - q[i][d][0]
                out.append((e, float(gain), float(et[d][e])))
        return out

    while True:
        # deadline 内已无可行方法的段永久放弃
        for i in range(n):
            if (not done[i]) and sim.dl_done[i] <= t and not feasible_methods(i, t):
                done[i], he[i] = True, 0
        # EDF：此刻可启动的段中选段序最小
        avail = [(i, feasible_methods(i, t)) for i in range(n)
                 if (not done[i]) and sim.dl_done[i] <= t]
        avail = [(i, fm) for i, fm in avail if fm]
        if avail:
            i = min(x[0] for x in avail)
            fe = next(x[1] for x in avail if x[0] == i)
            es = [x[0] for x in fe]
            if policy == "bones":
                cand = [x for x in fe if x[0] == he_bones[i]]
                chosen = cand[0] if cand else max(fe, key=lambda x: x[0])
            elif policy == "oracle":
                chosen = max(fe, key=lambda x: x[1])     # 真值增益最高
            elif policy == "smooth":
                # 纯本地：限制相邻级别跳变，优先 [prev_e-1, prev_e+1] 内最高，爬坡限速
                band = [x for x in fe if abs(x[0] - prev_e) <= 1]
                chosen = max(band or fe, key=lambda x: x[0])
            else:  # max：最高级别
                chosen = max(fe, key=lambda x: x[0])
            e, _, c = chosen
            he[i], done[i] = e, True
            prev_e = e
            t += c
            continue
        future = [i for i in range(n) if (not done[i]) and sim.dl_done[i] > t]
        if future:
            t = min(sim.dl_done[i] for i in future)
            continue
        break

    gpu = sum(float(et[hd[i]][he[i]]) for i in range(n))
    return he, gpu


def score(sim, hd, he, freeze_ms):
    """用真值 vmaf_enh_seg 计分；freeze 由码率层唯一确定，三线相同。"""
    n = len(hd)
    q = np.array([sim.vmaf_enh_seg[i][hd[i]][he[i]] for i in range(n)])
    avg = float(q.mean())
    osc = float(np.sum(np.abs(q[1:] - q[:-1])) / (n - 1))
    freeze_avg = freeze_ms / n
    qoe = avg - osc - freeze_avg / 10
    return {"qoe": qoe, "vmaf": avg, "osc": osc,
            "enh_ratio": float(np.mean(he != 0)),
            "gpu_ms": sum(float(sim.enhance_time[hd[i]][he[i]]) for i in range(n))}


def run_one(trace, sm=1.0):
    sim = TimedBONES(make_args(trace, sm))
    sim.run()
    rep = sim.report()
    hd = np.array(rep["download history"])
    he_bones = np.array(rep["enhance history"])
    n = len(hd)
    play, freeze_seg = build_playline(sim.dl_done, sim.seg_time, n)
    freeze_recon = float(freeze_seg.sum())
    freeze_sim = float(sim.buffer_download.idle_time)

    he_max, _ = best_effort_schedule(sim, hd, play, "max", he_bones)
    he_oracle, _ = best_effort_schedule(sim, hd, play, "oracle", he_bones)
    he_smooth, _ = best_effort_schedule(sim, hd, play, "smooth", he_bones)
    he_bp, _ = best_effort_schedule(sim, hd, play, "bones", he_bones)

    out = {
        "bones_orig": {"qoe": float(rep["qoe"]), "vmaf": float(rep["average quality vmaf"]),
                       "osc": float(rep["average oscillation vmaf"]),
                       "enh_ratio": float(np.mean(he_bones != 0)),
                       "gpu_ms": sum(float(sim.enhance_time[hd[i]][he_bones[i]]) for i in range(n))},
        "edf_bones": score(sim, hd, he_bp, freeze_sim),
        "edf_max": score(sim, hd, he_max, freeze_sim),
        "edf_smooth": score(sim, hd, he_smooth, freeze_sim),
        "edf_oracle": score(sim, hd, he_oracle, freeze_sim),
        "_freeze_recon_err_ms": abs(freeze_recon - freeze_sim),
    }
    return out


def collect_traces():
    traces = glob.glob("../data/trace/3Glogs/*.json") + glob.glob("../data/trace/4Glogs/*.json")
    for p in glob.glob("../data/trace/*.json"):
        if os.path.basename(p) != "dummy.json":
            traces.append(p)
    return sorted(traces)


def main():
    traces = collect_traces()
    SMS = [1.0, 0.6, 0.4, 0.25]
    LINES = ["bones_orig", "edf_bones", "edf_max", "edf_smooth", "edf_oracle"]
    print(f"轨迹总数: {len(traces)}, GPU倍率: {SMS}")
    result = {}
    t0 = time.time()
    for sm in SMS:
        per = []
        recon = []
        for tr in traces:
            r = run_one(tr, sm)
            recon.append(r.pop("_freeze_recon_err_ms"))
            r["trace"] = os.path.relpath(tr, "..").replace("\\", "/")
            per.append(r)
        print(f"sm={sm} 完成, freeze重建最大误差{max(recon):.2f}ms, 累计{time.time()-t0:.0f}s")

        def agg(line, metric):
            return float(np.mean([r[line][metric] for r in per]))
        summary = {l: {k: agg(l, k) for k in ["qoe", "vmaf", "osc", "enh_ratio", "gpu_ms"]} for l in LINES}

        def pair(la, lb, metric):
            d = [r[la][metric] - r[lb][metric] for r in per]
            return float(np.mean(d)), float(np.median(d)), float(np.mean(np.array(d) > 1e-6))
        summary["pairs"] = {
            f"{a}_minus_{b}": {k: dict(zip(["mean", "median", "win"], pair(a, b, k)))
                               for k in ["qoe", "vmaf", "osc", "enh_ratio", "gpu_ms"]}
            for a, b in [("edf_bones", "bones_orig"), ("edf_max", "bones_orig"),
                         ("edf_smooth", "bones_orig"), ("edf_smooth", "edf_max"),
                         ("edf_oracle", "edf_max"), ("edf_oracle", "bones_orig")]}

        def net(t):
            return "3G" if "3Glogs" in t else "4G" if "4Glogs" in t else "root"
        for nw in ["3G", "4G", "root"]:
            idx = [i for i, r in enumerate(per) if net(r["trace"]) == nw]
            summary[f"net_{nw}"] = {"n": len(idx),
                **{l: float(np.mean([per[i][l]["qoe"] for i in idx])) for l in LINES}}
        result[f"sm_{sm}"] = {"summary": summary, "per_trace": per}

    outp = os.path.join(ROOT, "workspace", "gain_predictor", "step22_server_guided.json")
    json.dump(result, open(outp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    for sm in SMS:
        s = result[f"sm_{sm}"]["summary"]
        print(f"\n########## GPU倍率 sm={sm} ##########")
        print(f"{'指标':<10}" + "".join(f"{l:>13}" for l in LINES))
        for k in ["qoe", "vmaf", "osc", "enh_ratio", "gpu_ms"]:
            print(f"{k:<10}" + "".join(f"{s[l][k]:>13.3f}" for l in LINES))
        for name, dd in s["pairs"].items():
            q, g = dd["qoe"], dd["gpu_ms"]
            print(f"  {name:<24} ΔQoE={q['mean']:+.3f}/med{q['median']:+.3f}/win{q['win']*100:.1f}%  ΔGPU={g['mean']:+.0f}ms")
        for nw in ["3G", "4G", "root"]:
            x = s[f"net_{nw}"]
            print(f"  {nw}(n={x['n']}): " + "  ".join(f"{l}={x[l]:.2f}" for l in LINES))
    print(f"\n已保存 {outp}, 总耗时{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
