# -*- coding: utf-8 -*-
"""
高频先验旁路信息论上界探针（纯 ffmpeg 版）。
side = grainextract(ref, bic) = ref - bic + 128，libx264 CRF 阶梯编码；
F    = grainmerge(bic, side)  = bic + side - 128；
测 F vs ref 的 4K VMAF。判决：旁路 7~15kbps（主码流 726kbps 的 1~2%）时 ΔVMAF>=+2。
"""
import os, json, subprocess, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODERN = os.path.join(os.path.dirname(HERE), "modern_sr")
TMP = os.path.join(HERE, "tmp")
os.makedirs(TMP, exist_ok=True)
REFDIR = os.path.join(MODERN, "tmp")
VMAF_MODEL_SRC = os.path.join(MODERN, "models", "vmaf_4k_v0.6.1.json")
W, H, FPS = 3840, 2160, 60
N = FPS * 20
CRFS = [38, 44, 50, 56, 62]
OUT_JSON = os.path.join(HERE, "side_channel_probe.json")
SMOKE = os.environ.get("SC_SMOKE") == "1"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd[:6]}\n{r.stderr[-400:]}")
    return r


def kbps_of(path):
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=bit_rate", "-of", "csv=p=0", path])
    return int(r.stdout.strip()) // 1000


def encode_side(ci, crf, nframes):
    out = os.path.join(TMP, f"side_c{ci}_crf{crf}.mp4")
    if os.path.exists(out):
        return kbps_of(out), out
    ref = os.path.join(REFDIR, f"ref_c{ci}.mkv")
    bic = os.path.join(TMP, f"bic_c{ci}.mkv")
    run(["ffmpeg", "-y", "-v", "error", "-i", ref, "-i", bic,
         "-filter_complex", "[0:v][1:v]blend=all_mode=grainextract",
         "-frames:v", str(nframes),
         "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
         "-pix_fmt", "yuv420p", "-an", out])
    return kbps_of(out), out


def rebuild_F(ci, side_path, nframes, tag):
    """F = grainmerge(bic, side) = bic + side - 128，crf0 落盘。"""
    out = os.path.join(TMP, f"F_c{ci}_{tag}.mkv")
    if os.path.exists(out):
        return out
    bic = os.path.join(TMP, f"bic_c{ci}.mkv")
    run(["ffmpeg", "-y", "-v", "error", "-i", bic, "-i", side_path,
         "-filter_complex", "[0:v][1:v]blend=all_mode=grainmerge",
         "-frames:v", str(nframes),
         "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", "-an", out])
    return out


def vmaf_vs_ref(ci, f_path, nframes, tag):
    log = os.path.join(TMP, f"vmaf_c{ci}_{tag}.json")
    if os.path.exists(log):
        os.remove(log)
    shutil.copy(VMAF_MODEL_SRC, os.path.join(TMP, "vmaf_4k_v0.6.1.json"))
    ref = os.path.join(REFDIR, f"ref_c{ci}.mkv")
    run(["ffmpeg", "-y", "-v", "error", "-i", ref, "-i", f_path,
         "-lavfi", f"libvmaf=n_threads=2:n_subsample=2:model=path=vmaf_4k_v0.6.1.json:log_path=vmaf_c{ci}_{tag}.json:log_fmt=json",
         "-frames:v", str(nframes), "-f", "null", "-"], cwd=TMP, timeout=3600)
    d = json.load(open(log, encoding="utf-8"))
    return float(np.mean([float(fr["metrics"]["vmaf"]) for fr in d["frames"]]))


def main():
    clips = [0] if SMOKE else [0, 1, 2, 3]
    crfs = [38, 50, 62] if SMOKE else CRFS
    nframes = 200 if SMOKE else N
    results = {"config": {"base_kbps": 726, "side": "grainextract D+128 libx264 veryfast",
                          "vmaf": "vmaf_4k_v0.6.1 n_subsample=2"}, "clips": {}}
    for ci in clips:
        row = {}
        # bic 基线：F=bic 本身
        bic_v = vmaf_vs_ref(ci, os.path.join(TMP, f"bic_c{ci}.mkv"), nframes, "bic")
        row["bic_vmaf"] = round(bic_v, 3)
        print(f"c{ci} bic_vmaf={bic_v:.2f}", flush=True)
        for q in crfs:
            kb, sp = encode_side(ci, q, nframes)
            F = rebuild_F(ci, sp, nframes, f"q{q}")
            v = vmaf_vs_ref(ci, F, nframes, f"q{q}")
            row[f"crf{q}"] = {"side_kbps": kb, "side_pct": round(kb/726*100, 2),
                              "vmaf": round(v, 3), "delta": round(v - bic_v, 3)}
            print(f"  crf{q}: side={kb}kbps ({kb/726*100:.1f}%) F_vmaf={v:.2f} Δ={v-bic_v:+.2f}", flush=True)
        results["clips"][str(ci)] = row
        json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ALL DONE ->", OUT_JSON)


if __name__ == "__main__":
    main()
