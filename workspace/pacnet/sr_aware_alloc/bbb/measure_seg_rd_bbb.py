# -*- coding: utf-8 -*-
"""
SR 感知率失真分配跨内容复核 —— BBB 逐段 post-SR R-D 测量。
与 TOS 版（../measure_seg_rd.py）同管线，差异仅素材参数：
BBB 1920x1080@30fps，4s 段 = 120 帧，426 宽 LR（426x240），x4 后 bicubic 对齐
1920x1080；rawvideo 堆叠管道双 libvmaf 连续时间序打分；ffprobe 逐帧 pkt_size 按段聚合。
输出 seg_rd_bbb.json，结构与 TOS seg_rd.json 相同，供 alloc_compare --rd 复用。
"""
import os, json, subprocess
import numpy as np
import torch, cv2
import sys

sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
from istream_player.modules.enhancer.imdn_model import IMDN_RTC

DEVICE = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = os.path.join(HERE, "ladder")
TMP = os.path.join(HERE, "tmp")
os.makedirs(TMP, exist_ok=True)
WEIGHT = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight\IMDN_medium_DIV2K_x4.pth"
REF_VIDEO = r"E:\dev\Code\Python\bones\data\video_mp4box\output_1080p.mp4"
W, H = 1920, 1080
FPS, SEG = 30, 120
CRFS = [22, 26, 30, 34, 38, 42]
OUT_JSON = os.path.join(HERE, "seg_rd_bbb.json")


def build_l2():
    m = IMDN_RTC(upscale=4)
    m.load_state_dict(torch.load(WEIGHT, map_location="cpu"))
    return m.to(DEVICE).eval()


def infer_l2(model, bgr):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        o = model(t)
        o = torch.nn.functional.interpolate(o, size=(H, W), mode="bicubic", align_corners=False)
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    o = (o * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)


def frame_bits_by_segment(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_frames", "-show_entries", "frame=pts_time,pkt_size",
           "-of", "csv=p=0", path]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    seg_bits = {}
    for line in res.stdout.strip().split("\n"):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            pts, size = float(parts[0]), int(parts[1])
        except ValueError:
            continue
        seg = int(round(pts * FPS)) // SEG
        seg_bits[seg] = seg_bits.get(seg, 0) + size * 8
    return seg_bits


def vmaf_both_via_pipe(bic_tag, sr_tag):
    logs = {"bic": f"vmaf_{bic_tag}.json", "sr": f"vmaf_{sr_tag}.json"}
    for p in logs.values():
        ap = os.path.join(TMP, p)
        if os.path.exists(ap):
            os.remove(ap)
    fc = (f"[0:v]crop={W}:{H}:0:0[d1];[0:v]crop={W}:{H}:0:{H}[d2];"
          "[1:v]split[r1][r2];"
          f"[d1][r1]libvmaf=log_path={logs['bic']}:log_fmt=json[a];"
          f"[d2][r2]libvmaf=log_path={logs['sr']}:log_fmt=json[b]")
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pixel_format", "bgr24",
           "-video_size", f"{W}x{2*H}", "-framerate", str(FPS), "-i", "-",
           "-i", REF_VIDEO, "-filter_complex", fc,
           "-map", "[a]", "-map", "[b]", "-f", "null", "-"]
    proc = subprocess.Popen(cmd, cwd=TMP, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return proc, logs


def read_vmaf_json(path):
    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)
    return [float(fr["metrics"]["vmaf"]) for fr in log["frames"]]


def seg_means(frame_scores, nseg):
    return [float(np.mean(frame_scores[s * SEG:(s + 1) * SEG])) for s in range(nseg)]


def main():
    results = {}
    if os.path.exists(OUT_JSON):
        results = json.load(open(OUT_JSON, "r", encoding="utf-8"))

    model = build_l2()
    cap0 = cv2.VideoCapture(REF_VIDEO)
    total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT)); cap0.release()
    nseg = total // SEG
    print(f"BBB source total frames={total}, full segments={nseg}", flush=True)

    for c in CRFS:
        key = f"crf{c}"
        if key in results and len(results[key].get("segments", [])) == nseg:
            print(f"SKIP {key}"); continue
        lr_path = os.path.join(LADDER, f"lr426_crf{c}.mp4")
        if not os.path.exists(lr_path):
            print(f"MISSING {lr_path}"); continue

        seg_bits = frame_bits_by_segment(lr_path)
        proc, logs = vmaf_both_via_pipe(f"bic_crf{c}", f"sr_crf{c}")
        cap = cv2.VideoCapture(lr_path)
        n = 0
        while True:
            ret, f = cap.read()
            if not ret:
                break
            bic = cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)
            sr = infer_l2(model, f)
            proc.stdin.write(np.vstack([bic, sr]).tobytes())
            n += 1
            if n % 2000 == 0:
                print(f"  crf{c} frames {n}", flush=True)
        cap.release()
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "ignore")
        rc = proc.wait()
        if rc != 0:
            print(err[-3000:]); raise RuntimeError(f"ffmpeg rc={rc} crf{c}")
        assert n == total, (n, total)

        bic_frames = read_vmaf_json(os.path.join(TMP, logs["bic"]))
        sr_frames = read_vmaf_json(os.path.join(TMP, logs["sr"]))
        assert len(bic_frames) == total and len(sr_frames) == total, \
            (len(bic_frames), len(sr_frames), total)
        bic_seg, sr_seg = seg_means(bic_frames, nseg), seg_means(sr_frames, nseg)

        segs = [{"seg": s, "bits": int(seg_bits.get(s, 0)),
                 "bic_vmaf": round(bic_seg[s], 4), "sr_vmaf": round(sr_seg[s], 4)}
                for s in range(nseg)]
        results[key] = {"crf": c, "nseg": nseg, "segments": segs,
                        "mean_bits": float(np.mean([x["bits"] for x in segs])),
                        "mean_bic_vmaf": float(np.mean(bic_seg)),
                        "mean_sr_vmaf": float(np.mean(sr_seg))}
        json.dump(results, open(OUT_JSON, "w", encoding="utf-8"), indent=1)
        for p in logs.values():
            try: os.remove(os.path.join(TMP, p))
            except OSError: pass
        print(f"  crf{c} DONE bits/seg={results[key]['mean_bits']:.0f} "
              f"bic={results[key]['mean_bic_vmaf']:.2f} sr={results[key]['mean_sr_vmaf']:.2f}",
              flush=True)

    print("ALL CRF DONE BBB", flush=True)


if __name__ == "__main__":
    main()
