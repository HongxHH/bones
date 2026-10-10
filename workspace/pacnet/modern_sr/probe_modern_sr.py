# -*- coding: utf-8 -*-
"""
现代分辨率超分生死探针：1080p -> 4K (x2)。

设定（与 426 宽 x4 证伪链不同的新区间）：
  参考：原生 3840x2160@60fps BBB（bbb_2160p_60fps.mp4）。
  退化：4K bicubic 下采样到 1920x1080，libx264 CRF 18/22/26/30/34（gen_1080p_ladder.ps1）。
  复原：bicubic x2 / IMDN low(nf=6,m=3) / medium(nf=12,m=5) / high(nf=32,m=6)，全部 DIV2K 泛化权重。
  打分：连续片段（4x20s，自然 60fps 时间序，禁止抽样帧打包），4K VMAF 模型逐帧
        （libvmaf n_subsample=2：每 2 帧打 1 帧，json frameNum 记原始偶数帧号，
        1200 帧输入→600 个分）；LPIPS 全量每 200 帧取 1 帧（6 帧/片段）、
        冒烟每 100 帧（无运动项，抽样不影响均值方向）；单帧延迟在 1080p 输入上实测。
输出 modern_sr_probe.json。冒烟：set MODERN_SR_SMOKE=1。
"""
import os, json, subprocess, time, shutil
import numpy as np
import torch, cv2
import sys

sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
from istream_player.modules.enhancer.imdn_model import IMDN_RTC, IMDN

DEVICE = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
LADDER = os.path.join(HERE, "ladder")
TMP = os.path.join(HERE, "tmp")
os.makedirs(TMP, exist_ok=True)
SRC = r"E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4"
VMAF_4K = os.path.join(HERE, "models", "vmaf_4k_v0.6.1.json")
W2, H2 = 3840, 2160      # 4K
W1, H1 = 1920, 1080      # 1080p
FPS = 60
CLIP_SEC = 20
N = FPS * CLIP_SEC       # 1200 帧/片段
STARTS = [60, 220, 380, 540]
CRFS = [18, 22, 26, 30, 34]
# 权重在 player 目录，统一从那里读
PLAYER_W = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight"
LEVELS = [
    ("low",  IMDN_RTC(upscale=2, num_modules=3, nf=6), "IMDN_low_DIV2K_x2.pth"),
    ("med",  IMDN_RTC(upscale=2),                      "IMDN_medium_DIV2K_x2.pth"),
    ("high", IMDN(upscale=2, nf=32),                   "IMDN_high_DIV2K_x2.pth"),
]
OUT_JSON = os.path.join(HERE, "modern_sr_probe.json")
SMOKE = os.environ.get("MODERN_SR_SMOKE") == "1"


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def cut_ref_clips():
    """精确 seek 切 4K 参考片段，x264 无损、GOP=240，供输入 seek 帧对齐。"""
    for i, st in enumerate(STARTS):
        out = os.path.join(TMP, f"ref_c{i}.mkv")
        if os.path.exists(out):
            continue
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", str(st), "-i", SRC,
               "-t", str(CLIP_SEC), "-c:v", "libx264", "-crf", "0",
               "-preset", "veryfast", "-g", "240", "-pix_fmt", "yuv420p",
               "-an", out]
        r = run(cmd, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"ref cut {i} failed: {r.stderr[-500:]}")
        print(f"ref clip {i} cut")


def _read_exact(stream, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    if len(buf) < n:
        raise RuntimeError(f"short rawvideo read: {len(buf)}/{n}")
    return bytes(buf)


def stream_pair(ci, crf, nframes):
    """逐帧产出 (ref 4K bgr, lr 1080p bgr)，双解码管道流式读取（内存恒定）。"""
    ref_path = os.path.join(TMP, f"ref_c{ci}.mkv")
    lr_path = os.path.join(LADDER, f"lr1080_crf{crf}.mp4")
    c_ref = ["ffmpeg", "-v", "error", "-i", ref_path,
             "-frames:v", str(nframes), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    c_lr = ["ffmpeg", "-v", "error", "-ss", str(STARTS[ci]), "-i", lr_path,
            "-frames:v", str(nframes), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p_ref = subprocess.Popen(c_ref, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p_lr = subprocess.Popen(c_lr, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rb, lb = W2 * H2 * 3, W1 * H1 * 3
    try:
        for _ in range(nframes):
            fr = np.frombuffer(_read_exact(p_ref.stdout, rb), np.uint8).reshape(H2, W2, 3)
            fl = np.frombuffer(_read_exact(p_lr.stdout, lb), np.uint8).reshape(H1, W1, 3)
            yield fr, fl
    finally:
        p_ref.wait(); p_lr.wait()


def build_models():
    models = {}
    for name, m, wf in LEVELS:
        m.load_state_dict(torch.load(os.path.join(PLAYER_W, wf), map_location="cpu"))
        models[name] = m.to(DEVICE).eval()
    return models


def infer(model, bgr):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        o = model(t)
    # uint8 化在 GPU 上完成，避免 4K float 张量回 CPU 后的非连续 permute 慢路径
    o = (o[0] * 255.0).clamp(0, 255).round().to(torch.uint8)
    o = o.permute(1, 2, 0).contiguous().cpu().numpy()
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)


def vmaf_pipe(clip_idx, crf, n_levels=3):
    """竖排 [ref,bic,low,med,high] 一次喂入，4 路 libvmaf（4K 模型）。"""
    names = ["bic"] + [lv[0] for lv in LEVELS[:n_levels]]
    rows = 1 + len(names)
    logs = {nm: f"vmaf_c{clip_idx}_crf{crf}_{nm}.json" for nm in names}
    for p in logs.values():
        ap = os.path.join(TMP, p)
        if os.path.exists(ap):
            os.remove(ap)
    # 模型文件放 TMP，用相对路径（Windows libvmaf 不认盘符冒号）
    shutil.copy(VMAF_4K, os.path.join(TMP, "vmaf_4k_v0.6.1.json"))
    fc = "".join(f"[0:v]crop={W2}:{H2}:0:{k*H2}[r{k}];" for k in range(rows))
    fc += f"[r0]split={len(names)}" + "".join(f"[s{k}]" for k in range(len(names))) + ";"
    maps = []
    for k, nm in enumerate(names):
        fc += (f"[r{k+1}][s{k}]libvmaf=n_threads=2:n_subsample=2:"
               f"model=path=vmaf_4k_v0.6.1.json:"
               f"log_path={logs[nm]}:log_fmt=json[m{k}];")
        maps += ["-map", f"[m{k}]"]
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pixel_format", "bgr24",
           "-video_size", f"{W2}x{rows*H2}", "-framerate", str(FPS), "-i", "-",
           "-filter_complex", fc.rstrip(";")] + maps + ["-f", "null", "-"]
    logf = open(os.path.join(TMP, f"ffmpeg_c{clip_idx}_crf{crf}.log"), "wb")
    p = subprocess.Popen(cmd, cwd=TMP, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=logf)
    return p, logs, names


def read_vmaf(path):
    log = json.load(open(path, "r", encoding="utf-8"))
    return [float(fr["metrics"]["vmaf"]) for fr in log["frames"]]


def latency_bench(models):
    """单帧延迟：GPU 前向（fp32/fp16）与端到端（含回传+后处理，fp32）。"""
    import copy
    out = {}
    dummy = np.random.randint(0, 255, (H1, W1, 3), dtype=np.uint8)

    def gpu_forward(model, bgr, half=False):
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
        if half:
            t = t.half()
        with torch.no_grad():
            model(t)
        torch.cuda.synchronize()

    def timeit(fn, n=40, warm=8):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(n):
            t0 = time.time(); fn(); ts.append((time.time() - t0) * 1000)
        return round(float(np.mean(ts)), 2), round(float(np.median(ts)), 2)

    for name, m32 in models.items():
        g32 = timeit(lambda: gpu_forward(m32, dummy))
        e2e = timeit(lambda: infer(m32, dummy))
        rec = {"gpu_fp32_mean_ms": g32[0], "gpu_fp32_p50_ms": g32[1],
               "e2e_fp32_mean_ms": e2e[0], "e2e_fp32_p50_ms": e2e[1]}
        m16 = copy.deepcopy(m32).half()
        g16 = timeit(lambda: gpu_forward(m16, dummy, True))
        img = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img).permute(2, 0, 1).half().unsqueeze(0).to(DEVICE) / 255.0
        with torch.no_grad():
            finite = bool(torch.isfinite(m16(t)).all().item())
        rec.update({"gpu_fp16_mean_ms": g16[0], "gpu_fp16_p50_ms": g16[1],
                    "fp16_finite": finite})
        del m16
        torch.cuda.empty_cache()
        out[name] = rec
        print(f"latency {name}: {rec}", flush=True)
    return out


def main():
    cut_ref_clips()
    models = build_models()

    # 延迟只测一次，独立落盘（多 worker 并行时复用）
    lat_json = os.path.join(HERE, "latency_modern_sr.json")
    if os.path.exists(lat_json):
        latency = json.load(open(lat_json, "r", encoding="utf-8"))
    else:
        latency = latency_bench(models)
        json.dump(latency, open(lat_json, "w", encoding="utf-8"), indent=1)

    import lpips
    lp = lpips.LPIPS(net="vgg").to(DEVICE).eval()

    if SMOKE:
        clips, crfs, nframes = [0], [26], 120
        lpips_step = 100
    else:
        clips = [int(x) for x in os.environ.get("MODERN_SR_CLIPS", "0,1,2,3").split(",")]
        crfs = [int(x) for x in os.environ.get("MODERN_SR_CRFS", "18,22,26,30,34").split(",")]
        nframes, lpips_step = N, 200
    tag = "smoke" if SMOKE else "c" + "".join(str(i) for i in clips)
    out_json = os.path.join(HERE, f"modern_sr_probe_{tag}.json") if not SMOKE else OUT_JSON
    results = {"config": {"src": SRC, "clips": [STARTS[i] for i in clips],
                          "clip_sec": CLIP_SEC, "fps": FPS, "crfs": crfs,
                          "weights": "DIV2K zero-shot x2",
                          "vmaf_model": "vmaf_4k_v0.6.1",
                          "vmaf_n_subsample": 2},
               "latency_1080p_in_ms": latency, "clips": {}}

    for ci in clips:
        results["clips"][str(ci)] = {}
        for crf in crfs:
            lr_path = os.path.join(LADDER, f"lr1080_crf{crf}.mp4")
            # 比特率（整片口径）
            pr = run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "format=bit_rate", "-of",
                      "csv=p=0", lr_path])
            kbps = round(int(pr.stdout.strip()) / 1000)

            proc, logs, names = vmaf_pipe(ci, crf)
            lpips_acc = {nm: [] for nm in names}
            t0 = time.time()
            for fi, (ref_f, lr_f) in enumerate(stream_pair(ci, crf, nframes)):
                bic = cv2.resize(lr_f, (W2, H2), interpolation=cv2.INTER_CUBIC)
                outs = [bic]
                for name, _m, _ in LEVELS:
                    outs.append(infer(models[name], lr_f))
                stack = np.concatenate([ref_f] + outs, axis=0)
                proc.stdin.write(np.ascontiguousarray(stack).tobytes())
                if fi % lpips_step == 0:
                    with torch.no_grad():
                        rt = torch.from_numpy(
                            cv2.cvtColor(ref_f, cv2.COLOR_BGR2RGB)
                        ).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 127.5 - 1
                        for nm, im in zip(names, outs):
                            dt = torch.from_numpy(
                                cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            ).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 127.5 - 1
                            lpips_acc[nm].append(float(lp(dt, rt).item()))
            proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg vmaf rc={rc}, see tmp log c{ci} crf{crf}")
            row = {"kbps": kbps}
            for nm in names:
                sc = read_vmaf(os.path.join(TMP, logs[nm]))
                row[f"vmaf_{nm}"] = round(float(np.mean(sc)), 3)
                row[f"lpips_{nm}"] = round(float(np.mean(lpips_acc[nm])), 4)
            results["clips"][str(ci)][str(crf)] = row
            print(f"clip{ci} crf{crf} {kbps}kbps "
                  + " ".join(f"{nm}:V={row[f'vmaf_{nm}']:.2f}/L={row[f'lpips_{nm}']:.3f}"
                             for nm in names)
                  + f" ({time.time()-t0:.0f}s)")
            json.dump(results, open(out_json, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)
    print("ALL DONE ->", out_json)


if __name__ == "__main__":
    main()
