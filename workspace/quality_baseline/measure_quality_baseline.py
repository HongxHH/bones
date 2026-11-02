# -*- coding: utf-8 -*-
"""
统一帧级质量基线测量（BBB，参考 1080p）。

按 docs/frame_quality_measurement.md 金标准管道：
  - LR 输入由 ffmpeg 解码为连续 rawvideo rgb24（按显示时间序，不抽帧）；
  - bicubic 臂：纯 ffmpeg，YUV 域 scale=flags=bicubic 后直接对参考跑 libvmaf；
  - sr_low/medium/high 臂：BBB 训练权重，模型 forward 后 bicubic 对齐 1920x1080，
    每臂一个 ffmpeg（rawvideo stdin + 参考第二输入 + 单 libvmaf，n_threads=1，
    log_fmt=json 出逐帧分）；各臂由独立 feeder 线程喂帧，互不阻塞。
  - 帧数锁：两端 trim=end_frame=N（勿依赖输出 -frames:v；ffmpeg-6.1.1 双输入会多打分）。
不使用多臂堆叠单图（4 臂堆叠+split+多 map 的 filtergraph 在本机调度死锁）。
输出 data/quality/bbb_{240,360,480,720}.json。

用法：
  python measure_quality_baseline.py --smoke        # 240p 前120帧，验证+估时
  python measure_quality_baseline.py                 # 全量，断点续跑
  python measure_quality_baseline.py --only 240      # 指定输入档
"""
import os, sys, json, subprocess, threading, queue, hashlib, argparse
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import numpy as np
import torch

PLAYER = r"E:\dev\Code\Python\bones\player"
sys.path.insert(0, PLAYER)
from istream_player.modules.enhancer.imdn_model import IMDN, IMDN_RTC

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"E:\dev\Code\Python\bones"
TMP = os.path.join(HERE, "tmp")
os.makedirs(TMP, exist_ok=True)
OUT_DIR = os.path.join(REPO, "data", "quality")
os.makedirs(OUT_DIR, exist_ok=True)

REF_VIDEO = os.path.join(REPO, r"data\video_mp4box\output_1080p.mp4")
W, H = 1920, 1080
FPS = 30
FRAMES_PER_SEG = 120
TOTAL_FRAMES = 19039
FULL_SEGMENTS = 158
DEVICE = "cuda:0"
# ffmpeg 可执行文件：默认 PATH；可用 --ffmpeg 或环境变量 BONES_FFMPEG 指定
# （PATH 中 ffmpeg-6.1.1 残缺构建的 libvmaf 异常缓慢，必须显式指定正常构建）
FFMPEG = os.environ.get("BONES_FFMPEG", "ffmpeg")
WEIGHT_DIR = os.path.join(PLAYER, r"istream_player\modules\enhancer\imdn_weight")

# 输入档 -> (LR 文件, SR scale, (lw,lh))
SR_GRID = {
    240: (os.path.join(REPO, r"data\video_mp4box\output_240p.mp4"), 4, (426, 240)),
    360: (os.path.join(REPO, r"data\video_mp4box\output_360p.mp4"), 3, (640, 360)),
    480: (os.path.join(REPO, r"data\video_mp4box\output_480p.mp4"), 2, (854, 480)),
}
BIC_ONLY = {
    720: (os.path.join(REPO, r"data\video_mp4box\output_720p.mp4"), (1280, 720)),
}
SR_ARMS = ["sr_low", "sr_medium", "sr_high"]

METHOD_META = {
    "bicubic": ("ffmpeg_scale_bicubic_yuv", None),
    "sr_low": ("IMDN_RTC num_modules=3 nf=6", "low"),
    "sr_medium": ("IMDN_RTC default nf=12 num_modules=5", "medium"),
    "sr_high": ("IMDN nf=32", "high"),
}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_path(level, scale):
    return os.path.join(WEIGHT_DIR, f"IMDN_{level}_bbb_x{scale}.pth")


def build_models(scale):
    models = {}
    specs = {"sr_low": IMDN_RTC(upscale=scale, num_modules=3, nf=6),
             "sr_medium": IMDN_RTC(upscale=scale),
             "sr_high": IMDN(upscale=scale, nf=32)}
    for name, model in specs.items():
        level = METHOD_META[name][1]
        model.load_state_dict(torch.load(weight_path(level, scale), map_location="cpu"))
        models[name] = model.to(DEVICE).eval()
    return models


def infer_rgb(model, rgb):
    t = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        o = model(t)
        o = torch.nn.functional.interpolate(o, size=(H, W), mode="bicubic",
                                            align_corners=False)
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (o * 255).clip(0, 255).astype(np.uint8)


def err_file(name):
    return open(os.path.join(TMP, f"err_{name}.log"), "wb")


def start_lr_producer(lr_path, n_frames):
    # -frames:v 必须作为输出选项，否则冒烟时生产者持续解码全片
    cmd = [FFMPEG, "-v", "error", "-i", lr_path,
           "-vf", "format=rgb24", "-frames:v", str(n_frames),
           "-f", "rawvideo", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file("lr"))


def start_bic_consumer(lr_path, tag, n_frames):
    log_name = f"vmaf_{tag}_bicubic.json"
    p = os.path.join(TMP, log_name)
    if os.path.exists(p):
        os.remove(p)
    # 双输入 libvmaf：本机 ffmpeg-6.1.1 上输出侧 -frames:v 会多打分（冒烟 137≠120）。
    # 两端 trim=end_frame 才能锁死帧数；setpts 避免 trim 后时间戳空洞。
    lavfi = (
        f"[0:v]trim=end_frame={n_frames},setpts=PTS-STARTPTS,"
        f"scale={W}:{H}:flags=bicubic,format=yuv420p[d];"
        f"[1:v]trim=end_frame={n_frames},setpts=PTS-STARTPTS[r];"
        f"[d][r]libvmaf=n_threads=1:log_path={log_name}:log_fmt=json"
    )
    cmd = [FFMPEG, "-v", "error", "-i", lr_path, "-i", REF_VIDEO,
           "-lavfi", lavfi, "-f", "null", "-"]
    return subprocess.Popen(cmd, cwd=TMP, stdout=subprocess.DEVNULL,
                            stderr=err_file(f"{tag}_bicubic")), log_name


def start_sr_consumer(tag, arm, n_frames):
    log_name = f"vmaf_{tag}_{arm}.json"
    p = os.path.join(TMP, log_name)
    if os.path.exists(p):
        os.remove(p)
    lavfi = (
        f"[0:v]format=yuv420p,trim=end_frame={n_frames},setpts=PTS-STARTPTS[d];"
        f"[1:v]trim=end_frame={n_frames},setpts=PTS-STARTPTS[r];"
        f"[d][r]libvmaf=n_threads=1:log_path={log_name}:log_fmt=json"
    )
    cmd = [FFMPEG, "-v", "error",
           "-f", "rawvideo", "-pixel_format", "rgb24",
           "-video_size", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
           "-i", REF_VIDEO, "-lavfi", lavfi, "-f", "null", "-"]
    proc = subprocess.Popen(cmd, cwd=TMP, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=err_file(f"{tag}_{arm}"))
    q = queue.Queue(maxsize=4)

    def feeder():
        while True:
            item = q.get()
            if item is None:
                proc.stdin.close()
                return
            proc.stdin.write(item)

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    return proc, q, th, log_name


def read_vmaf_json(path):
    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)
    return [float(fr["metrics"]["vmaf"]) for fr in log["frames"]]


def check_rc(proc, name):
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg rc={rc} for {name}; see tmp/err_{name}.log")


def measure_sr_input(res, smoke=False):
    lr_path, scale, (lw, lh) = SR_GRID[res]
    tag = f"bbb{res}" + ("_smoke" if smoke else "")
    n_target = FRAMES_PER_SEG if smoke else TOTAL_FRAMES

    models = build_models(scale)
    p_lr = start_lr_producer(lr_path, n_target)
    p_bic, bic_log = start_bic_consumer(lr_path, tag, n_target)
    sr_procs, sr_qs, sr_threads, sr_logs = {}, {}, {}, {}
    for arm in SR_ARMS:
        p, q, th, lg = start_sr_consumer(tag, arm, n_target)
        sr_procs[arm], sr_qs[arm], sr_threads[arm], sr_logs[arm] = p, q, th, lg

    n = 0
    lr_nbytes = lw * lh * 3
    while n < n_target:
        buf = p_lr.stdout.read(lr_nbytes)
        if len(buf) < lr_nbytes:
            break
        lr = np.frombuffer(buf, dtype=np.uint8).reshape(lh, lw, 3)
        for arm in SR_ARMS:
            sr_qs[arm].put(infer_rgb(models[arm], lr).tobytes())
        n += 1
        if n % 2000 == 0:
            print(f"  bbb{res} frames {n}/{n_target}", flush=True)

    if n != n_target:
        raise RuntimeError(f"LR frame count {n} != {n_target} bbb{res}")
    check_rc(p_lr, "lr")
    for arm in SR_ARMS:
        sr_qs[arm].put(None)
    for arm in SR_ARMS:
        sr_threads[arm].join()
        check_rc(sr_procs[arm], f"{tag}_{arm}")
    check_rc(p_bic, f"{tag}_bicubic")

    arm_logs = {"bicubic": bic_log}
    arm_logs.update({arm: sr_logs[arm] for arm in SR_ARMS})
    frame_scores = {}
    for arm, lg in arm_logs.items():
        s = read_vmaf_json(os.path.join(TMP, lg))
        if len(s) != n_target:
            raise RuntimeError(f"vmaf {arm} len {len(s)} != {n_target} bbb{res}")
        frame_scores[arm] = s

    if smoke:
        print(f"SMOKE bbb{res} {n} frames OK; arm means:", flush=True)
        for arm in ["bicubic"] + SR_ARMS:
            print(f"  {arm}: {np.mean(frame_scores[arm]):.3f}", flush=True)
        for lg in arm_logs.values():
            try: os.remove(os.path.join(TMP, lg))
            except OSError: pass
        return frame_scores

    methods = {}
    for arm in ["bicubic"] + SR_ARMS:
        kind, level = METHOD_META[arm]
        if level is None:
            wp, wmd5 = None, None
        else:
            wp_full = weight_path(level, scale)
            wp = os.path.relpath(wp_full, REPO)
            wmd5 = md5_file(wp_full)
        methods[arm] = {"kind": kind, "weights": wp, "weights_md5": wmd5,
                        "frames": [round(float(x), 4) for x in frame_scores[arm]]}

    result = {
        "video": "bbb",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "ref_video": os.path.relpath(REF_VIDEO, REPO),
        "input_video": os.path.relpath(lr_path, REPO),
        "input_size": [lw, lh], "target_size": [W, H],
        "scale": scale, "fps": FPS,
        "frames_total": TOTAL_FRAMES, "full_segments": FULL_SEGMENTS,
        "frames_per_seg": FRAMES_PER_SEG,
        "libvmaf_model": "default vmaf_v0.6.1", "libvmaf_n_threads": 1,
        "methods": methods,
    }
    out_path = os.path.join(OUT_DIR, f"bbb_{res}.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), separators=(",", ":"))
    for lg in arm_logs.values():
        try: os.remove(os.path.join(TMP, lg))
        except OSError: pass
    print(f"DONE bbb{res} -> {out_path}", flush=True)


def measure_bic_only(res):
    lr_path, (lw, lh) = BIC_ONLY[res]
    tag = f"bbb{res}"
    proc, log_name = start_bic_consumer(lr_path, tag, TOTAL_FRAMES)
    check_rc(proc, f"{tag}_bicubic")
    frames = read_vmaf_json(os.path.join(TMP, log_name))
    if len(frames) != TOTAL_FRAMES:
        raise RuntimeError(f"vmaf len {len(frames)} != {TOTAL_FRAMES} bbb{res}")
    result = {
        "video": "bbb",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "ref_video": os.path.relpath(REF_VIDEO, REPO),
        "input_video": os.path.relpath(lr_path, REPO),
        "input_size": [lw, lh], "target_size": [W, H],
        "scale": 1, "fps": FPS,
        "frames_total": TOTAL_FRAMES, "full_segments": FULL_SEGMENTS,
        "frames_per_seg": FRAMES_PER_SEG,
        "libvmaf_model": "default vmaf_v0.6.1", "libvmaf_n_threads": 1,
        "methods": {
            "bicubic": {"kind": "ffmpeg_scale_bicubic_yuv", "weights": None,
                        "weights_md5": None,
                        "frames": [round(float(x), 4) for x in frames]},
        },
    }
    out_path = os.path.join(OUT_DIR, f"bbb_{res}.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), separators=(",", ":"))
    try: os.remove(os.path.join(TMP, log_name))
    except OSError: pass
    print(f"DONE bbb{res} -> {out_path}", flush=True)


def output_done(res):
    p = os.path.join(OUT_DIR, f"bbb_{res}.json")
    if not os.path.exists(p):
        return False
    d = json.load(open(p, "r", encoding="utf-8"))
    narms = 1 if res == 720 else 4
    return (d.get("frames_total") == TOTAL_FRAMES
            and len(d["methods"]) == narms
            and all(len(m["frames"]) == TOTAL_FRAMES for m in d["methods"].values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--ffmpeg", default=None,
                    help="ffmpeg.exe 绝对路径；默认取 BONES_FFMPEG 或 PATH")
    args = ap.parse_args()
    if args.ffmpeg:
        global FFMPEG
        FFMPEG = args.ffmpeg

    if args.smoke:
        t0 = datetime.now()
        measure_sr_input(args.only or 240, smoke=True)
        dt = (datetime.now() - t0).total_seconds()
        print(f"SMOKE wall {dt:.1f}s for {FRAMES_PER_SEG} frames; "
              f"extrapolated per SR input full film ~{dt*TOTAL_FRAMES/FRAMES_PER_SEG/60:.1f} min",
              flush=True)
        return

    for res in (240, 360, 480):
        if args.only and args.only != res:
            continue
        if output_done(res):
            print(f"SKIP bbb{res} (done)")
            continue
        measure_sr_input(res)
    if not args.only or args.only == 720:
        if output_done(720):
            print("SKIP bbb720 (done)")
        else:
            measure_bic_only(720)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
