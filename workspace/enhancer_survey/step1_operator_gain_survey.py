"""
Step1: 替代增强算子离线增益普查（同口径对齐 measure_per_frame_vmaf）。

目的（决定"换增强地基"方向生死的第一问）：
  现有 IMDN 单帧超分的增益曲线"对级别单调、段平均后鲁棒"，选择层无差异可挖。
  本脚本换用其他增强算子，在完全相同的口径下测逐段 VMAF，回答：
    Q1 新算子整体增益是否高于 IMDN 各级；
    Q2 增益是否内容非单调（不同段相对优势是否变化），这是"选择/协同"能否重生的前提。

同口径保证（与 workspace/measure_per_frame_vmaf.py 完全一致）：
  - 源帧从 data/video_mp4box/output_{res}p.mp4 按段定位，GT 为 output_1080p.mp4；
  - base = 双三次放大到 1920x1080；所有算子输出统一 bicubic 对齐到 1080；
  - 逐帧 ffmpeg libvmaf（enh/ref 都经相同 cv2 mp4v 临时编码，对比公平）；
  - IMDN 各级对照直接读 data/bbb/per_frame_quality.json（不重跑）。
  - 自检：本脚本重测的 base 必须与 json 内 {res}_base_{seg} 一致，否则口径有误。

算子通过 OPERATORS 注册，新增模型只需实现 (frames_bgr, scale)->frames_bgr(1080)。
"""
import os, sys, json, time, subprocess, tempfile, argparse
import numpy as np
import cv2
import torch

ROOT = r"E:\dev\Code\Python\bones"
VIDEO_DIR = os.path.join(ROOT, "data/video_mp4box")
PFQ = os.path.join(ROOT, "data/bbb/per_frame_quality.json")
WORK_DIR = os.path.join(ROOT, "workspace/enhancer_survey/vmaf_tmp")
OUT = os.path.join(ROOT, "workspace/enhancer_survey/step1_survey.json")
FPS, FRAMES_PER_SEG, DEVICE = 30, 120, "cuda:0"
RES_SCALE = {240: 4, 360: 3, 480: 2}


# ---------- 算子 ----------
def op_base(frames, scale):
    return [cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC) for f in frames]


class NAFNetDeblur:
    def __init__(self):
        sys.path.insert(0, os.path.join(ROOT, "enhance/NAFNet"))
        from basicsr.models.archs.NAFNet_arch import NAFNetLocal
        net = NAFNetLocal(img_channel=3, width=32, enc_blk_nums=[1, 1, 1, 28],
                          middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
        w = os.path.join(ROOT, "enhance/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width32.pth")
        sd = torch.load(w, map_location="cpu"); sd = sd.get("params", sd)
        net.load_state_dict(sd, strict=True); net.eval().to(DEVICE)
        self.net = net

    def __call__(self, frames, scale):
        out = []
        for f in frames:
            up = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                o = self.net(t)
            o = o.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            out.append(cv2.cvtColor((o * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        return out


def _yuv_upscale_to_bgr(y_lr, uv_bgr_lr):
    """Y(LR,0..1) 与中间帧 BGR(LR) -> Y/UV 分别 bicubic 到 1080 -> BGR。"""
    y = (y_lr.clip(0, 1) * 255).round().astype(np.uint8)
    yup = cv2.resize(y, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    yuv = cv2.cvtColor(uv_bgr_lr, cv2.COLOR_BGR2YUV)
    uup = cv2.resize(yuv[:, :, 1], (1920, 1080), interpolation=cv2.INTER_CUBIC)
    vup = cv2.resize(yuv[:, :, 2], (1920, 1080), interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(np.stack([yup, uup, vup], -1), cv2.COLOR_YUV2BGR)


def op_yuv_up(frames, scale):
    """对照：仅走 YUV 分离/分通道 bicubic 放大，不增强，检验 YUV 往返是否中性。"""
    out = []
    for f in frames:
        y = cv2.cvtColor(f, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32) / 255.0
        out.append(_yuv_upscale_to_bgr(y, f))
    return out


class STDFEnhancer:
    """LR 尺度 7 帧时序压缩复原（STDF, AAAI'20），再 bicubic 到 1080。"""
    def __init__(self, weight="MFQEv2_R3_enlarge300x/ckp_290000.pt", radius=3):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from collections import OrderedDict
        from stdf_model import MFVQE
        net = MFVQE(radius=radius)
        ck = torch.load(os.path.join(ROOT, "enhance/stdf-pytorch/exp", weight), map_location="cpu")
        sd = ck.get("state_dict", ck)
        if "module." in list(sd)[0]:
            sd = OrderedDict((k[7:], v) for k, v in sd.items())
        net.load_state_dict(sd, strict=True); net.eval().to(DEVICE)
        self.net, self.radius = net, radius

    def enhance_lr(self, frames):
        """LR 尺度 7 帧时序去压缩伪影，返回去块后的 LR BGR 帧（不放大）。"""
        n = len(frames)
        ys = [cv2.cvtColor(f, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32) / 255.0 for f in frames]
        h, w = ys[0].shape
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        out = []
        for i in range(n):
            idxs = np.clip(np.arange(i - self.radius, i + self.radius + 1), 0, n - 1)
            stack = np.stack([np.pad(ys[j], ((0, ph), (0, pw)), mode="edge") for j in idxs], 0)
            t = torch.from_numpy(stack).unsqueeze(0).float().to(DEVICE)
            with torch.no_grad():
                ey = self.net(t)[0, 0, :h, :w].clamp(0, 1).cpu().numpy()
            ey = (ey * 255).round().astype(np.uint8)
            yuv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2YUV); yuv[:, :, 0] = ey
            out.append(cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR))
        return out

    def __call__(self, frames, scale):
        lr = self.enhance_lr(frames)
        return [_yuv_upscale_to_bgr(
            cv2.cvtColor(f, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32) / 255.0, f) for f in lr]


class IMDNSR:
    """IMDN-high 超分（player 内实现）。dataset: 'bbb'=目标视频微调版 / 'DIV2K'=通用版，输出对齐 1080。"""
    def __init__(self, dataset="bbb"):
        sys.path.insert(0, os.path.join(ROOT, "player"))
        from istream_player.modules.enhancer.imdn_model import IMDN
        wdir = os.path.join(ROOT, "player/istream_player/modules/enhancer/imdn_weight")
        self.dataset = dataset
        self.models = {}
        for scale in (2, 3, 4):
            m = IMDN(upscale=scale, nf=32)
            m.load_state_dict(torch.load(
                os.path.join(wdir, f"IMDN_high_{dataset}_x{scale}.pth"), map_location="cpu"))
            self.models[scale] = m.eval().to(DEVICE)

    def __call__(self, frames, scale):
        m = self.models[scale]; out = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                o = torch.nn.functional.interpolate(m(t), size=(1080, 1920), mode="bicubic",
                                                    align_corners=False)
            o = o.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            out.append(cv2.cvtColor((o * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        return out


class Cascade:
    """STDF LR 去块 -> IMDN 超分（多算子协同雏形）。"""
    def __init__(self, stdf, imdn):
        self.stdf, self.imdn = stdf, imdn

    def __call__(self, frames, scale):
        clean = self.stdf.enhance_lr(frames)
        return self.imdn(clean, scale)


class IMDNLeaveOut:
    """leave-segments-out 对症微调得到的 IMDN（仅 x4），用于跨段泛化判决。"""
    def __init__(self, weight):
        sys.path.insert(0, os.path.join(ROOT, "player"))
        from istream_player.modules.enhancer.imdn_model import IMDN
        m = IMDN(upscale=4, nf=32)
        m.load_state_dict(torch.load(weight, map_location="cpu"))
        self.net = m.eval().to(DEVICE)

    def __call__(self, frames, scale):
        assert scale == 4, "leave-out 只训了 x4"
        out = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                o = torch.nn.functional.interpolate(self.net(t), size=(1080, 1920), mode="bicubic",
                                                    align_corners=False)
            o = o.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            out.append(cv2.cvtColor((o * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))
        return out


class NanoVSREnhancer:
    """NanoVSR-644k（ECCV'26 轻量双向循环 VSR，固定 4x），分窗推理后对齐 1080。"""
    T = 15
    def __init__(self):
        sys.path.insert(0, os.path.join(ROOT, "enhance/nanovsr"))
        from models.nanovsr import NanoVSR
        m = NanoVSR(num_feat=48, num_blocks=12, deploy=False)
        ck = torch.load(os.path.join(ROOT, "enhance/nanovsr/checkpoints/nanovsr_644k.pth"),
                        map_location="cpu", weights_only=False)
        sd = ck.get("params", ck.get("state_dict", ck))
        m.load_state_dict(sd, strict=True); m.switch_to_deploy(); m.eval().to(DEVICE)
        self.net = m

    def __call__(self, frames, scale):
        n = len(frames); out = [None] * n
        for s in range(0, n, self.T):
            idxs = list(range(s, min(s + self.T, n)))
            arr = np.stack([cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                            for i in idxs], 0)  # (T,H,W,3)
            t = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(DEVICE)  # (1,T,3,H,W)
            with torch.no_grad():
                o = self.net(t)  # (1,T,3,4H,4W)
                o = torch.nn.functional.interpolate(o[0], size=(1080, 1920), mode="bicubic",
                                                    align_corners=False).clamp(0, 1)
            o = (o.permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            for j, i in enumerate(idxs):
                out[i] = cv2.cvtColor(o[j], cv2.COLOR_RGB2BGR)
        return out


# ---------- 读帧 ----------
def read_seg(cap, seg, n=FRAMES_PER_SEG):
    cap.set(cv2.CAP_PROP_POS_FRAMES, seg * n)
    fs = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        fs.append(f)
    return fs


# ---------- 逐帧 VMAF（与 measure_per_frame_vmaf 同口径） ----------
def per_frame_vmaf(enh, ref, tag):
    os.makedirs(WORK_DIR, exist_ok=True)
    d = tempfile.mkdtemp(dir=WORK_DIR)
    ep, rp = os.path.join(d, "e.mp4"), os.path.join(d, "r.mp4")
    h, w = enh[0].shape[:2]
    we = cv2.VideoWriter(ep, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    wr = cv2.VideoWriter(rp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in enh: we.write(f)
    for f in ref: wr.write(f)
    we.release(); wr.release()
    logn = f"vmaf_{tag}.json"
    lp = os.path.join(WORK_DIR, logn)
    if os.path.exists(lp): os.remove(lp)
    filt = ("[0:v]setpts=PTS-STARTPTS[distorted];"
            "[1:v]setpts=PTS-STARTPTS[reference];"
            f"[distorted][reference]libvmaf=log_path={logn}:log_fmt=json:n_threads=4")
    subprocess.run(["ffmpeg", "-y", "-i", ep, "-i", rp, "-lavfi", filt, "-f", "null", "-"],
                   capture_output=True, text=True, timeout=180, cwd=WORK_DIR)
    import shutil
    if not os.path.exists(lp):
        shutil.rmtree(d, ignore_errors=True); return None
    log = json.load(open(lp)); v = [fr["metrics"]["vmaf"] for fr in log["frames"]]
    os.remove(lp); shutil.rmtree(d, ignore_errors=True)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, nargs="+", default=[240])
    ap.add_argument("--segs", type=int, nargs="+", default=[2, 60, 129])
    ap.add_argument("--ops", nargs="+", default=["base", "yuv_up", "stdf"])
    ap.add_argument("--stdf_weight", default="MFQEv2_R3_enlarge300x/ckp_290000.pt")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--lo_weight", default=os.path.join(ROOT, "workspace/enhancer_survey/weights/IMDN_high_bbbLO_x4_best.pth"))
    args = ap.parse_args()

    pfq = json.load(open(PFQ, encoding="utf-8"))["results"]
    runners = {}
    if "nafnet" in args.ops:
        print("loading NAFNet..."); runners["nafnet"] = NAFNetDeblur()
    if "stdf" in args.ops or "cascade" in args.ops:
        print("loading STDF...", args.stdf_weight); runners["stdf"] = STDFEnhancer(args.stdf_weight)
    if "imdn" in args.ops or "cascade" in args.ops:
        print("loading IMDN-high-bbb..."); runners["imdn"] = IMDNSR("bbb")
    if "imdn_div" in args.ops:
        print("loading IMDN-high-DIV2K..."); runners["imdn_div"] = IMDNSR("DIV2K")
    if "imdn_lo" in args.ops:
        print("loading IMDN leave-out..."); runners["imdn_lo"] = IMDNLeaveOut(args.lo_weight)
    if "nano" in args.ops:
        print("loading NanoVSR-644k..."); runners["nano"] = NanoVSREnhancer()
    if "cascade" in args.ops:
        runners["cascade"] = Cascade(runners["stdf"], runners["imdn"])
    stateless = {"base": op_base, "yuv_up": op_yuv_up}

    result = {}
    ref_cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, "output_1080p.mp4"))
    for res in args.res:
        scale = RES_SCALE[res]
        comp = cv2.VideoCapture(os.path.join(VIDEO_DIR, f"output_{res}p.mp4"))
        for seg in args.segs:
            ref = read_seg(ref_cap, seg); lr = read_seg(comp, seg)
            if len(ref) < FRAMES_PER_SEG or len(lr) < FRAMES_PER_SEG:
                print(f"skip {res} seg{seg}: frames {len(ref)}/{len(lr)}"); continue
            row = {}
            # json 已有对照：base 与 IMDN 三级
            for m in ["base", "sr_low", "sr_medium", "sr_high"]:
                k = f"{res}_{m}_{seg}"
                if k in pfq: row[f"json_{m}"] = float(np.mean(pfq[k]))
            # 现算算子
            cands = dict(stateless); cands.update(runners)
            for opn in args.ops:
                fn = cands.get(opn)
                if fn is None: continue
                t0 = time.time()
                enh = fn(lr, scale)
                v = per_frame_vmaf(enh, ref, f"{res}_{opn}_{seg}")
                row[opn] = float(np.mean(v)) if v else None
                row[f"{opn}_ms"] = round((time.time() - t0) * 1000)
            # 增益（相对 json_base）
            jb = row.get("json_base")
            for m in list(row):
                if m.startswith("json_sr") and jb is not None:
                    row[m + "_gain"] = round(row[m] - jb, 3)
            if jb is not None and row.get("base") is not None:
                row["base_selfcheck_diff"] = round(row["base"] - jb, 3)
            if row.get("imdn") is not None and row.get("json_sr_high") is not None:
                row["imdn_selfcheck_diff"] = round(row["imdn"] - row["json_sr_high"], 3)
            for opn in ("nafnet", "stdf", "yuv_up", "imdn", "imdn_div", "imdn_lo", "cascade", "nano"):
                if jb is not None and row.get(opn) is not None:
                    row[f"{opn}_gain"] = round(row[opn] - jb, 3)
            if row.get("nano") is not None and row.get("imdn") is not None:
                row["nano_minus_imdn"] = round(row["nano"] - row["imdn"], 3)
            if row.get("nano") is not None and row.get("imdn_div") is not None:
                row["nano_minus_imdn_div"] = round(row["nano"] - row["imdn_div"], 3)
            result[f"{res}_{seg}"] = row
            print(f"{res}p seg{seg}: " + "  ".join(f"{k}={v}" for k, v in row.items()
                  if not k.endswith("_ms")))
        comp.release()
    ref_cap.release()
    json.dump(result, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
