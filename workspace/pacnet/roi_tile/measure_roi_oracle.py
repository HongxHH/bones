# -*- coding: utf-8 -*-
"""
原生像素 ROI + 其余超分：等码率离线 oracle（方向生死判定，不碰播放器）。

动机：PAC-Net（加容量）/H1（加码流条件）/H2（加邻帧像素）三路离线证伪，
瓶颈在压缩后幸存的输入像素。本实验把杠杆移到输入端：关键 tile 直接收原生
高分辨率像素（从 1080p 源裁切、libx264 高质量编码，模拟 DASH 高码率 tile），
其余区域仍用低码率整帧 + l2 超分，拼回 1920x800，对 1080p 源逐帧算 VMAF/LPIPS。

码率口径（真实，非独立小 tile 假账）：
  - 低码率整帧 B_low：ffprobe 对退化整流逐 packet 计字节 / 帧数（真实单流码率）。
  - 原生 tile 速率 b_nat[t]：每个 tile 位置在连续 96 帧上以正常 GOP 编码，bits/帧，
    含空间复杂度差异（crf20 近无损、crf26 便宜档）。
  - 换 tile 的边际码率 = b_nat[t] − B_low/60；被替换区域按面积均摊低码率返还
    （ROI 通常更复杂，面积均摊少返还，对本方向偏保守）。
  - ROI 方案 bits/帧 = B_low + Σ_{t∈ROI}(b_nat[t] − B_low/60)。
等码率对手：均匀提分辨率/质量整帧（320/426/640/854 宽 × 多 CRF）+ 全幅 l2，
在 (bits/帧, VMAF) 上建前沿，ROI 点按同码率线性插值比 ΔVMAF/ΔLPIPS。

ROI 选择：perframe（每帧按 Y-MSE 改善排序，上帝上限）/ fixed（跨帧平均，静态掩码）
/ center（到画面中心距离，无参考朴素策略）。

判据（预注册）：同码率下，中/重档 fixed 或 center 稳定 ≥2~3 VMAF 且 LPIPS 同向改善，
方向才活；连 perframe 上限都越不过均匀前沿则止损。

用法：python measure_roi_oracle.py ；SMOKE=1 跑 4 帧 1 档并出对比图。
"""
import os, sys, json, subprocess, tempfile
import numpy as np
import torch, cv2
from scipy.stats import binomtest

sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
sys.path.insert(0, r"E:\dev\Code\Python\bones\workspace\pacnet\hard_survey")
from istream_player.modules.enhancer.imdn_model import IMDN_RTC
import measure_hard_survey as H

DEVICE = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
HS = r"E:\dev\Code\Python\bones\workspace\pacnet\hard_survey"
LAD = os.path.join(HERE, "ladder")
WEIGHT = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight\IMDN_medium_DIV2K_x4.pth"
TILE = 160
COLS, ROWS = 1920 // TILE, 800 // TILE
NT = COLS * ROWS
NATIVE_CRFS = [26, 30, 34]
FRACS = [0.0, 0.05, 0.10, 0.20, 0.40]
BASE_TIERS = ["426_crf26", "426_crf34", "426_crf42", "320_crf26", "320_crf34", "320_crf42"]
# 均匀对手前沿：(标签, 路径)
UNIFORM = ([(f"{w}_crf{c}", os.path.join(HS, f"lr_{w}_crf{c}.mp4"))
            for w in (426, 320) for c in (26, 34, 42)] +
           [(f"{w}_crf{c}", os.path.join(LAD, f"lr_{w}_crf{c}.mp4"))
            for w in (640, 854) for c in (34, 30, 26, 22) if os.path.exists(os.path.join(LAD, f"lr_{w}_crf{c}.mp4"))])
RATE_CHUNK = 96
OUT_JSON = os.path.join(HERE, "roi_oracle.json")

_lpips = None
def lpips_metric():
    global _lpips
    if _lpips is None:
        import lpips
        _lpips = lpips.LPIPS(net="squeeze", verbose=False).to(DEVICE).eval()
    return _lpips

def build_l2():
    m = IMDN_RTC(upscale=4)
    m.load_state_dict(torch.load(WEIGHT, map_location="cpu"))
    return m.to(DEVICE).eval()

def video_bits_per_frame(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "packet=size", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    sizes = [int(x.split(",")[0]) for x in p.stdout.split() if x.strip()]
    return 8.0 * sum(sizes) / len(sizes)

def tile_rect(t, lw, lh):
    r, c = divmod(t, COLS)
    sx, sy = lw / 1920.0, lh / 800.0
    x0 = int(np.floor(c * TILE * sx)); x1 = int(np.ceil((c + 1) * TILE * sx))
    y0 = int(np.floor(r * TILE * sy)); y1 = int(np.ceil((r + 1) * TILE * sy))
    x1 = min(lw, x1); y1 = min(lh, y1)
    if (x1 - x0) % 2: x1 = min(lw, x1 + 1)
    if (y1 - y0) % 2: y1 = min(lh, y1 + 1)
    return max(0, x0), max(0, y0), x1, y1

def encode_tiles(crops, crf, tmp, name, gop=None, all_intra=False):
    """crops: 一个 tile 跨帧 BGR list。返回 (mp4, bits/帧)。"""
    h, w = crops[0].shape[:2]
    raw = np.stack(crops, 0).tobytes()
    mp4 = os.path.join(tmp, f"{name}.mp4")
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
           "-video_size", f"{w}x{h}", "-framerate", str(H.FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    if all_intra:
        cmd += ["-g", "1"]
    else:
        cmd += ["-g", str(gop or len(crops)), "-bf", "0"]
    cmd.append(mp4)
    p = subprocess.run(cmd, input=raw, capture_output=True)
    if p.returncode != 0 or not os.path.exists(mp4):
        raise RuntimeError(f"encode {name}: {p.stderr.decode()[-400:]}")
    return mp4, os.path.getsize(mp4) * 8 / len(crops)

def decode_tiles(mp4, n, t):
    cap = cv2.VideoCapture(mp4); out = []
    while len(out) < n:
        ret, f = cap.read()
        if not ret: break
        if f.shape[0] != TILE or f.shape[1] != TILE:
            f = cv2.resize(f, (TILE, TILE), interpolation=cv2.INTER_CUBIC)
        out.append(f)
    cap.release()
    if len(out) != n: raise RuntimeError(f"decode tile {t}: {len(out)}/{n}")
    return out

def ymse(a, b):
    ya = cv2.cvtColor(a, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float64)
    yb = cv2.cvtColor(b, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float64)
    return float(np.mean((ya - yb) ** 2))

def vmaf_perframe(dist, ref, tmp, tag):
    ep = os.path.join(tmp, f"d_{tag}.mp4"); rp = os.path.join(tmp, f"r_{tag}.mp4")
    log = f"v_{tag}.json"
    h, w = dist[0].shape[:2]; fc = cv2.VideoWriter_fourcc(*"mp4v")
    we, wr = cv2.VideoWriter(ep, fc, H.FPS, (w, h)), cv2.VideoWriter(rp, fc, H.FPS, (w, h))
    for d, r in zip(dist, ref): we.write(d); wr.write(r)
    we.release(); wr.release()
    subprocess.run(["ffmpeg", "-i", ep, "-i", rp, "-lavfi",
                    f"libvmaf=log_path={log}:log_fmt=json", "-f", "null", "-"],
                   cwd=tmp, capture_output=True, text=True, timeout=300)
    with open(os.path.join(tmp, log), "r", encoding="utf-8") as f:
        j = json.load(f)
    return [float(fr["metrics"]["vmaf"]) for fr in j["frames"]]

def _t(img):
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    return t / 127.5 - 1.0

def lpips_perframe(dist, ref):
    m = lpips_metric(); out = []
    with torch.no_grad():
        for d, r in zip(dist, ref):
            out.append(float(m(_t(d), _t(r)).item()))
    return out

def sign_p(delta):
    pos = int(np.sum(np.array(delta) > 0))
    try: return float(binomtest(pos, len(delta), 0.5, alternative="two-sided").pvalue)
    except Exception: return None

def paste(frame, t, img):
    r, c = divmod(t, COLS)
    frame[r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE] = img

def interp_frontier(xs, ys, x):
    order = np.argsort(xs); xs, ys = np.array(xs)[order], np.array(ys)[order]
    if x <= xs[0]: return float(ys[0])
    if x >= xs[-1]: return float(ys[-1])
    return float(np.interp(x, xs, ys))

def main():
    smoke = os.environ.get("SMOKE") == "1"
    n = 4 if smoke else H.N
    base_tiers = ["426_crf34"] if smoke else BASE_TIERS
    native_crfs = [26] if smoke else NATIVE_CRFS
    l2 = build_l2()

    cap = cv2.VideoCapture(H.REF_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idxs = np.linspace(0, total - 10, n, dtype=int)
    refs = H.read_frames(H.REF_VIDEO, idxs)
    centers = np.array([[(c + 0.5) * TILE, (r + 0.5) * TILE]
                        for r in range(ROWS) for c in range(COLS)])
    center_order = np.argsort(np.linalg.norm(centers - np.array([960.0, 400.0]), axis=1))
    print(f"frames={n} tiles={NT}", flush=True)

    results = {"config": {"frames": n, "tile": TILE, "n_tiles": NT, "native_crfs": native_crfs,
                          "fracs": FRACS, "lpips_impl": "lpips_pkg_v0.1_squeeze",
                          "frame_idx": [int(i) for i in idxs]},
               "uniform_frontier": [], "tiers": {}}

    with tempfile.TemporaryDirectory() as tmp:
        # ---- 原生 tile：速率（连续帧，正常 GOP）+ 质量（采样帧 all-intra）----
        rate_start = int(total * 0.45)
        rate_idxs = np.arange(rate_start, rate_start + RATE_CHUNK)
        rate_refs = H.read_frames(H.REF_VIDEO, rate_idxs)
        nat_bits = {crf: [0.0] * NT for crf in native_crfs}
        nat_dec = {crf: [None] * NT for crf in native_crfs}
        for t in range(NT):
            r, c = divmod(t, COLS)
            rcrops = [f[r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE] for f in rate_refs]
            qcrops = [f[r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE] for f in refs]
            for crf in native_crfs:
                _, bits = encode_tiles(rcrops, crf, tmp, f"rate_{crf}_{t}", gop=48)
                nat_bits[crf][t] = bits
                mp4, _ = encode_tiles(qcrops, crf, tmp, f"qual_{crf}_{t}", all_intra=True)
                nat_dec[crf][t] = decode_tiles(mp4, n, t)
            if t % 10 == 0: print(f"native tile {t}/{NT}", flush=True)

        # ---- 均匀对手前沿：真实码率 + l2 质量 ----
        uni_x, uni_v, uni_l = [], [], []
        for tag, path in UNIFORM:
            if not os.path.exists(path): continue
            bpf = video_bits_per_frame(path)
            fr = H.read_frames(path, idxs)
            if len(fr) != n: continue
            sr = [H.infer(l2, f) for f in fr]
            v = float(np.mean(vmaf_perframe(sr, refs, tmp, f"uni_{tag}")))
            l = float(np.mean(lpips_perframe(sr, refs)))
            uni_x.append(bpf); uni_v.append(v); uni_l.append(l)
            results["uniform_frontier"].append({"tag": tag, "bits_per_frame": bpf, "vmaf": v, "lpips": l})
            print(f"[uniform] {tag} bits={bpf:.0f} vmaf={v:.2f} lpips={l:.4f}", flush=True)

        for tier in base_tiers:
            low_path = H.DEG[tier]
            low = H.read_frames(low_path, idxs)
            lh, lw = low[0].shape[:2]
            B_low = video_bits_per_frame(low_path)
            base_sr = [H.infer(l2, f) for f in low]
            base_vmaf = vmaf_perframe(base_sr, refs, tmp, f"{tier}_base")
            base_lpips = lpips_perframe(base_sr, refs)
            bic = [H.bic(f) for f in low]
            bic_vmaf = float(np.mean(vmaf_perframe(bic, refs, tmp, f"{tier}_bic")))

            # 每帧每 tile 原生（crf26）相对超分的 Y-MSE 改善（选 tile 用，与质量档弱相关）
            gain = np.zeros((n, NT))
            qref = nat_dec[native_crfs[0]]
            for fi in range(n):
                for t in range(NT):
                    r, c = divmod(t, COLS)
                    srt = base_sr[fi][r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE]
                    rt = refs[fi][r*TILE:(r+1)*TILE, c*TILE:(c+1)*TILE]
                    gain[fi, t] = ymse(srt, rt) - ymse(qref[t][fi], rt)
            fixed_order = np.argsort(-gain.mean(0))

            tier_res = {"lw": lw, "lh": lh, "B_low": B_low,
                        "bicubic_vmaf": bic_vmaf,
                        "baseline_l2": {"vmaf": float(np.mean(base_vmaf)),
                                        "lpips": float(np.mean(base_lpips))},
                        "arms": {}}
            for crf in native_crfs:
                for strat in ["perframe", "fixed", "center"]:
                    rows = []
                    for frac in FRACS:
                        k = int(round(frac * NT))
                        comp, bseq = [], []
                        for fi in range(n):
                            fr = base_sr[fi].copy()
                            order = (np.argsort(-gain[fi]) if strat == "perframe"
                                     else fixed_order if strat == "fixed" else center_order)
                            sel = order[:k]
                            for t in sel: paste(fr, int(t), nat_dec[crf][int(t)][fi])
                            comp.append(fr)
                            bits = B_low + float(sum(nat_bits[crf][int(t)] - B_low / NT for t in sel))
                            bseq.append(bits)
                        vv = vmaf_perframe(comp, refs, tmp, f"{tier}_{crf}_{strat}_{int(frac*100)}")
                        ll = lpips_perframe(comp, refs)
                        bits = float(np.mean(bseq))
                        dv = [vv[i] - base_vmaf[i] for i in range(n)]
                        uni_at_v = interp_frontier(uni_x, uni_v, bits)
                        uni_at_l = interp_frontier(uni_x, uni_l, bits)
                        rows.append({"frac": frac, "k": k, "bits_per_frame": bits,
                                     "vmaf": float(np.mean(vv)),
                                     "dvmaf_vs_l2": float(np.mean(dv)),
                                     "dvmaf_vs_uniform_equalbits": float(np.mean(vv) - uni_at_v),
                                     "lpips": float(np.mean(ll)),
                                     "uniform_lpips_equalbits": uni_at_l,
                                     "dlpips_vs_uniform_equalbits": uni_at_l - float(np.mean(ll)),
                                     "vmaf_sign_p": sign_p(dv)})
                        if smoke:
                            print(f"[smoke] {tier} crf{crf} {strat} f={frac} bits={bits:.0f} "
                                  f"v={np.mean(vv):.2f} dV_l2={np.mean(dv):+.2f} dV_uni={np.mean(vv)-uni_at_v:+.2f}", flush=True)
                    tier_res["arms"][f"crf{crf}_{strat}"] = rows
            results["tiers"][tier] = tier_res
            print(f"=== {tier} B_low={B_low:.0f} base_vmaf={tier_res['baseline_l2']['vmaf']:.2f} ===", flush=True)

            if smoke:
                k = int(0.2 * NT); crf = native_crfs[0]
                cf, cp = [], []
                for fi in range(n):
                    a = base_sr[fi].copy(); b = base_sr[fi].copy()
                    for t in fixed_order[:k]: paste(a, int(t), nat_dec[crf][int(t)][fi])
                    for t in np.argsort(-gain[fi])[:k]: paste(b, int(t), nat_dec[crf][int(t)][fi])
                    cf.append(a); cp.append(b)
                strip = np.hstack([cv2.resize(bic[0], (384, 160)), cv2.resize(base_sr[0], (384, 160)),
                                   cv2.resize(cf[0], (384, 160)), cv2.resize(cp[0], (384, 160)),
                                   cv2.resize(refs[0], (384, 160))])
                cv2.imwrite(os.path.join(HERE, "smoke_strip.png"), strip)
                print("smoke_strip.png", flush=True)

    if not smoke:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print("SAVED", OUT_JSON, flush=True)
    print("ROI_ORACLE_DONE", flush=True)

if __name__ == "__main__":
    main()
