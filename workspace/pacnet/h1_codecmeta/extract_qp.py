# -*- coding: utf-8 -*-
"""H1 数据通路：解析 `ffmpeg -threads 1 -debug qp` 文本，得到逐 16x16 宏块 QP，
上采样成与帧对齐的 QP 图。

单线程下每帧格式固定：一行 "New frame, type: X"，随后 nMBH 行纯数字，
每行 nMBW 个两位 QP 紧密拼接（426x178 → 27x12=324）。只收“去前缀后整行纯数字、
偶数长度”的行，其余日志（nal_unit_type / graph / detected cores）自然排除；
每帧必须凑满 324，否则标记 MISMATCH。h264 解码器不导出 QP side data，这是官方命令行出口。
"""
import os, re, io, subprocess
import numpy as np
import av

FFMPEG = "ffmpeg"


def parse_qp_debug(path, n_frames=120):
    p = subprocess.run([FFMPEG, "-hide_banner", "-threads", "1", "-debug", "qp",
                        "-i", path, "-frames:v", str(n_frames), "-an", "-f", "null", "-"],
                       capture_output=True, text=True, errors="ignore")
    frames = []
    cur = None
    for raw in p.stderr.splitlines():
        s = re.sub(r"^\[[^\]]*\]\s*", "", raw).strip()
        m = re.match(r"New frame,\s*type:\s*(\w+)", s)
        if m:
            cur = []
            frames.append((m.group(1), cur))
            continue
        if cur is not None and s.isdigit() and len(s) >= 2 and len(s) % 2 == 0:
            cur.extend(int(s[j:j+2]) for j in range(0, len(s), 2))
    return frames


def decode_frames(path, n_frames=120):
    cont = av.open(path); st = cont.streams.video[0]
    W, H = st.codec_context.width, st.codec_context.height
    rgbs, ptypes = [], []
    for fi, fr in enumerate(cont.decode(st)):
        if fi >= n_frames:
            break
        rgbs.append(fr.to_ndarray(format="rgb24")); ptypes.append(str(fr.pict_type))
    cont.close()
    return rgbs, ptypes, W, H


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = r"E:\dev\Code\Python\bones\workspace\pacnet\hard_survey\lr_426_crf34.mp4"
    outdir = r"E:\dev\Code\Python\bones\workspace\pacnet\h1_codecmeta"
    N = 480
    qp_frames = parse_qp_debug(path, N)
    rgbs, ptypes, W, H = decode_frames(path, N)
    nmbw, nmbh = int(np.ceil(W/16)), int(np.ceil(H/16))
    nmb = nmbw * nmbh
    type_map = {"1": "I", "2": "P", "3": "B"}
    log = io.open(os.path.join(outdir, "extract_qp.txt"), "w", encoding="utf-8")
    log.write(f"W={W} H={H} MB={nmbw}x{nmbh}={nmb}; qp_frames={len(qp_frames)} decode={len(rgbs)}\n")
    ok, typo = 0, 0
    qp_maps = {}
    for i, (t, qps) in enumerate(qp_frames):
        good = len(qps) == nmb and (not qps or max(qps) <= 51)
        if i < len(ptypes) and type_map.get(ptypes[i]) != t:
            typo += 1
        if good:
            ok += 1
            qpmb = np.array(qps, np.float32).reshape(nmbh, nmbw)
            qp_maps[i] = np.repeat(np.repeat(qpmb, 16, 0), 16, 1)[:H, :W]
        elif i < 8:
            log.write(f"frame {i} {t} nqp={len(qps)} MISMATCH\n")
    log.write(f"exact-{nmb} frames: {ok}/{len(qp_frames)}; type-mismatch vs PyAV: {typo}\n")
    scored = sorted(((float(q.std()), i) for i, q in qp_maps.items()
                     if i < len(ptypes) and ptypes[i] != "1"),
                    reverse=True)
    for k, (sd, i) in enumerate(scored[:4]):
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), dpi=110)
        ax[0].imshow(rgbs[i]); ax[0].set_title(f"frame {i} {ptypes[i]} decoded"); ax[0].axis("off")
        im = ax[1].imshow(qp_maps[i], cmap="magma", vmin=15, vmax=45)
        ax[1].set_title(f"QP map mean={qp_maps[i].mean():.1f} std={sd:.1f}"); ax[1].axis("off")
        fig.colorbar(im, ax=ax[1], fraction=0.025)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"qp_check_{k}_f{i}.png")); plt.close(fig)
    if scored:
        i = scored[0][1]
        np.savez_compressed(os.path.join(outdir, "qp_align_sample.npz"),
                            rgb=rgbs[i], qp=qp_maps[i], frame_idx=i, W=W, H=H)
    log.write("top QP-variance frames: " + str([i for _, i in scored[:4]]) + "\nDONE\n")
    log.close(); print("done exact", ok, "/", len(qp_frames), "typo", typo)


if __name__ == "__main__":
    main()
