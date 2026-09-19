# -*- coding: utf-8 -*-
"""H1 数据通路验收：对同一帧生成 LR 帧 + 前向 MV 场 + QP 图（全部 HxW 对齐），
存 npz 并画三联图；同时统计解码过程出现过的全部 side data 类型（确认有无残差/QP 表）。"""
import os, re, io, subprocess
import numpy as np
import av
from extract_mv import decode_with_mv, mv_field
from extract_qp import parse_qp_debug

TARGET = 275
path = r"E:\dev\Code\Python\bones\workspace\pacnet\hard_survey\lr_426_crf34.mp4"
outdir = r"E:\dev\Code\Python\bones\workspace\pacnet\h1_codecmeta"

# MV + 帧（PyAV）
frames, W, H = decode_with_mv(path, max_frames=TARGET + 4)
# 统计解码过程出现的全部 side data 类型（确认有无残差/QP 表）
cont = av.open(path); st = cont.streams.video[0]
st.codec_context.flags2 = int(st.codec_context.flags2) | (1 << 28)
side_types = set()
for fi, fr in enumerate(cont.decode(st)):
    if fi > TARGET + 4: break
    for sd in fr.side_data:
        side_types.add(sd.type.name)
cont.close()

fi, ptype, rgb, recs = frames[TARGET]
mvx, mvy, cover = mv_field(H, W, recs)

# QP（ffmpeg 单线程 debug qp）
qpf = parse_qp_debug(path, n_frames=TARGET + 4)
nmbw, nmbh = int(np.ceil(W/16)), int(np.ceil(H/16))
t, qps = qpf[TARGET]
assert len(qps) == nmbw*nmbh, (len(qps), nmbw*nmbh)
qpmb = np.array(qps, np.float32).reshape(nmbh, nmbw)
qp = np.repeat(np.repeat(qpmb, 16, 0), 16, 1)[:H, :W]

np.savez_compressed(os.path.join(outdir, f"codec_features_f{TARGET}.npz"),
                    rgb=rgb, mvx=mvx, mvy=mvy, cover=cover, qp=qp,
                    frame_idx=TARGET, W=W, H=H)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16, 3.4), dpi=120)
ax[0].imshow(rgb); ax[0].set_title(f"LR decoded frame {TARGET} {ptype}"); ax[0].axis("off")
ax[1].imshow(rgb); ax[1].set_title("forward MV (px)")
step = 8
ys, xs = np.mgrid[0:H:step, 0:W:step]
vx, vy, cv = mvx[ys, xs], mvy[ys, xs], cover[ys, xs]
m = cv > 0
ax[1].quiver(xs[m], ys[m], vx[m], vy[m], color="lime", scale=80, width=0.004, headwidth=4); ax[1].axis("off")
im = ax[2].imshow(qp, cmap="magma", vmin=15, vmax=45)
ax[2].set_title(f"QP map mean={qp.mean():.1f}"); ax[2].axis("off")
fig.colorbar(im, ax=ax[2], fraction=0.03)
fig.tight_layout(); fig.savefig(os.path.join(outdir, f"codec_features_f{TARGET}.png")); plt.close(fig)

log = io.open(os.path.join(outdir, "combine_features.txt"), "w", encoding="utf-8")
log.write(f"side data types seen: {sorted(side_types)}\n")
log.write(f"frame {TARGET} {ptype}: rgb{rgb.shape} mvx{mvx.shape} qp{qp.shape} "
          f"cover={cover.mean():.2f} mvx[{mvx.min():.2f},{mvx.max():.2f}] "
          f"mvy[{mvy.min():.2f},{mvy.max():.2f}] qp[{qp.min():.0f},{qp.max():.0f}]\n")
log.write("NOTE: no residual/prediction side data in h264 decoder; only MOTION_VECTORS "
          "(+SEI). Residual needs decoder patch / reference SW or pixel-domain approximation.\nDONE\n")
log.close()
print("done side_types", sorted(side_types))
