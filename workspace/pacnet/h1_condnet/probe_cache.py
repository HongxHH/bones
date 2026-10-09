# -*- coding: utf-8 -*-
"""小样本自检：单个 BBB 退化视频上前/后向 MV、QP、条件张量是否正常。"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import codec_lib as C

vp = os.path.join(HERE, "deg", "bbb_426_crf34.mp4")
wanted = [120, 200, 280]
mv, W, H = C.scan_mv(vp, wanted)
qp = C.parse_qp(vp, wanted, H, W, max_frames=320)
print("W,H", W, H, "nMB", int(np.ceil(W/16))*int(np.ceil(H/16)))
for i in wanted:
    rgb, fwd, bwd = mv[i]
    q = qp.get(i)
    fmag = np.hypot(fwd[0], fwd[1]); bmag = np.hypot(bwd[0], bwd[1])
    print(f"f{i}: rgb{rgb.shape} fwd_cov={fwd[2].mean():.2f} med|f|="
          f"{np.median(fmag[fwd[2]>0]) if fwd[2].mean()>0 else 0:.2f} "
          f"bwd_cov={bwd[2].mean():.2f} med|b|="
          f"{np.median(bmag[bwd[2]>0]) if bwd[2].mean()>0 else 0:.2f} "
          f"qp={'NA' if q is None else f'{q.mean():.1f}/{q.std():.1f}'}")
    c = np.stack([(q if q is not None else np.zeros((H, W), np.float32))/C.QP_SCALE,
                  fwd[0]/C.MV_SCALE, fwd[1]/C.MV_SCALE, fwd[2],
                  bwd[0]/C.MV_SCALE, bwd[1]/C.MV_SCALE, bwd[2]], 0).astype(np.float32)
    print("   cond", c.shape, "finite", np.isfinite(c).all(),
          "range", round(float(c.min()),3), round(float(c.max()),3))
print("PROBE_OK")
