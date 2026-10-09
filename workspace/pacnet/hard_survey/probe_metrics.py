# -*- coding: utf-8 -*-
import io, traceback, torch
import pyiqa
out = io.open(r"E:\dev\Code\Python\bones\workspace\pacnet\hard_survey\probe_metrics.txt", "w", encoding="utf-8")
def log(*a): out.write(" ".join(str(x) for x in a) + "\n"); out.flush()
dev = "cuda"
try:
    torch.manual_seed(0)
    x = torch.rand(1, 3, 256, 256, device=dev).clamp(0, 1)
    y = torch.rand(1, 3, 256, 256, device=dev).clamp(0, 1)
    fr = {}
    for name in ["lpips", "dists", "cw_ssim", "vif"]:
        m = pyiqa.create_metric(name, device=dev)
        fr[name] = m
        same = float(m(x, x)); diff = float(m(x, y))
        log(f"[FR] {name:8s} same={same:.4f} diff={diff:.4f}  (higher/lower-better see doc)")
    for name in ["clipiqa"]:
        m = pyiqa.create_metric(name, device=dev)
        log(f"[NR] {name:8s} score(x)={float(m(x)):.4f}  (higher=better)")
    log("ALL METRICS OK")
except Exception:
    log("ERROR"); log(traceback.format_exc())
out.close()
print("done")
