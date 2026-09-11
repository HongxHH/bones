"""Step3 严格同口径：NanoVSR vs IMDN-DIV2K(泛化) vs IMDN-BBB(微调)，并看非单调与内容相关性。"""
import json, os
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
fair = json.load(open(os.path.join(ROOT, "workspace/enhancer_survey/step3_fair16.json"), encoding="utf-8"))
bbb = json.load(open(os.path.join(ROOT, "workspace/enhancer_survey/step2_survey16.json"), encoding="utf-8"))
cf = json.load(open(os.path.join(ROOT, "data/bbb/per_frame_quality.json"), encoding="utf-8"))["content_features"]

rows = []
for k in fair:
    seg = int(k.split("_")[1]); r = fair[k]
    if "nano" not in r or "imdn_div" not in r: continue
    b = bbb[k]
    mot = float(np.mean(cf[str(seg)]["motion"])); tex = float(np.mean(cf[str(seg)]["texture"]))
    rows.append(dict(seg=seg, base=r["json_base"], div=r["imdn_div"], nano=r["nano"],
                     bbb=b["imdn"], d_nd=r["nano"]-r["imdn_div"], d_nb=r["nano"]-b["imdn"],
                     home=b["imdn"]-r["imdn_div"], motion=mot, texture=tex))
rows.sort(key=lambda x: x["seg"])

m = lambda f: np.mean([f(x) for x in rows])
print(f"段数={len(rows)}  (全部同口径：逐帧 libvmaf)\n")
print(f"base 均值           = {m(lambda x:x['base']):.2f}")
print(f"IMDN-DIV2K(泛化)均值= {m(lambda x:x['div']):.2f}")
print(f"NanoVSR(泛化)均值   = {m(lambda x:x['nano']):.2f}")
print(f"IMDN-BBB(微调)均值  = {m(lambda x:x['bbb']):.2f}")
dnd = np.array([x["d_nd"] for x in rows]); dnb = np.array([x["d_nb"] for x in rows])
print(f"\nNano - IMDN-DIV2K: 均值{dnd.mean():+.2f} 中位{np.median(dnd):+.2f} 范围[{dnd.min():+.1f},{dnd.max():+.1f}] "
      f"赢{(dnd>0.3).sum()}/输{(dnd<-0.3).sum()}/平{((np.abs(dnd)<=0.3)).sum()}")
print(f"Nano - IMDN-BBB : 均值{dnb.mean():+.2f} 赢{(dnb>0.3).sum()}/输{(dnb<-0.3).sum()}")
print(f"IMDN 主场红利(bbb-div) = {m(lambda x:x['home']):+.2f}")
from scipy.stats import spearmanr
mot=np.array([x["motion"] for x in rows]); tex=np.array([x["texture"] for x in rows])
print(f"\nSpearman: (Nano-DIV)~motion={spearmanr(dnd,mot).statistic:+.3f}  ~texture={spearmanr(dnd,tex).statistic:+.3f}")
print("\nseg  base   DIV    Nano   BBB   |N-DIV  N-BBB  home | motion texture")
for x in rows:
    print(f"{x['seg']:3d} {x['base']:6.2f} {x['div']:6.2f} {x['nano']:6.2f} {x['bbb']:6.2f} | "
          f"{x['d_nd']:+6.2f} {x['d_nb']:+6.2f} {x['home']:+5.1f} | {x['motion']:6.2f} {x['texture']:8.1f}")
