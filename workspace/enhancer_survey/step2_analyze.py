"""Step2 分析：NanoVSR(VSR) vs IMDN-high(ISR) 跨段非单调性与内容相关性、异构选择 Oracle 上限。"""
import json, os, sys
import numpy as np

ROOT = r"E:\dev\Code\Python\bones"
SURVEY = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "workspace/enhancer_survey/step2_survey16.json")
PFQ = os.path.join(ROOT, "data/bbb/per_frame_quality.json")

s = json.load(open(SURVEY, encoding="utf-8"))
cf = json.load(open(PFQ, encoding="utf-8"))["content_features"]

rows = []
for key, r in s.items():
    if "nano" not in r or "imdn" not in r: continue
    seg = int(key.split("_")[1])
    mot = float(np.mean(cf[str(seg)]["motion"])) if str(seg) in cf else np.nan
    tex = float(np.mean(cf[str(seg)]["texture"])) if str(seg) in cf else np.nan
    rows.append(dict(seg=seg, base=r["json_base"], imdn=r["imdn"], nano=r["nano"],
                     d=r["nano_minus_imdn"], motion=mot, texture=tex))
rows.sort(key=lambda x: x["seg"])

d = np.array([x["d"] for x in rows])
print(f"段数={len(rows)}")
print(f"IMDN-high 均值={np.mean([x['imdn'] for x in rows]):.3f}  NanoVSR 均值={np.mean([x['nano'] for x in rows]):.3f}  "
      f"base 均值={np.mean([x['base'] for x in rows]):.3f}")
print(f"Nano-IMDN 差值: 均值={d.mean():.3f}  中位={np.median(d):.3f}  min={d.min():.2f} max={d.max():.2f}  std={d.std():.2f}")
win = (d > 0.3).sum(); lose = (d < -0.3).sum(); tie = len(d) - win - lose
print(f"Nano 赢(>+0.3)={win}  输(<-0.3)={lose}  近平={tie}  => 非单调胜率={win/len(d)*100:.0f}%")
oracle = np.mean([max(x["imdn"], x["nano"]) for x in rows])
imdn_m = np.mean([x["imdn"] for x in rows])
print(f"异构 Oracle(逐段选优)={oracle:.3f}  vs 纯IMDN={imdn_m:.3f}  选择上限 Δ={oracle-imdn_m:.3f}")

def spearman(a, b):
    from scipy.stats import spearmanr
    m = ~(np.isnan(a) | np.isnan(b))
    return spearmanr(a[m], b[m]).statistic
mot = np.array([x["motion"] for x in rows]); tex = np.array([x["texture"] for x in rows])
try:
    print(f"Spearman 相关性: Δ(VSR-ISR)~motion={spearman(d,mot):.3f}  Δ~texture={spearman(d,tex):.3f}")
except Exception as e:
    print("spearman skipped:", e)

print("\nseg  base   IMDN   Nano   Δ(N-I)  motion texture")
for x in rows:
    print(f"{x['seg']:3d} {x['base']:6.2f} {x['imdn']:6.2f} {x['nano']:6.2f} {x['d']:+7.2f}  {x['motion']:6.2f} {x['texture']:9.1f}")
