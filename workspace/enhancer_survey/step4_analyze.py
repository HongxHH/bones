"""阶段17闭环判决：leave-segments-out 对症微调(LO) 在16个留出段 vs IMDN-DIV2K/BBB/NanoVSR。"""
import json, os, sys
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
SD = os.path.join(ROOT, "workspace/enhancer_survey")
lo = json.load(open(os.path.join(SD, sys.argv[1] if len(sys.argv)>1 else "step4_leaveout16.json"), encoding="utf-8"))
bb = json.load(open(os.path.join(SD, "step2_survey16.json"), encoding="utf-8"))

rows=[]
for k in lo:
    seg=int(k.split("_")[1]); r=lo[k]
    if "imdn_lo" not in r or r.get("imdn_lo") is None: continue
    b=bb[k]
    rows.append(dict(seg=seg, base=r["json_base"], div=r["imdn_div"], lo=r["imdn_lo"],
                     bbb=b["imdn"], nano=b["nano"]))
rows.sort(key=lambda x:x["seg"])
m=lambda f: np.mean([f(x) for x in rows])
print(f"留出段 n={len(rows)}（这些段模型训练时从未见过）\n")
print(f"base(bicubic)          {m(lambda x:x['base']):.2f}")
print(f"IMDN-DIV2K 泛化起点     {m(lambda x:x['div']):.2f}")
print(f"IMDN-LO 对症微调(留出)  {m(lambda x:x['lo']):.2f}")
print(f"IMDN-BBB 泄漏上界       {m(lambda x:x['bbb']):.2f}")
print(f"NanoVSR               {m(lambda x:x['nano']):.2f}")
d_ld=np.array([x['lo']-x['div'] for x in rows]); d_lb=np.array([x['lo']-x['bbb'] for x in rows])
print(f"\nLO-DIV(跨段泛化增益): 均值{d_ld.mean():+.2f} 中位{np.median(d_ld):+.2f} 范围[{d_ld.min():+.1f},{d_ld.max():+.1f}] 赢{(d_ld>0.3).sum()}/输{(d_ld<-0.3).sum()}")
print(f"LO-BBB(距泄漏上界):   均值{d_lb.mean():+.2f}（越接近0越说明泛化能追上整片微调）")
print(f"\nseg  base   DIV    LO     BBB   Nano | LO-DIV LO-BBB")
for x in rows:
    print(f"{x['seg']:3d} {x['base']:6.2f} {x['div']:6.2f} {x['lo']:6.2f} {x['bbb']:6.2f} {x['nano']:6.2f} | {x['lo']-x['div']:+6.2f} {x['lo']-x['bbb']:+6.2f}")
