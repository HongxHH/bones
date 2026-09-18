# -*- coding: utf-8 -*-
import glob, json, os, datetime
d = r"E:\dev\Code\Python\bones\player\output\data"
files = sorted(glob.glob(os.path.join(d, "bones_*.json")), key=os.path.getmtime)
print(f"{'file':<72} {'vmaf':>7} {'QoE':>7} {'stall#':>7} {'stalls':>7} {'fcSeg':>6}")
for f in files[-8:]:
    j = json.load(open(f, encoding="utf-8"))
    ts = os.path.basename(f)
    # 出口帧分布（PACNet 专有）
    tag = ""
    segs = j.get("segments", [])
    tot = [0,0,0]
    for s in segs:
        efc = s.get("exit_frame_counts")
        if efc:
            for i in range(3): tot[i] += int(efc[i] or 0)
    if sum(tot) > 0:
        tag = " e1/e2/e3=%.0f/%.0f/%.0f%%" % (100*tot[0]/sum(tot), 100*tot[1]/sum(tot), 100*tot[2]/sum(tot))
    elc = j.get("enhance_level_count")
    print(f"{ts[-60:]:<72} {j.get('avg_vmaf',0):7.2f} {j.get('QOE',0):7.2f} "
          f"{str(j.get('num_stall')):>7} {j.get('dur_stall',0):6.2f}s {str(j.get('fast_complete_segments_count')):>6}{tag}")
    if elc: print(f"      enhance_level_count={elc}")
