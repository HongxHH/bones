# -*- coding: utf-8 -*-
import glob, json, os
d = r"E:\dev\Code\Python\bones\player\output\data"
f = sorted(glob.glob(os.path.join(d, "bones_pacnet*.json")), key=os.path.getmtime)[-1]
j = json.load(open(f, encoding="utf-8"))
segs = j["segments"]
print("segment keys:", list(segs[0].keys()))
print()
# 找日志里降档的段：seg2(dist84/4/8) seg3(91/3/2) seg35(0/0/96) seg183(12/11/73)
for tgt in [1, 2, 34, 182]:
    s = segs[tgt]
    idx = s.get("index", s.get("segment_index", "?"))
    print(f"seg idx={idx} q(download)={s.get('quality')} enhance_action={s.get('enhance_action')} "
          f"is_enhance={s.get('is_enhance')} fast_complete={s.get('fast_complete')}")
    print(f"   exit_frame_counts={s.get('exit_frame_counts')} enhanced_cnt={s.get('enhanced_cnt')} "
          f"interpolated={s.get('interpolated_cnt')}")
    print(f"   base_vmaf={s.get('base_vmaf')} enhance_vmaf={s.get('enhance_vmaf')} "
          f"real_vmaf={s.get('real_vmaf')} vmaf={s.get('vmaf')}")
    print()
# 汇总出口帧占比
tot = [0,0,0]
fc = 0
for s in segs:
    efc = s.get("exit_frame_counts")
    if efc:
        for i in range(3): tot[i] += int(efc[i] or 0)
    if s.get("fast_complete"): fc += 1
print("total frames by exit e1/e2/e3:", tot, "sum=", sum(tot))
print("e3 fraction=%.1f%%  e1 fraction=%.1f%%" % (100*tot[2]/sum(tot), 100*tot[0]/sum(tot)))
print("fast_complete segs:", fc)
print("avg_vmaf field:", j.get("avg_vmaf"), " QOE:", j.get("QOE"), " stall:", j.get("num_stall"), j.get("dur_stall"))
