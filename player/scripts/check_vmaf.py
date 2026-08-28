import json
import glob
import os

files = sorted(glob.glob('output/data/*.json'), key=lambda f: os.path.getmtime(f), reverse=True)
for f in files[:4]:
    with open(f) as fp:
        data = json.load(fp)
    segs = data.get('segments', [])
    has_vmaf = sum(1 for s in segs if s.get('real_vmaf') is not None)
    fname = os.path.basename(f)[:55]
    print(f"{fname}: {len(segs)} segs, {has_vmaf} with real_vmaf")
    if has_vmaf > 0:
        vmafs = [s['real_vmaf'] for s in segs if s.get('real_vmaf') is not None]
        avg_field = data.get('avg_vmaf')
        computed = sum(vmafs) / len(vmafs)
        print(f"  avg_vmaf field={avg_field:.2f}, computed={computed:.2f}, match={abs(avg_field-computed)<0.01}")
