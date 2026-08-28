import json
import glob
import os

files_by_type = {}
for f in glob.glob('output/data/*.json'):
    basename = os.path.basename(f)
    if basename.startswith('conservative'):
        key = 'conservative_ca'
    elif basename.startswith('bones_imdn') and 'content' not in basename and 'drl' not in basename:
        key = 'bones'
    elif basename.startswith('content_aware'):
        key = 'content_aware'
    elif basename.startswith('drl'):
        key = 'drl'
    else:
        continue
    if key not in files_by_type or os.path.getmtime(f) > os.path.getmtime(files_by_type[key]):
        files_by_type[key] = f

for key in ['bones', 'content_aware', 'conservative_ca', 'drl']:
    f = files_by_type.get(key)
    if not f:
        print(f"=== {key.upper()} === NOT FOUND")
        continue
    with open(f) as fp:
        data = json.load(fp)
    segs = data.get('segments', [])
    has_vmaf = sum(1 for s in segs if s.get('real_vmaf') is not None)
    enhance_levels = {}
    for s in segs:
        lvl = s.get('enhance_action', 0)
        enhance_levels[lvl] = enhance_levels.get(lvl, 0) + 1
    completed = sum(1 for s in segs if s.get('is_enhance'))
    gpu = data.get('gpu_metrics', {})
    print(f"=== {key.upper()} ===")
    print(f"  Segments: {len(segs)}, with real_vmaf: {has_vmaf}")
    print(f"  avg_vmaf: {data.get('avg_vmaf', 'N/A'):.2f}")
    print(f"  QoE: {data.get('QOE', 'N/A'):.2f}")
    print(f"  num_stall: {data.get('num_stall', 'N/A')}")
    print(f"  dur_stall: {data.get('dur_stall', 'N/A'):.1f}s")
    print(f"  quality_switches: {data.get('num_quality_switches', 'N/A')}")
    print(f"  enhancer_aborts: {data.get('enhancer_abort_total', 'N/A')}")
    print(f"  enhance_levels: {dict(sorted(enhance_levels.items()))}")
    print(f"  completed: {completed}/{len(segs)} ({completed/len(segs)*100:.1f}%)")
    print(f"  gpu_power: {gpu.get('avg_power_w', 'N/A')}W, gpu_mem: {gpu.get('peak_memory_mb', 'N/A')}MB")
    print()
