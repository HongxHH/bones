import json
import os
import glob

def analyze_result(path):
    with open(path) as f:
        data = json.load(f)
    segs = data.get('segments', [])
    enhance_levels = {}
    completed_by_level = {}
    for s in segs:
        lvl = s.get('enhance_action', 0)
        enhance_levels[lvl] = enhance_levels.get(lvl, 0) + 1
        if s.get('is_enhance'):
            completed_by_level[lvl] = completed_by_level.get(lvl, 0) + 1
    gpu = data.get('gpu_metrics', {})
    return {
        'file': os.path.basename(path),
        'avg_vmaf': data.get('avg_vmaf'),
        'QoE': data.get('QOE'),
        'enhance_levels': dict(sorted(enhance_levels.items())),
        'completed_by_level': dict(sorted(completed_by_level.items())),
        'peak_memory_mb': gpu.get('peak_memory_mb'),
        'model_pool_mb': gpu.get('model_pool_memory_mb'),
    }

# Find all multi-trace results (from the latest run)
results = []
for f in sorted(glob.glob('output/data/*.json'), key=lambda x: os.path.getmtime(x), reverse=True):
    basename = os.path.basename(f)
    # Only look at recent results (after multitrace run started)
    mtime = os.path.getmtime(f)
    if mtime > 1756300000:  # approx timestamp for recent run
        results.append(analyze_result(f))
    if len(results) >= 8:
        break

for r in results:
    print(f"=== {r['file'][:70]} ===")
    print(f"  avg_vmaf={r['avg_vmaf']:.2f}, QoE={r['QoE']:.2f}")
    print(f"  enhance_levels: {r['enhance_levels']}")
    print(f"  completed_by_level: {r['completed_by_level']}")
    print(f"  peak_memory={r['peak_memory_mb']}MB, model_pool={r['model_pool_mb']}MB")
    print()
