import json
import os
import glob
import numpy as np

def analyze_trace(path):
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        return None
    bws = np.array([x.get('bandwidth_kbps', 0) for x in data])
    durs = np.array([x.get('duration_ms', 1000) for x in data])
    total_s = durs.sum() / 1000
    # Time-weighted average
    weighted_mean = np.sum(bws * durs) / np.sum(durs)
    return {
        'mean_kbps': weighted_mean,
        'std_kbps': np.std(bws),
        'min_kbps': np.min(bws),
        'max_kbps': np.max(bws),
        'duration_s': total_s,
        'n_samples': len(data)
    }

print('=== Current 3G trace ===')
r = analyze_trace('data/3Glogs/report.2010-09-13_1003CEST.json')
print(f'  mean={r["mean_kbps"]:.0f}kbps, std={r["std_kbps"]:.0f}, min={r["min_kbps"]:.0f}, max={r["max_kbps"]:.0f}, dur={r["duration_s"]:.0f}s')

print('\n=== Diverse 3G traces ===')
candidates = [
    'data/3Glogs/report.2010-09-13_1046CEST.json',
    'data/3Glogs/report.2010-09-14_1038CEST.json',
    'data/3Glogs/report.2010-09-28_1003CEST.json',
    'data/3Glogs/report.2011-02-01_0840CEST.json',
    'data/3Glogs/report.2011-02-10_1611CEST.json',
    'data/3Glogs/report.2011-04-21_1135CEST.json',
]
for t in candidates:
    if os.path.exists(t):
        r = analyze_trace(t)
        if r:
            print(f'  {os.path.basename(t)}: mean={r["mean_kbps"]:.0f}kbps, std={r["std_kbps"]:.0f}, min={r["min_kbps"]:.0f}, max={r["max_kbps"]:.0f}, dur={r["duration_s"]:.0f}s')

print('\n=== hd_fs traces ===')
for t in sorted(glob.glob('data/hd_fs/report_*.json'))[:6]:
    r = analyze_trace(t)
    if r:
        print(f'  {os.path.basename(t)}: mean={r["mean_kbps"]:.0f}kbps, std={r["std_kbps"]:.0f}, min={r["min_kbps"]:.0f}, max={r["max_kbps"]:.0f}, dur={r["duration_s"]:.0f}s')
