import json
import glob
import os

files_by_type = {}
for f in glob.glob('output/data/*.json'):
    basename = os.path.basename(f)
    if basename.startswith('bones_imdn') and 'content' not in basename and 'drl' not in basename:
        key = 'bones'
    elif basename.startswith('content_aware'):
        key = 'content_aware'
    elif basename.startswith('drl'):
        key = 'drl'
    else:
        continue
    if key not in files_by_type or os.path.getmtime(f) > os.path.getmtime(files_by_type[key]):
        files_by_type[key] = f

for key in ['bones', 'content_aware', 'drl']:
    f = files_by_type.get(key)
    if not f:
        continue
    with open(f) as fp:
        data = json.load(fp)
    segs = data.get('segments', [])
    
    # Bitrate distribution
    bitrates = {}
    for s in segs:
        br = s.get('bitrate', 0)
        bitrates[br] = bitrates.get(br, 0) + 1
    
    # Quality level distribution (repr_id - min_repr)
    qualities = {}
    for s in segs:
        q = s.get('quality', 0)
        qualities[q] = qualities.get(q, 0) + 1
    
    # VMAF by enhancement completion
    completed_vmafs = [s['real_vmaf'] for s in segs if s.get('is_enhance') and s.get('real_vmaf') is not None]
    aborted_vmafs = [s['real_vmaf'] for s in segs if not s.get('is_enhance') and s.get('real_vmaf') is not None]
    
    print(f"=== {key.upper()} ===")
    print(f"  Bitrate distribution: {dict(sorted(bitrates.items()))}")
    print(f"  Quality level dist: {dict(sorted(qualities.items()))}")
    print(f"  Completed segments: {len(completed_vmafs)}, avg_vmaf={sum(completed_vmafs)/len(completed_vmafs):.2f}" if completed_vmafs else "  No completed")
    print(f"  Aborted segments: {len(aborted_vmafs)}, avg_vmaf={sum(aborted_vmafs)/len(aborted_vmafs):.2f}" if aborted_vmafs else "  No aborted")
    print()
