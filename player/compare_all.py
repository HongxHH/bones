import json, glob, os

all_files = sorted(glob.glob('output/data/ca_bones_*.json') + glob.glob('output/data/bones_*.json'), key=os.path.getmtime)

# 按时间戳标识各配置
configs = {
    'BONES': '093254',
    'CA_L1_base_known': '094523',
    'CA_L2_per_seg_old': '100458',
    'CA_pred_per_seg': '121954',
    'CA_base_per_seg_new': '123358',
    'CA_pred_global_lat': '124550',
}

results = {}
for name, ts in configs.items():
    matching = [f for f in all_files if ts in f]
    if matching:
        results[name] = json.load(open(matching[0]))
        print(f"{name}: {os.path.basename(matching[0])}")
    else:
        print(f"{name}: NOT FOUND ({ts})")

print("\n" + "=" * 110)
print(f"{'Metric':<25} {'BONES':>8} {'CA_L1':>8} {'CA_L2':>8} {'PRED+L':>8} {'BASE+L':>8} {'PRED+G':>8}")
print("=" * 110)

keys = ['avg_vmaf','QOE','num_stall','dur_stall','avg_freeze','num_quality_switches','oscillation_vmaf','enhancer_abort_total']
for k in keys:
    vals = []
    for name in configs:
        v = results.get(name, {}).get(k, 0) or 0
        vals.append(v)
    line = f"{k:<25}"
    for v in vals:
        line += f" {v:>8.2f}"
    print(line)

print("\nEnhance level distribution:")
for name in configs:
    dist = results.get(name, {}).get('enhance_level_count', {})
    print(f"  {name:<20}: {dist}")

# dQoE vs BONES
print("\n" + "=" * 110)
print("dQoE vs BONES:")
bones_qoe = results.get('BONES', {}).get('QOE', 0)
for name in configs:
    if name == 'BONES':
        continue
    qoe = results.get(name, {}).get('QOE', 0)
    d = qoe - bones_qoe if bones_qoe else 0
    print(f"  {name:<20}: {d:+.2f}")
