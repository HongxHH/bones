import json, glob, os

bones_file = sorted(glob.glob('output/data/bones_*.json'), key=os.path.getmtime)[-1]
cares_file = sorted(glob.glob('output/data/ca_bones_*.json'), key=os.path.getmtime)[-1]

bones = json.load(open(bones_file))
cares = json.load(open(cares_file))

print(f"BONES: {os.path.basename(bones_file)}")
print(f"CARES: {os.path.basename(cares_file)}")
print()

print("=" * 80)
print(f"{'Metric':<30} {'BONES':>12} {'CARES':>12} {'Difference':>12}")
print("=" * 80)

metrics = [
    ('avg_vmaf', 'VMAF'),
    ('QOE', 'QoE'),
    ('num_stall', 'Stall count'),
    ('dur_stall', 'Stall duration (s)'),
    ('avg_freeze', 'Avg freeze (s)'),
    ('num_quality_switches', 'Quality switches'),
    ('oscillation_vmaf', 'VMAF oscillation'),
    ('enhancer_abort_total', 'Abort count'),
]

for key, label in metrics:
    b = bones.get(key, 0) or 0
    c = cares.get(key, 0) or 0
    d = c - b
    pct = (d / b * 100) if b != 0 else 0
    print(f"{label:<30} {b:>12.2f} {c:>12.2f} {d:>+10.2f} ({pct:+.1f}%)")

print()
print("=== GPU Metrics ===")
gpu_b = bones.get('gpu_metrics', {})
gpu_c = cares.get('gpu_metrics', {})
for key in ['avg_power_w', 'peak_power_w', 'total_energy_kj', 'avg_temp_c', 'peak_temp_c', 'duration_seconds']:
    b = gpu_b.get(key, 0) or 0
    c = gpu_c.get(key, 0) or 0
    d = c - b
    pct = (d / b * 100) if b != 0 else 0
    print(f"{key:<30} {b:>12.2f} {c:>12.2f} {d:>+10.2f} ({pct:+.1f}%)")

print()
print("=== Enhance Level Distribution ===")
print(f"  BONES: {bones.get('enhance_level_count')}")
print(f"  CARES: {cares.get('enhance_level_count')}")

print()
print("=== Enhance Level Stats ===")
for level in ['0', '1', '2', '3']:
    bs = bones.get('enhance_level_stats', {}).get(level, {})
    cs = cares.get('enhance_level_stats', {}).get(level, {})
    if bs or cs:
        b_lat = bs.get('avg_latency', 0) or 0
        c_lat = cs.get('avg_latency', 0) or 0
        b_cnt = bs.get('count', 0) or 0
        c_cnt = cs.get('count', 0) or 0
        print(f"  Level {level}: BONES count={b_cnt}, avg_lat={b_lat:.3f}s | CARES count={c_cnt}, avg_lat={c_lat:.3f}s")

# 计算总增强时间
print()
print("=== Total Enhancement Time ===")
b_time = 0
c_time = 0
for level in ['1', '2', '3']:
    bs = bones.get('enhance_level_stats', {}).get(level, {})
    cs = cares.get('enhance_level_stats', {}).get(level, {})
    b_completed = bs.get('completed', 0) or 0
    c_completed = cs.get('completed', 0) or 0
    b_lat = bs.get('avg_latency', 0) or 0
    c_lat = cs.get('avg_latency', 0) or 0
    b_time += b_completed * b_lat
    c_time += c_completed * c_lat
print(f"  BONES total enhance time: {b_time:.1f}s")
print(f"  CARES total enhance time: {c_time:.1f}s")
print(f"  Savings: {b_time - c_time:.1f}s ({100*(b_time-c_time)/b_time:.1f}%)")

# 资源效率指标
print()
print("=== Resource Efficiency ===")
b_vmaf = bones.get('avg_vmaf', 0) or 0
c_vmaf = cares.get('avg_vmaf', 0) or 0
b_energy = gpu_b.get('total_energy_kj', 0) or 0
c_energy = gpu_c.get('total_energy_kj', 0) or 0
print(f"  BONES: VMAF/energy = {b_vmaf/b_energy:.2f} VMAF/kJ")
print(f"  CARES: VMAF/energy = {c_vmaf/c_energy:.2f} VMAF/kJ")
print(f"  Efficiency improvement: {100*(c_vmaf/c_energy - b_vmaf/b_energy)/(b_vmaf/b_energy):.1f}%")
