import json, glob, os

bones_files = sorted(glob.glob('output/data/bones_*.json'), key=os.path.getmtime)
ca_files = sorted(glob.glob('output/data/ca_bones_*.json'), key=os.path.getmtime)

# BONES with layer1 (improved latency) = latest bones
bones_l1 = json.load(open(bones_files[-1]))
# CA-BONES with layer2 = latest ca_bones
ca_l2 = json.load(open(ca_files[-1]))
# CA-BONES with layer1 (094523, before layer2)
ca_l1 = json.load(open(ca_files[-2]))

out = []
out.append(f"BONES(L1): {os.path.basename(bones_files[-1])}")
out.append(f"CA-BONES(L1): {os.path.basename(ca_files[-2])}")
out.append(f"CA-BONES(L2): {os.path.basename(ca_files[-1])}")
out.append("")
keys = ['avg_vmaf','QOE','num_stall','dur_stall','avg_freeze','num_quality_switches','oscillation_vmaf','enhancer_abort_total']
out.append(f"{'Metric':<25} {'BONES_L1':>10} {'CA_L1':>10} {'CA_L2':>10} {'L2-BONES':>10}")
out.append("-" * 70)
for k in keys:
    b = bones_l1.get(k,0) or 0
    c1 = ca_l1.get(k,0) or 0
    c2 = ca_l2.get(k,0) or 0
    d = c2 - b
    out.append(f"{k:<25} {b:>10.2f} {c1:>10.2f} {c2:>10.2f} {d:>+10.2f}")
out.append("")
out.append("Enhance level distribution:")
out.append(f"  BONES_L1: {bones_l1.get('enhance_level_count')}")
out.append(f"  CA_L1:    {ca_l1.get('enhance_level_count')}")
out.append(f"  CA_L2:    {ca_l2.get('enhance_level_count')}")

with open('final_comparison.txt', 'w') as f:
    f.write('\n'.join(out))
print("Done")
