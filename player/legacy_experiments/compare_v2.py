import json, glob, os

bones_files = sorted(glob.glob('output/data/bones_*.json'), key=os.path.getmtime)
ca_files = sorted(glob.glob('output/data/ca_bones_*.json'), key=os.path.getmtime)
bones_new = json.load(open(bones_files[-1]))
bones_old = json.load(open(bones_files[-2]))
ca_new = json.load(open(ca_files[-1]))
ca_old = json.load(open(ca_files[-3]))

out = []
out.append(f"BONES_new: {os.path.basename(bones_files[-1])}")
out.append(f"CA-BONES_new: {os.path.basename(ca_files[-1])}")
out.append("")
keys = ['avg_vmaf','QOE','num_stall','dur_stall','avg_freeze','num_quality_switches','oscillation_vmaf','enhancer_abort_total']
out.append(f"{'Metric':<25} {'BONES_old':>10} {'BONES_new':>10} {'CA_old':>10} {'CA_new':>10}")
out.append("-" * 70)
for k in keys:
    out.append(f"{k:<25} {bones_old.get(k,0):>10.2f} {bones_new.get(k,0):>10.2f} {ca_old.get(k,0):>10.2f} {ca_new.get(k,0):>10.2f}")
out.append("")
out.append("Enhance level distribution:")
out.append(f"  BONES_old: {bones_old.get('enhance_level_count')}")
out.append(f"  BONES_new: {bones_new.get('enhance_level_count')}")
out.append(f"  CA_old:    {ca_old.get('enhance_level_count')}")
out.append(f"  CA_new:    {ca_new.get('enhance_level_count')}")

with open('comparison_result.txt', 'w') as f:
    f.write('\n'.join(out))
print("Done")
