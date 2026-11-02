import json, glob, os

bones_files = sorted(glob.glob('output/data/bones_*.json'), key=os.path.getmtime)
ca_files = sorted(glob.glob('output/data/ca_bones_*.json'), key=os.path.getmtime)

# BONES baseline (latest)
bones = json.load(open(bones_files[-1]))
# CA-predicted (latest)
ca_pred = json.load(open(ca_files[-1]))

# Previous CA results (find by timestamp)
# Layer1: 093254 (BONES), 094523 (CA base_known layer1)
# Layer2: 100458 (CA layer2)
ca_l1 = None
ca_l2 = None
for f in ca_files:
    if '094523' in f:
        ca_l1 = json.load(open(f))
    elif '100458' in f:
        ca_l2 = json.load(open(f))

out = []
out.append(f"BONES baseline: {os.path.basename(bones_files[-1])}")
if ca_l1: out.append(f"CA base_known(L1): {os.path.basename([f for f in ca_files if '094523' in f][0])}")
if ca_l2: out.append(f"CA layer2: {os.path.basename([f for f in ca_files if '100458' in f][0])}")
out.append(f"CA predicted: {os.path.basename(ca_files[-1])}")
out.append("")

keys = ['avg_vmaf','QOE','num_stall','dur_stall','avg_freeze','num_quality_switches','oscillation_vmaf','enhancer_abort_total']
out.append(f"{'Metric':<25} {'BONES':>10} {'CA_L1':>10} {'CA_L2':>10} {'CA_PRED':>10} {'PRED-B':>10}")
out.append("-" * 80)
for k in keys:
    b = bones.get(k,0) or 0
    c1 = ca_l1.get(k,0) if ca_l1 else 0
    c2 = ca_l2.get(k,0) if ca_l2 else 0
    cp = ca_pred.get(k,0) or 0
    d = cp - b
    out.append(f"{k:<25} {b:>10.2f} {c1:>10.2f} {c2:>10.2f} {cp:>10.2f} {d:>+10.2f}")
out.append("")
out.append("Enhance level distribution:")
out.append(f"  BONES:   {bones.get('enhance_level_count')}")
if ca_l1: out.append(f"  CA_L1:   {ca_l1.get('enhance_level_count')}")
if ca_l2: out.append(f"  CA_L2:   {ca_l2.get('enhance_level_count')}")
out.append(f"  CA_PRED: {ca_pred.get('enhance_level_count')}")

with open('pred_comparison.txt', 'w') as f:
    f.write('\n'.join(out))
print('\n'.join(out))
