import json, glob, os

files = sorted(glob.glob('output/data/ca_bones_*.json'), key=os.path.getmtime)
print("CA-BONES files:", [os.path.basename(f) for f in files])

ca_conservative = json.load(open(files[-1]))
ca_aggressive = json.load(open(files[-2]))
bones = json.load(open('output/data/bones_imdn_always_fast-True_video-bbb_trace-report.2010-09-13_1003CEST_20260828-084148.json'))

keys = ['avg_vmaf','QOE','num_stall','dur_stall','avg_freeze','num_quality_switches','oscillation_vmaf','enhancer_abort_total']

print(f"\n{'Metric':<25} {'BONES':>10} {'CA_aggr':>10} {'CA_cons':>10}")
print("-" * 60)
for k in keys:
    b = bones.get(k, 0)
    a = ca_aggressive.get(k, 0)
    c = ca_conservative.get(k, 0)
    if isinstance(b, float):
        print(f"{k:<25} {b:>10.2f} {a:>10.2f} {c:>10.2f}")
    else:
        print(f"{k:<25} {b:>10} {a:>10} {c:>10}")

print(f"\nEnhance level distribution:")
print(f"  BONES:          {bones.get('enhance_level_count')}")
print(f"  CA_aggressive:  {ca_aggressive.get('enhance_level_count')}")
print(f"  CA_conservative: {ca_conservative.get('enhance_level_count')}")

# Save to file
out = {}
for k in keys:
    out[k] = {'BONES': bones.get(k), 'CA_aggressive': ca_aggressive.get(k), 'CA_conservative': ca_conservative.get(k)}
out['enhance_level_count'] = {'BONES': bones.get('enhance_level_count'), 'CA_aggressive': ca_aggressive.get('enhance_level_count'), 'CA_conservative': ca_conservative.get('enhance_level_count')}
with open('workspace_comparison.json', 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to workspace_comparison.json")
