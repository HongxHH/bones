"""
更新multitask数据文件：用实测Unsharp Mask数据替换NAFNet估算值
更新内容：
1. deblur_model名称
2. enhancement_fps（deblur相关组合）
3. enhanced_quality_vmaf（sr_deblur, deblur_only组合）
4. metadata备注
"""
import json
import numpy as np

DATA_FILE = r'E:\dev\Code\Python\bones\data\bbb\multitask_bbb_rtx5080.json'
OUTPUT_FILE = DATA_FILE  # 原地更新

# 实测USM数据（5段平均）
USM_DATA = {
    '360p': {'vmaf_gain': 14.68, 'latency_ms': 0.325, 'fps': 3077.0},
    '480p': {'vmaf_gain': 6.80,  'latency_ms': 0.830, 'fps': 1205.0},
    '720p': {'vmaf_gain': -1.55, 'latency_ms': 1.503, 'fps': 665.0},
    '1080p': {'vmaf_gain': -5.06, 'latency_ms': 3.264, 'fps': 306.0},
}
RES_ORDER = ['360p', '480p', '720p', '1080p']

# Combo索引: 0=sr_only, 1=sr_deblur, 2=deblur_only, 3=sr_deblur_fi

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

print("=== 更新前 ===")
print(f"deblur_model: {data['_metadata']['deblur_model']}")
print(f"enhancement_fps:")
for i, row in enumerate(data['enhancement_fps']):
    print(f"  {RES_ORDER[i]}: sr_only={row[0]:.1f}, sr_deblur={row[1]:.1f}, deblur_only={row[2]:.1f}, sr_deblur_fi={row[3]:.1f}")

# === 1. 更新deblur_model ===
data['_metadata']['deblur_model'] = 'UnsharpMask (amount=1.5, radius=3)'
data['_metadata']['note'] = (
    'Deblur=UnsharpMask runs at input resolution before SR. '
    'VMAF boosts are measured averages over 5 sampled segments (content-dependent). '
    'USM latency measured on RTX 5080 via OpenCV CPU.'
)

# === 2. 更新enhancement_fps ===
for i, res in enumerate(RES_ORDER):
    usm = USM_DATA[res]
    sr_only_fps = data['enhancement_fps'][i][0]
    sr_time_ms = 1000.0 / sr_only_fps
    usm_time_ms = usm['latency_ms']

    # sr_deblur = SR + USM
    sr_deblur_time = sr_time_ms + usm_time_ms
    sr_deblur_fps = 1000.0 / sr_deblur_time

    # deblur_only = USM only
    deblur_only_fps = usm['fps']

    # sr_deblur_fi = SR + USM + FI (FI时间从原数据估算)
    orig_sr_deblur_fi_time = 1000.0 / data['enhancement_fps'][i][3]
    orig_sr_deblur_time = 1000.0 / data['enhancement_fps'][i][1]
    fi_time_ms = orig_sr_deblur_fi_time - orig_sr_deblur_time
    new_sr_deblur_fi_time = sr_deblur_time + fi_time_ms
    sr_deblur_fi_fps = 1000.0 / new_sr_deblur_fi_time

    data['enhancement_fps'][i] = [
        round(sr_only_fps, 6),
        round(sr_deblur_fps, 6),
        round(deblur_only_fps, 6),
        round(sr_deblur_fi_fps, 6),
    ]

print("\n=== 更新后 enhancement_fps ===")
for i, row in enumerate(data['enhancement_fps']):
    print(f"  {RES_ORDER[i]}: sr_only={row[0]:.1f}, sr_deblur={row[1]:.1f}, deblur_only={row[2]:.1f}, sr_deblur_fi={row[3]:.1f}")

# === 3. 更新enhanced_quality_vmaf ===
# combo[1]=sr_deblur: sr_only VMAF + USM gain (近似)
# combo[2]=deblur_only: base VMAF + USM gain
for i, res in enumerate(RES_ORDER):
    usm_gain = USM_DATA[res]['vmaf_gain']
    num_segs = len(data['base_quality_vmaf'][i])

    # deblur_only = base + usm_gain
    for seg_idx in range(num_segs):
        base_vmaf = data['base_quality_vmaf'][i][seg_idx]
        data['enhanced_quality_vmaf'][i][2][seg_idx] = round(base_vmaf + usm_gain, 6)

    # sr_deblur = sr_only + usm_gain (近似：USM在SR之前，增益近似叠加)
    for seg_idx in range(num_segs):
        sr_only_vmaf = data['enhanced_quality_vmaf'][i][0][seg_idx]
        data['enhanced_quality_vmaf'][i][1][seg_idx] = round(sr_only_vmaf + usm_gain, 6)

    # sr_deblur_fi: 保持原FI增益不变（sr_deblur_fi = sr_deblur + FI_gain）
    # 原FI_gain = orig_sr_deblur_fi - orig_sr_deblur
    # 新sr_deblur_fi = 新sr_deblur + 原FI_gain
    for seg_idx in range(num_segs):
        orig_sr_deblur = data['enhanced_quality_vmaf'][i][1][seg_idx] - usm_gain  # 恢复原值
        orig_sr_deblur_fi = data['enhanced_quality_vmaf'][i][3][seg_idx]
        fi_gain = orig_sr_deblur_fi - orig_sr_deblur
        new_sr_deblur = data['enhanced_quality_vmaf'][i][1][seg_idx]
        data['enhanced_quality_vmaf'][i][3][seg_idx] = round(new_sr_deblur + fi_gain, 6)

print("\n=== 更新后 enhanced_quality_vmaf (平均) ===")
for i, res in enumerate(RES_ORDER):
    usm_gain = USM_DATA[res]['vmaf_gain']
    avg_base = np.mean(data['base_quality_vmaf'][i])
    avg_sr_only = np.mean(data['enhanced_quality_vmaf'][i][0])
    avg_sr_deblur = np.mean(data['enhanced_quality_vmaf'][i][1])
    avg_deblur_only = np.mean(data['enhanced_quality_vmaf'][i][2])
    avg_sr_deblur_fi = np.mean(data['enhanced_quality_vmaf'][i][3])
    print(f"  {res}: base={avg_base:.2f}, sr_only={avg_sr_only:.2f}, "
          f"sr_deblur={avg_sr_deblur:.2f}(+{usm_gain:+.2f}), "
          f"deblur_only={avg_deblur_only:.2f}(+{usm_gain:+.2f}), "
          f"sr_deblur_fi={avg_sr_deblur_fi:.2f}")

# === 保存 ===
with open(OUTPUT_FILE, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n已保存到: {OUTPUT_FILE}")
print("更新完成！")
