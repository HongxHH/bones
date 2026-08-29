"""
修复版：更新multitask数据文件
修复：1) FI增益计算（先读原值再修改） 2) VMAF裁剪到[0,100]
"""
import json
import numpy as np
import shutil

DATA_FILE = r'E:\dev\Code\Python\bones\data\bbb\multitask_bbb_rtx5080.json'
BACKUP_FILE = DATA_FILE + '.bak'

# 备份
shutil.copy2(DATA_FILE, BACKUP_FILE)
print(f"已备份到: {BACKUP_FILE}")

# 实测USM数据
USM_DATA = {
    '360p': {'vmaf_gain': 14.68, 'latency_ms': 0.325, 'fps': 3077.0},
    '480p': {'vmaf_gain': 6.80,  'latency_ms': 0.830, 'fps': 1205.0},
    '720p': {'vmaf_gain': -1.55, 'latency_ms': 1.503, 'fps': 665.0},
    '1080p': {'vmaf_gain': -5.06, 'latency_ms': 3.264, 'fps': 306.0},
}
RES_ORDER = ['360p', '480p', '720p', '1080p']

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

# === 1. 更新metadata ===
data['_metadata']['deblur_model'] = 'UnsharpMask (amount=1.5, radius=3)'
data['_metadata']['note'] = (
    'Deblur=UnsharpMask runs at input resolution before SR. '
    'VMAF boosts are measured averages over 5 sampled segments (content-dependent). '
    'USM latency measured on RTX 5080 via OpenCV CPU. VMAF clipped to [0,100].'
)

# === 2. 更新enhancement_fps ===
for i, res in enumerate(RES_ORDER):
    usm = USM_DATA[res]
    sr_only_fps = data['enhancement_fps'][i][0]
    sr_time_ms = 1000.0 / sr_only_fps
    usm_time_ms = usm['latency_ms']

    sr_deblur_time = sr_time_ms + usm_time_ms
    sr_deblur_fps = 1000.0 / sr_deblur_time
    deblur_only_fps = usm['fps']

    # FI时间从原数据估算
    orig_sr_deblur_fi_time = 1000.0 / data['enhancement_fps'][i][3]
    orig_sr_deblur_time = 1000.0 / data['enhancement_fps'][i][1]
    fi_time_ms = max(0, orig_sr_deblur_fi_time - orig_sr_deblur_time)
    new_sr_deblur_fi_time = sr_deblur_time + fi_time_ms
    sr_deblur_fi_fps = 1000.0 / new_sr_deblur_fi_time

    data['enhancement_fps'][i] = [
        round(sr_only_fps, 6),
        round(sr_deblur_fps, 6),
        round(deblur_only_fps, 6),
        round(sr_deblur_fi_fps, 6),
    ]

# === 3. 更新enhanced_quality_vmaf ===
# 先读取原始sr_deblur_fi和sr_deblur，计算FI增益
for i, res in enumerate(RES_ORDER):
    usm_gain = USM_DATA[res]['vmaf_gain']
    num_segs = len(data['base_quality_vmaf'][i])

    # 先计算FI增益（基于原始值）
    fi_gains = []
    for seg_idx in range(num_segs):
        orig_sr_deblur = data['enhanced_quality_vmaf'][i][1][seg_idx]
        orig_sr_deblur_fi = data['enhanced_quality_vmaf'][i][3][seg_idx]
        fi_gains.append(orig_sr_deblur_fi - orig_sr_deblur)

    # 更新deblur_only = base + usm_gain
    for seg_idx in range(num_segs):
        base_vmaf = data['base_quality_vmaf'][i][seg_idx]
        val = max(0, min(100, base_vmaf + usm_gain))
        data['enhanced_quality_vmaf'][i][2][seg_idx] = round(val, 6)

    # 更新sr_deblur = sr_only + usm_gain
    for seg_idx in range(num_segs):
        sr_only_vmaf = data['enhanced_quality_vmaf'][i][0][seg_idx]
        val = max(0, min(100, sr_only_vmaf + usm_gain))
        data['enhanced_quality_vmaf'][i][1][seg_idx] = round(val, 6)

    # 更新sr_deblur_fi = 新sr_deblur + fi_gain
    for seg_idx in range(num_segs):
        new_sr_deblur = data['enhanced_quality_vmaf'][i][1][seg_idx]
        val = max(0, min(100, new_sr_deblur + fi_gains[seg_idx]))
        data['enhanced_quality_vmaf'][i][3][seg_idx] = round(val, 6)

# === 验证输出 ===
print("\n=== 更新后汇总 ===")
print(f"{'Res':6s} {'Base':>8s} {'SR_only':>8s} {'SR+USM':>8s} {'USM_only':>8s} {'SR+USM+FI':>10s}")
print("-" * 55)
for i, res in enumerate(RES_ORDER):
    avg_base = np.mean(data['base_quality_vmaf'][i])
    avg_sr = np.mean(data['enhanced_quality_vmaf'][i][0])
    avg_sr_usm = np.mean(data['enhanced_quality_vmaf'][i][1])
    avg_usm = np.mean(data['enhanced_quality_vmaf'][i][2])
    avg_all = np.mean(data['enhanced_quality_vmaf'][i][3])
    print(f"{res:6s} {avg_base:8.2f} {avg_sr:8.2f} {avg_sr_usm:8.2f} {avg_usm:8.2f} {avg_all:10.2f}")

print(f"\n{'Res':6s} {'SR_only':>10s} {'SR+USM':>10s} {'USM_only':>10s} {'SR+USM+FI':>12s}")
print("-" * 55)
for i, res in enumerate(RES_ORDER):
    fps = data['enhancement_fps'][i]
    print(f"{res:6s} {fps[0]:10.1f} {fps[1]:10.1f} {fps[2]:10.1f} {fps[3]:12.1f}")

# 保存
with open(DATA_FILE, 'w') as f:
    json.dump(data, f, indent=2)

print(f"\n已保存到: {DATA_FILE}")
print("更新完成！")
