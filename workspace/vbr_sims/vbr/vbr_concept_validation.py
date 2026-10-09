import json
import numpy as np

# 加载数据
quality = json.load(open(r'E:\dev\Code\Python\bones\data\bbb\imdn_bbb_quality.json'))
features = json.load(open(r'E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json'))
video_info = json.load(open(r'E:\dev\Code\Python\bones\data\bbb\bbb_video_info.json'))

base_vmaf = np.array(quality['base_quality_vmaf'])  # shape: (4, 159)
enhanced_vmaf = np.array(quality['enhanced_quality_vmaf'])  # shape: (3, 4, 159)
motion = np.array([f['motion_intensity'] for f in features['features']])
texture = np.array([f['texture_complexity'] for f in features['features']])
sizes = np.array(video_info['segment_sizes_bits'])  # shape: (159, 5)

print('=== 数据形状 ===')
print(f'base_vmaf: {base_vmaf.shape} (分辨率 x 段)')
print(f'enhanced_vmaf: {enhanced_vmaf.shape} (增强级别 x 分辨率 x 段)')
print(f'motion: {motion.shape}')
print(f'texture: {texture.shape}')
print(f'sizes: {sizes.shape} (段 x 码率档)')
print(f'bitrates_kbps: {video_info["bitrates_kbps"]}')

print('\n=== 基础VMAF统计 (每分辨率平均) ===')
res_names = ['240p', '360p', '480p', '720p']
for i, name in enumerate(res_names):
    print(f'  {name}: mean={base_vmaf[i].mean():.2f}, min={base_vmaf[i].min():.2f}, max={base_vmaf[i].max():.2f}')

print('\n=== 1080p基础VMAF估计 ===')
# 1080p没有在base_quality中，可能是参考质量
# 检查enhanced_vmaf中是否有1080p
print(f'enhanced_vmaf shape: {enhanced_vmaf.shape}')
for level in range(3):
    for res in range(4):
        print(f'  level{level} res{res}: mean={enhanced_vmaf[level, res].mean():.2f}')

print('\n=== 码率大小统计 (每档平均) ===')
for i, br in enumerate(video_info['bitrates_kbps']):
    actual_br = sizes[:, i].mean() * 8 / 4 / 1000  # kbps
    print(f'  档{i} ({br}kbps目标): 实际平均={actual_br:.0f}kbps, 段大小mean={sizes[:, i].mean()/8/1024:.1f}KB')

print('\n=== 内容特征与VMAF的相关性 ===')
for i, name in enumerate(res_names):
    corr_motion = np.corrcoef(motion, base_vmaf[i])[0, 1]
    corr_texture = np.corrcoef(texture, base_vmaf[i])[0, 1]
    print(f'  {name}: motion相关={corr_motion:.3f}, texture相关={corr_texture:.3f}')

print('\n=== 概念验证：内容感知码率分配 ===')
# 目标：在相同平均码率下，给高运动段高码率，低运动段低码率，是否能提升平均VMAF？
# 方法：
# 1. 选择一个目标平均码率（比如档2=1203kbps）
# 2. CBR方案：所有段都用档2
# 3. VBR方案：高运动段用档3，低运动段用档1，使平均码率等于档2
# 4. 对比两种方案的平均VMAF

target_br_idx = 2  # 档2 = 1203kbps
cbr_vmaf = base_vmaf[target_br_idx].mean()
cbr_actual_br = sizes[:, target_br_idx].mean() * 8 / 4 / 1000

# VBR方案：按运动强度排序，高运动段用高码率，低运动段用低码率
# 选择阈值：前50%高运动段用档3，后50%低运动段用档1
motion_sorted_idx = np.argsort(-motion)
high_motion_mask = np.zeros(159, dtype=bool)
high_motion_mask[motion_sorted_idx[:80]] = True  # 前80段高运动

vbr_vmaf_values = np.where(high_motion_mask, base_vmaf[3], base_vmaf[1])
vbr_actual_br = np.where(high_motion_mask, sizes[:, 3], sizes[:, 1]).mean() * 8 / 4 / 1000
vbr_vmaf = vbr_vmaf_values.mean()

print(f'\n目标码率档: {target_br_idx} ({video_info["bitrates_kbps"][target_br_idx]}kbps)')
print(f'CBR方案: 平均VMAF={cbr_vmaf:.2f}, 平均码率={cbr_actual_br:.0f}kbps')
print(f'VBR方案(高运动用档3,低运动用档1): 平均VMAF={vbr_vmaf:.2f}, 平均码率={vbr_actual_br:.0f}kbps')
print(f'VMAF变化: {vbr_vmaf - cbr_vmaf:+.2f} ({(vbr_vmaf - cbr_vmaf)/cbr_vmaf*100:+.2f}%)')
print(f'码率变化: {vbr_actual_br - cbr_actual_br:+.0f}kbps ({(vbr_actual_br - cbr_actual_br)/cbr_actual_br*100:+.2f}%)')

# 调整VBR比例，使平均码率等于CBR
print(f'\n--- 调整比例使平均码率相等 ---')
best_gain = -999
best_ratio = 0
for ratio in np.arange(0.1, 0.9, 0.05):
    n_high = int(159 * ratio)
    mask = np.zeros(159, dtype=bool)
    mask[motion_sorted_idx[:n_high]] = True
    vbr_br = np.where(mask, sizes[:, 3], sizes[:, 1]).mean() * 8 / 4 / 1000
    if abs(vbr_br - cbr_actual_br) / cbr_actual_br < 0.05:  # 码率差<5%
        vbr_v = np.where(mask, base_vmaf[3], base_vmaf[1]).mean()
        gain = vbr_v - cbr_vmaf
        if gain > best_gain:
            best_gain = gain
            best_ratio = ratio
            best_n_high = n_high

if best_ratio > 0:
    mask = np.zeros(159, dtype=bool)
    mask[motion_sorted_idx[:best_n_high]] = True
    vbr_br = np.where(mask, sizes[:, 3], sizes[:, 1]).mean() * 8 / 4 / 1000
    vbr_v = np.where(mask, base_vmaf[3], base_vmaf[1]).mean()
    print(f'最优比例: {best_ratio:.2f} ({best_n_high}段高运动用档3)')
    print(f'VBR方案: 平均VMAF={vbr_v:.2f}, 平均码率={vbr_br:.0f}kbps')
    print(f'VMAF增益: {vbr_v - cbr_vmaf:+.2f} ({(vbr_v - cbr_vmaf)/cbr_vmaf*100:+.2f}%)')
else:
    print('未找到码率差<5%的比例')

# 用纹理复杂度做同样的分析
print(f'\n--- 用纹理复杂度做VBR分配 ---')
texture_sorted_idx = np.argsort(-texture)
best_gain_t = -999
best_ratio_t = 0
for ratio in np.arange(0.1, 0.9, 0.05):
    n_high = int(159 * ratio)
    mask = np.zeros(159, dtype=bool)
    mask[texture_sorted_idx[:n_high]] = True
    vbr_br = np.where(mask, sizes[:, 3], sizes[:, 1]).mean() * 8 / 4 / 1000
    if abs(vbr_br - cbr_actual_br) / cbr_actual_br < 0.05:
        vbr_v = np.where(mask, base_vmaf[3], base_vmaf[1]).mean()
        gain = vbr_v - cbr_vmaf
        if gain > best_gain_t:
            best_gain_t = gain
            best_ratio_t = ratio
            best_n_high_t = n_high

if best_ratio_t > 0:
    mask = np.zeros(159, dtype=bool)
    mask[texture_sorted_idx[:best_n_high_t]] = True
    vbr_br = np.where(mask, sizes[:, 3], sizes[:, 1]).mean() * 8 / 4 / 1000
    vbr_v = np.where(mask, base_vmaf[3], base_vmaf[1]).mean()
    print(f'最优比例: {best_ratio_t:.2f} ({best_n_high_t}段高纹理用档3)')
    print(f'VBR方案: 平均VMAF={vbr_v:.2f}, 平均码率={vbr_br:.0f}kbps')
    print(f'VMAF增益: {vbr_v - cbr_vmaf:+.2f} ({(vbr_v - cbr_vmaf)/cbr_vmaf*100:+.2f}%)')
else:
    print('未找到码率差<5%的比例')
