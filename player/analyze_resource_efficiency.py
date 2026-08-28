"""
资源效率分析：识别 BONES 的低性价比增强，估算潜在节省
"""
import json
import numpy as np
import glob
import os

# 加载 BONES 结果
bones_file = sorted(glob.glob('output/data/bones_*.json'), key=os.path.getmtime)[-1]
bones = json.load(open(bones_file))
segments = bones['segments']

# 加载内容特征
features = json.load(open('data/content_features/bbb_features.json'))
feat_dict = {f['segment_index']: f for f in features}

print(f"=== BONES 资源效率分析 ===")
print(f"结果文件: {os.path.basename(bones_file)}")
print(f"总段数: {len(segments)}")
print(f"avg_vmaf: {bones['avg_vmaf']:.2f}, QOE: {bones['QOE']:.2f}")
print()

# 每段分析
seg_data = []
total_enhance_time = 0
total_vmaf_gain = 0
enhanced_segs = 0

for seg in segments:
    idx = seg['index']
    action = seg.get('enhance_action', 0)
    latency = seg.get('enhance_latency', None)
    base_vmaf = seg.get('base_vmaf', 0)
    real_vmaf = seg.get('real_vmaf', 0)
    vmaf_gain = real_vmaf - base_vmaf if (base_vmaf and real_vmaf) else 0

    feat = feat_dict.get(idx - 1, {})  # segment index is 1-based, features are 0-based
    texture = feat.get('texture_complexity', 0)
    motion = feat.get('motion_intensity', 0)

    if latency and latency > 0:
        cost_eff = vmaf_gain / latency  # VMAF points per second
    else:
        cost_eff = 0

    seg_data.append({
        'idx': idx,
        'action': action,
        'latency': latency or 0,
        'base_vmaf': base_vmaf or 0,
        'real_vmaf': real_vmaf or 0,
        'vmaf_gain': vmaf_gain,
        'cost_eff': cost_eff,
        'texture': texture,
        'motion': motion,
    })

    if action > 0 and latency:
        total_enhance_time += latency
        total_vmaf_gain += vmaf_gain
        enhanced_segs += 1

print(f"增强段数: {enhanced_segs}/{len(segments)}")
print(f"总增强时间: {total_enhance_time:.1f}s")
print(f"总 VMAF 增益: {total_vmaf_gain:.1f}")
print(f"平均性价比: {total_vmaf_gain/total_enhance_time:.2f} VMAF/s")
print()

# 按增强级别分析
print("=== 按增强级别分析 ===")
for level in range(4):
    level_segs = [s for s in seg_data if s['action'] == level]
    if not level_segs:
        continue
    avg_lat = np.mean([s['latency'] for s in level_segs if s['latency'] > 0])
    avg_gain = np.mean([s['vmaf_gain'] for s in level_segs])
    avg_ce = np.mean([s['cost_eff'] for s in level_segs if s['cost_eff'] > 0])
    avg_text = np.mean([s['texture'] for s in level_segs])
    avg_mot = np.mean([s['motion'] for s in level_segs])
    print(f"  Level {level}: {len(level_segs)} segs, avg_lat={avg_lat:.3f}s, "
          f"avg_gain={avg_gain:.2f}, avg_cost_eff={avg_ce:.2f}, "
          f"avg_texture={avg_text:.3f}, avg_motion={avg_mot:.3f}")

print()

# 识别低性价比段（cost_eff < 中位数的 50%）
enhanced = [s for s in seg_data if s['action'] > 0 and s['cost_eff'] > 0]
ce_values = [s['cost_eff'] for s in enhanced]
median_ce = np.median(ce_values)
low_ce_threshold = median_ce * 0.5
low_ce_segs = [s for s in enhanced if s['cost_eff'] < low_ce_threshold]

print(f"=== 低性价比段分析 ===")
print(f"性价比中位数: {median_ce:.2f} VMAF/s")
print(f"低性价比阈值 (<50%中位数): {low_ce_threshold:.2f}")
print(f"低性价比段数: {len(low_ce_segs)}/{len(enhanced)} ({100*len(low_ce_segs)/len(enhanced):.1f}%)")
print(f"低性价比段总增强时间: {sum(s['latency'] for s in low_ce_segs):.1f}s "
      f"({100*sum(s['latency'] for s in low_ce_segs)/total_enhance_time:.1f}% of total)")
print(f"低性价比段总 VMAF 增益: {sum(s['vmaf_gain'] for s in low_ce_segs):.1f} "
      f"({100*sum(s['vmaf_gain'] for s in low_ce_segs)/total_vmaf_gain:.1f}% of total)")
print()

# 低性价比段的内容特征
if low_ce_segs:
    print(f"低性价比段平均内容特征:")
    print(f"  texture: {np.mean([s['texture'] for s in low_ce_segs]):.3f} "
          f"(all enhanced: {np.mean([s['texture'] for s in enhanced]):.3f})")
    print(f"  motion:  {np.mean([s['motion'] for s in low_ce_segs]):.3f} "
          f"(all enhanced: {np.mean([s['motion'] for s in enhanced]):.3f})")
    print(f"  增强级别分布: {dict(zip(*np.unique([s['action'] for s in low_ce_segs], return_counts=True)))}")
print()

# 潜在节省估算：如果把低性价比段降级到 level 0
print("=== 潜在节省估算（低性价比段降级到 level 0）===")
saved_time = sum(s['latency'] for s in low_ce_segs)
lost_gain = sum(s['vmaf_gain'] for s in low_ce_segs)
print(f"节省增强时间: {saved_time:.1f}s ({100*saved_time/total_enhance_time:.1f}%)")
print(f"损失 VMAF 增益: {lost_gain:.1f} ({100*lost_gain/total_vmaf_gain:.1f}%)")
print(f"平均每段损失: {lost_gain/len(low_ce_segs):.2f} VMAF")
print(f"对整体 avg_vmaf 影响: -{lost_gain/len(segments):.2f} VMAF")
print()

# 更激进：把最低 30% 性价比段降级
sorted_enhanced = sorted(enhanced, key=lambda s: s['cost_eff'])
n_30 = int(len(sorted_enhanced) * 0.3)
bottom_30 = sorted_enhanced[:n_30]
saved_30 = sum(s['latency'] for s in bottom_30)
lost_30 = sum(s['vmaf_gain'] for s in bottom_30)
print(f"=== 潜在节省估算（最低 30% 性价比段降级）===")
print(f"节省增强时间: {saved_30:.1f}s ({100*saved_30/total_enhance_time:.1f}%)")
print(f"损失 VMAF 增益: {lost_30:.1f} ({100*lost_30/total_vmaf_gain:.1f}%)")
print(f"对整体 avg_vmaf 影响: -{lost_30/len(segments):.2f} VMAF")
print()

# 内容复杂度与性价比的相关性
textures = [s['texture'] for s in enhanced]
motions = [s['motion'] for s in enhanced]
ces = [s['cost_eff'] for s in enhanced]
corr_text = np.corrcoef(textures, ces)[0, 1]
corr_mot = np.corrcoef(motions, ces)[0, 1]
print(f"=== 内容特征与性价比相关性 ===")
print(f"texture_complexity vs cost_eff: r={corr_text:.3f}")
print(f"motion_intensity vs cost_eff: r={corr_mot:.3f}")
print(f"（正相关=高内容段性价比高，负相关=高内容段性价比低）")

# 保存分析结果
analysis = {
    'total_enhance_time': total_enhance_time,
    'total_vmaf_gain': total_vmaf_gain,
    'enhanced_segs': enhanced_segs,
    'median_cost_eff': median_ce,
    'low_ce_count': len(low_ce_segs),
    'low_ce_saved_time': saved_time,
    'low_ce_lost_gain': lost_gain,
    'bottom_30_saved_time': saved_30,
    'bottom_30_lost_gain': lost_30,
    'corr_texture_ce': corr_text,
    'corr_motion_ce': corr_mot,
}
json.dump(analysis, open('../workspace/resource_efficiency_analysis.json', 'w'), indent=2)
print(f"\n分析结果已保存到 workspace/resource_efficiency_analysis.json")
