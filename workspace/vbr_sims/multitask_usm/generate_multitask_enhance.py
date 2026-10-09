"""
多任务增强数据生成器
将 BONES 的"增强级别"维度重新解释为"任务组合ID"：
  0: None (基线，无增强)
  1: SR only (NanoVSR 超分)
  2: SR + Deblur (低分辨率去模糊 + 超分)
  3: Deblur only (仅去模糊，适用于高码率)
  4: SR + Deblur + FI (超分 + 去模糊 + 插帧)

延迟基于 RTX 5080 FP16 实测：
  SR (NanoVSR-644k, 270x480): 7.4ms/帧
  Deblur (NAFNet-Small, 低分辨率): 22ms/帧
  FI (RIFE-lite, 估算): 15ms/帧

质量基于现有 IMDN VMAF 数据 + 去模糊增益估算。
"""
import json
import numpy as np
import os

# 路径
BASE_DIR = r'E:\dev\Code\Python\bones\data\bbb'
INPUT_FILE = os.path.join(BASE_DIR, 'imdn_bbb_3060ti.json')
OUTPUT_FILE = os.path.join(BASE_DIR, 'multitask_bbb_rtx5080.json')

# 任务组合定义
TASK_COMBOS = ['sr_only', 'sr_deblur', 'deblur_only', 'sr_deblur_fi']
NUM_COMBOS = len(TASK_COMBOS)  # 4 (excluding none)

# 每帧延迟 (ms)，按输入分辨率档位 [240p, 360p, 480p, 720p]
# 基于实测 + 分辨率缩放估算
SR_LATENCY_PER_FRAME = [7.4, 9.0, 12.0, 18.0]      # NanoVSR FP16
DEBLUR_LATENCY_PER_FRAME = [22.0, 22.0, 22.0, 28.0]  # NAFNet-Small FP16 (低分辨率GPU未占满)
FI_LATENCY_PER_FRAME = [15.0, 18.0, 24.0, 35.0]      # RIFE-lite 估算

FRAME_RATE = 30
SEG_TIME_MS = 4000  # 4秒每段
FRAMES_PER_SEG = FRAME_RATE * (SEG_TIME_MS / 1000)  # 120帧


def compute_segment_fps(latency_per_frame_ms):
    """单帧延迟(ms) → 段处理FPS"""
    if latency_per_frame_ms <= 0:
        return -1  # 无效
    return 1000.0 / latency_per_frame_ms


def compute_combo_latency(combo_idx, bitrate_idx):
    """计算任务组合在指定码率下的单帧总延迟"""
    sr_lat = SR_LATENCY_PER_FRAME[bitrate_idx]
    deblur_lat = DEBLUR_LATENCY_PER_FRAME[bitrate_idx]
    fi_lat = FI_LATENCY_PER_FRAME[bitrate_idx]

    if combo_idx == 0:  # sr_only
        return sr_lat
    elif combo_idx == 1:  # sr_deblur
        return deblur_lat + sr_lat  # 先去模糊(低分辨率)再超分
    elif combo_idx == 2:  # deblur_only
        return deblur_lat
    elif combo_idx == 3:  # sr_deblur_fi
        return deblur_lat + sr_lat + fi_lat
    else:
        return -1


def estimate_deblur_vmaf_boost(base_vmaf, sr_gain, segment_idx):
    """
    估算去模糊带来的 VMAF 增益。
    内容越复杂(SR增益越大)，去模糊增益越高。
    静态场景增益小，运动场景增益大。
    """
    # SR增益作为内容复杂度代理
    complexity = np.clip(sr_gain / 15.0, 0, 1)  # 归一化到[0,1]
    # 去模糊增益: 0.5 ~ 4.0 VMAF
    boost = 0.5 + 3.5 * complexity
    # 添加少量段级变化
    np.random.seed(segment_idx)
    boost *= (0.8 + 0.4 * np.random.random())
    return boost


def estimate_fi_vmaf_boost():
    """
    插帧对单帧VMAF影响很小，主要提升流畅度。
    估算 +0.5 ~ 1.5 VMAF (间接提升，减少运动模糊感知)
    """
    return 1.0


def main():
    # 加载现有增强数据
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    base_vmaf = np.array(data['base_quality_vmaf'])      # (4, 159)
    base_ssim = np.array(data['base_quality_ssim'])
    base_psnr = np.array(data['base_quality_psnr'])
    enh_vmaf = np.array(data['enhanced_quality_vmaf'])    # (3, 4, 159)
    enh_ssim = np.array(data['enhanced_quality_ssim'])
    enh_psnr = np.array(data['enhanced_quality_psnr'])

    num_bitrate_low = base_vmaf.shape[0]  # 4 (excluding highest)
    num_segments = base_vmaf.shape[1]      # 159

    print(f"Base VMAF shape: {base_vmaf.shape}")
    print(f"Enhanced VMAF shape: {enh_vmaf.shape}")
    print(f"Segments: {num_segments}")

    # === 1. 生成 enhancement_fps (4 bitrate × 4 task combos) ===
    enhancement_fps = np.zeros((num_bitrate_low, NUM_COMBOS))
    for br in range(num_bitrate_low):
        for combo in range(NUM_COMBOS):
            latency = compute_combo_latency(combo, br)
            fps = compute_segment_fps(latency)
            enhancement_fps[br, combo] = fps

    print("\nEnhancement FPS (bitrate × task_combo):")
    print(f"  {'':>10}", end='')
    for combo in TASK_COMBOS:
        print(f"{combo:>15}", end='')
    print()
    for br in range(num_bitrate_low):
        print(f"  bitrate{br:>3}:", end='')
        for combo in range(NUM_COMBOS):
            print(f"{enhancement_fps[br, combo]:>15.1f}", end='')
        print()

    # === 2. 生成 enhanced_quality_vmaf (4 task_combo × 4 bitrate × 159 segments) ===
    # 注意：enh_vmaf shape = (num_bitrate_enh=3, num_method=4, num_segment=159)
    # 即 (3个有增强数据的码率, 4个增强级别, 159段)
    # 选择增强级别 2 (index=2, 中高质量, ~40FPS) 作为 sr_only 基准
    SR_LEVEL = 2
    # sr_only_vmaf: (num_bitrate_low=4, num_segments=159)
    sr_only_vmaf = np.zeros((num_bitrate_low, num_segments))
    for br in range(num_bitrate_low):
        if br < enh_vmaf.shape[0]:
            sr_only_vmaf[br] = enh_vmaf[br, SR_LEVEL, :]  # 该码率下级别2的VMAF
        else:
            # 第4个码率无增强数据，用前一个码率的增益比例估算
            prev_gain = enh_vmaf[br-1, SR_LEVEL, :] - base_vmaf[br-1, :]
            sr_only_vmaf[br] = base_vmaf[br, :] + np.clip(prev_gain * 0.8, 0, 15)

    enhanced_quality_vmaf = np.zeros((NUM_COMBOS, num_bitrate_low, num_segments))
    enhanced_quality_ssim = np.zeros((NUM_COMBOS, num_bitrate_low, num_segments))
    enhanced_quality_psnr = np.zeros((NUM_COMBOS, num_bitrate_low, num_segments))

    for seg in range(num_segments):
        for br in range(num_bitrate_low):
            base_q = base_vmaf[br, seg]
            sr_q = max(sr_only_vmaf[br, seg], base_q)  # 确保SR不低于基线
            sr_gain = max(0, sr_q - base_q)

            # Combo 0: SR only
            enhanced_quality_vmaf[0, br, seg] = sr_q

            # Combo 1: SR + Deblur
            deblur_boost = estimate_deblur_vmaf_boost(base_q, sr_gain, seg)
            enhanced_quality_vmaf[1, br, seg] = min(100, sr_q + deblur_boost)

            # Combo 2: Deblur only
            deblur_boost_only = estimate_deblur_vmaf_boost(base_q, sr_gain * 0.5, seg + 1000)
            enhanced_quality_vmaf[2, br, seg] = min(100, base_q + deblur_boost_only)

            # Combo 3: SR + Deblur + FI
            fi_boost = estimate_fi_vmaf_boost()
            enhanced_quality_vmaf[3, br, seg] = min(100, sr_q + deblur_boost + fi_boost)

    # SSIM 和 PSNR 使用比例缩放 (简化)
    for combo in range(NUM_COMBOS):
        vmaf_ratio = enhanced_quality_vmaf[combo] / (base_vmaf + 1e-6)
        enhanced_quality_ssim[combo] = np.clip(base_ssim * vmaf_ratio, 0, 1)
        enhanced_quality_psnr[combo] = base_psnr * np.clip(vmaf_ratio, 0.9, 1.15)

    print(f"\nEnhanced VMAF shape: {enhanced_quality_vmaf.shape}")
    print("Average VMAF per combo (across all bitrate/segments):")
    for i, combo in enumerate(TASK_COMBOS):
        avg = np.mean(enhanced_quality_vmaf[i])
        print(f"  {combo:>15}: {avg:.2f}")

    # === 3. 组装输出 JSON ===
    output = {
        'frame_rate': FRAME_RATE,
        'enhancement_fps': enhancement_fps.tolist(),
        'base_quality_vmaf': base_vmaf.tolist(),
        'base_quality_ssim': base_ssim.tolist(),
        'base_quality_psnr': base_psnr.tolist(),
        'enhanced_quality_vmaf': enhanced_quality_vmaf.tolist(),
        'enhanced_quality_ssim': enhanced_quality_ssim.tolist(),
        'enhanced_quality_psnr': enhanced_quality_psnr.tolist(),
        # 元数据
        '_metadata': {
            'description': 'Multi-task enhancement data for BONES-MT',
            'task_combos': ['none'] + TASK_COMBOS,
            'gpu': 'RTX 5080',
            'precision': 'FP16',
            'sr_model': 'NanoVSR-644k',
            'deblur_model': 'NAFNet-Small',
            'fi_model': 'RIFE-lite (estimated)',
            'seg_time_ms': SEG_TIME_MS,
            'note': 'Deblur runs at input resolution before SR. VMAF boosts are content-dependent estimates.'
        }
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput saved to: {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024:.1f} KB")
    print("DONE")


if __name__ == '__main__':
    main()
