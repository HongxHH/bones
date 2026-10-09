"""
深入分析段级数据 - 健壮版
"""
import json
import numpy as np
from collections import Counter

def load_result(path):
    with open(path, 'r') as f:
        return json.load(f)

def safe_mean(vals):
    clean = [v for v in vals if v is not None]
    return np.mean(clean) if clean else 0

def safe_std(vals):
    clean = [v for v in vals if v is not None]
    return np.std(clean) if clean else 0

def analyze_segments(name, data):
    segs = data['segments']
    print(f"\n{'='*70}")
    print(f"{name} - 段级分析 (共{len(segs)}段)")
    print(f"{'='*70}")

    # 1. 下载码率分布
    bitrates = [s['bitrate'] for s in segs]
    print(f"\n码率分布: {Counter(bitrates).most_common()}")
    print(f"平均码率: {np.mean(bitrates):.0f}")

    # 2. 增强动作分布
    enhance_actions = [s.get('enhance_action', -1) for s in segs]
    print(f"\n增强动作分布: {Counter(enhance_actions).most_common()}")
    print(f"  0=无增强, 1=SR, 2=SR+USM, 3=USM, 4=SR+USM+FI")

    # 3. is_enhance分布
    is_enhance = [1 if s.get('is_enhance') else 0 for s in segs]
    print(f"\n是否增强: True={sum(is_enhance)}, False={len(is_enhance)-sum(is_enhance)}")

    # 4. VMAF分析
    base_vmaf = [s.get('base_vmaf') for s in segs]
    enhance_vmaf = [s.get('enhance_vmaf') for s in segs]
    real_vmaf = [s.get('real_vmaf') for s in segs]

    print(f"\nVMAF统计:")
    print(f"  base_vmaf:    mean={safe_mean(base_vmaf):.2f}, std={safe_std(base_vmaf):.2f}")
    print(f"  enhance_vmaf: mean={safe_mean(enhance_vmaf):.2f}, std={safe_std(enhance_vmaf):.2f}")
    print(f"  real_vmaf:    mean={safe_mean(real_vmaf):.2f}, std={safe_std(real_vmaf):.2f}")

    # 5. 增强增益 (仅对有增强的段)
    enhanced_segs = [s for s in segs if s.get('is_enhance') and s.get('real_vmaf') is not None and s.get('base_vmaf') is not None]
    if enhanced_segs:
        gains = [s['real_vmaf'] - s['base_vmaf'] for s in enhanced_segs]
        positive = sum(1 for g in gains if g > 0)
        print(f"\n增强段增益(real-base, n={len(enhanced_segs)}): mean={np.mean(gains):.2f}, 正增益={positive}/{len(enhanced_segs)}")

    # 6. enhance_scale分布
    scales = [s.get('enhance_scale') for s in segs if s.get('is_enhance') and s.get('enhance_scale') is not None]
    if scales:
        print(f"\nenhance_scale分布(增强段): {Counter(scales).most_common()}")

    # 7. fast_complete
    fast_complete = [1 if s.get('fast_complete') else 0 for s in segs]
    print(f"\nfast_complete: True={sum(fast_complete)}, False={len(fast_complete)-sum(fast_complete)}")

    # 8. 前15段详情
    print(f"\n前15段详情:")
    print(f"  {'idx':>4} {'bitrate':>8} {'e_act':>5} {'enh':>4} {'sc':>4} {'base_v':>7} {'real_v':>7} {'gain':>6} {'fast':>5}")
    for s in segs[:15]:
        bv = s.get('base_vmaf')
        rv = s.get('real_vmaf')
        gain = f"{rv-bv:+.1f}" if (bv is not None and rv is not None) else "N/A"
        print(f"  {s['index']:>4} {s['bitrate']:>8} {s.get('enhance_action',-1):>5} {('Y' if s.get('is_enhance') else 'N'):>4} {str(s.get('enhance_scale','-')):>4} {bv if bv is not None else 'N/A':>7} {rv if rv is not None else 'N/A':>7} {gain:>6} {('Y' if s.get('fast_complete') else 'N'):>5}")

    return segs

def main():
    bones_path = r"E:\dev\Code\Python\bones\player\output\data\bones_imdn_always_fast-False_video-bbb_trace-trace0000_20260828-232235.json"
    mt_path = r"E:\dev\Code\Python\bones\player\output\data\mt_bones_multitask_fast-True_video-bbb_trace-trace0000_20260829-142540.json"

    bones = load_result(bones_path)
    mt = load_result(mt_path)

    bones_segs = analyze_segments("BONES (SR-only, no fast)", bones)
    mt_segs = analyze_segments("MT-BONES (SR+USM, fast enabled)", mt)

    # 对比每段的real_vmaf
    print(f"\n{'='*70}")
    print("逐段real_vmaf对比 (MT - BONES)")
    print(f"{'='*70}")
    diffs = []
    for b, m in zip(bones_segs, mt_segs):
        bv = b.get('real_vmaf')
        mv = m.get('real_vmaf')
        if bv is not None and mv is not None:
            diffs.append(mv - bv)

    diffs_arr = np.array(diffs)
    print(f"平均差值: {diffs_arr.mean():.2f}")
    print(f"正差值段: {sum(1 for d in diffs if d > 0)}/{len(diffs)}")
    print(f"负差值段: {sum(1 for d in diffs if d < 0)}/{len(diffs)}")
    print(f"最大正差: {diffs_arr.max():.2f} (段{np.argmax(diffs_arr)})")
    print(f"最大负差: {diffs_arr.min():.2f} (段{np.argmin(diffs_arr)})")

    # 对比码率选择
    print(f"\n{'='*70}")
    print("码率选择差异")
    print(f"{'='*70}")
    b_bitrates = [s['bitrate'] for s in bones_segs]
    m_bitrates = [s['bitrate'] for s in mt_segs]
    same = sum(1 for b, m in zip(b_bitrates, m_bitrates) if b == m)
    higher = sum(1 for b, m in zip(b_bitrates, m_bitrates) if m > b)
    lower = sum(1 for b, m in zip(b_bitrates, m_bitrates) if m < b)
    print(f"码率相同: {same}/{len(b_bitrates)}")
    print(f"MT更高: {higher}/{len(b_bitrates)}")
    print(f"MT更低: {lower}/{len(b_bitrates)}")

    # GPU指标对比
    print(f"\n{'='*70}")
    print("GPU指标对比")
    print(f"{'='*70}")
    for key in ['gpu_metrics']:
        b_gpu = bones.get(key, {})
        m_gpu = mt.get(key, {})
        all_gpu_keys = set(b_gpu.keys()) | set(m_gpu.keys())
        for k in sorted(all_gpu_keys):
            bv = b_gpu.get(k, "N/A")
            mv = m_gpu.get(k, "N/A")
            print(f"  {k:<30} BONES={str(bv):<15} MT={str(mv):<15}")

if __name__ == "__main__":
    main()
