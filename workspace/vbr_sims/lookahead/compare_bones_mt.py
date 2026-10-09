"""
对比 BONES 基线 vs MT-BONES(SR+USM) 实验结果
"""
import json
import sys
import numpy as np

def load_result(path):
    with open(path, 'r') as f:
        return json.load(f)

def extract_metrics(data):
    """从结果JSON中提取关键指标"""
    metrics = {}

    # 尝试不同的字段名
    # 1. 平均VMAF
    for key in ['avg_vmaf', 'average_vmaf', 'vmaf_avg', 'mean_vmaf']:
        if key in data:
            metrics['avg_vmaf'] = data[key]
            break

    # 2. QoE
    for key in ['qoe', 'avg_qoe', 'qoe_score', 'total_qoe']:
        if key in data:
            metrics['qoe'] = data[key]
            break

    # 3. 卡顿/重缓冲
    for key in ['stall_rate', 'rebuffer_rate', 'stall_ratio', 'rebuffer_ratio', 'avg_stall']:
        if key in data:
            metrics['stall_rate'] = data[key]
            break
    for key in ['stall_count', 'rebuffer_count', 'num_stalls']:
        if key in data:
            metrics['stall_count'] = data[key]
            break
    for key in ['total_stall_duration', 'stall_duration', 'rebuffer_time', 'total_rebuffer']:
        if key in data:
            metrics['stall_duration'] = data[key]
            break

    # 4. 码率切换
    for key in ['bitrate_switches', 'num_switches', 'switch_count', 'quality_switches']:
        if key in data:
            metrics['bitrate_switches'] = data[key]
            break

    # 5. 平均码率
    for key in ['avg_bitrate', 'average_bitrate', 'bitrate_avg', 'mean_bitrate']:
        if key in data:
            metrics['avg_bitrate'] = data[key]
            break

    # 6. 增强相关
    for key in ['enhancement_rate', 'enhance_ratio', 'avg_enhance']:
        if key in data:
            metrics['enhancement_rate'] = data[key]
            break

    # 打印所有顶层key，帮助理解数据结构
    metrics['_top_keys'] = list(data.keys())

    return metrics

def analyze_segments(data):
    """分析段级数据"""
    seg_data = {}
    for key in ['segments', 'segment_data', 'per_segment', 'results']:
        if key in data and isinstance(data[key], list):
            seg_data = data[key]
            break
    return seg_data

def main():
    bones_path = r"E:\dev\Code\Python\bones\player\output\data\bones_imdn_always_fast-False_video-bbb_trace-trace0000_20260828-232235.json"
    mt_path = r"E:\dev\Code\Python\bones\player\output\data\mt_bones_multitask_fast-True_video-bbb_trace-trace0000_20260829-142540.json"

    bones = load_result(bones_path)
    mt = load_result(mt_path)

    print("=" * 70)
    print("BONES 基线 (SR-only, no fast)")
    print("=" * 70)
    print(f"顶层字段: {list(bones.keys())}")

    print("\n" + "=" * 70)
    print("MT-BONES (SR+USM, fast enabled)")
    print("=" * 70)
    print(f"顶层字段: {list(mt.keys())}")

    # 深入分析
    print("\n" + "=" * 70)
    print("详细字段值对比")
    print("=" * 70)

    all_keys = set(bones.keys()) | set(mt.keys())
    for key in sorted(all_keys):
        b_val = bones.get(key, "N/A")
        m_val = mt.get(key, "N/A")
        # 只打印标量值
        if isinstance(b_val, (int, float, str, bool)) or isinstance(m_val, (int, float, str, bool)):
            b_str = f"{b_val:.4f}" if isinstance(b_val, float) else str(b_val)
            m_str = f"{m_val:.4f}" if isinstance(m_val, float) else str(m_val)
            diff = ""
            if isinstance(b_val, (int, float)) and isinstance(m_val, (int, float)) and b_val != 0:
                pct = (m_val - b_val) / abs(b_val) * 100
                diff = f" ({pct:+.2f}%)"
            print(f"  {key:<35} BONES={b_str:<20} MT={m_str:<20}{diff}")

    # 段级数据分析
    print("\n" + "=" * 70)
    print("段级数据统计")
    print("=" * 70)

    for name, data in [("BONES", bones), ("MT-BONES", mt)]:
        # 查找段级数组
        for key in data:
            val = data[key]
            if isinstance(val, list) and len(val) > 10:
                # 可能是段级数据
                if all(isinstance(x, (int, float)) for x in val[:5]):
                    arr = np.array(val)
                    print(f"\n  {name} - {key}: len={len(arr)}, mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")
                elif all(isinstance(x, dict) for x in val[:5]):
                    print(f"\n  {name} - {key}: len={len(val)} (dict列表), 子字段={list(val[0].keys()) if val else 'empty'}")

if __name__ == "__main__":
    main()
