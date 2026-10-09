"""
MD-QoE（多维度QoE）计算模块

实现自定义的多维度评价指标，用于替代BONES的简单QoE公式。

BONES QoE = 平均VMAF - 平均振荡 - 平均卡顿/10

MD-QoE包含5个维度：
1. 感知质量分：平均VMAF + 低质量惩罚 + 方差惩罚
2. 时序平滑度：可感知振荡 + 高频振荡惩罚
3. 卡顿体验：非线性卡顿惩罚 + 频率惩罚
4. 资源效率：能耗 + 显存 + 增强延迟（可选，需要额外数据）
5. 启动体验：首帧延迟 + 启动质量（可选，需要额外数据）
"""
import numpy as np


def calculate_bones_qoe(quality_vmaf, freeze_time_per_seg):
    """
    计算BONES QoE

    参数:
        quality_vmaf: 逐段VMAF数组 (num_seg,)
        freeze_time_per_seg: 逐段卡顿时间数组 (num_seg,)，单位秒

    返回:
        dict: 包含qoe, avg_vmaf, oscillation, avg_freeze
    """
    num_seg = len(quality_vmaf)
    avg_vmaf = np.mean(quality_vmaf)

    # 振荡：相邻段VMAF绝对差的平均值
    if num_seg > 1:
        oscillation = np.mean(np.abs(np.diff(quality_vmaf)))
    else:
        oscillation = 0

    avg_freeze = np.mean(freeze_time_per_seg)

    # BONES QoE公式
    qoe = avg_vmaf - oscillation - avg_freeze / 10

    return {
        "qoe": float(qoe),
        "avg_vmaf": float(avg_vmaf),
        "oscillation": float(oscillation),
        "avg_freeze": float(avg_freeze),
    }


def calculate_perceived_quality(quality_vmaf, low_quality_threshold=80.0):
    """
    计算感知质量分

    包含：平均VMAF + 低质量惩罚 + 方差惩罚

    参数:
        quality_vmaf: 逐段VMAF数组
        low_quality_threshold: 低质量阈值，低于此值的段会被额外惩罚

    返回:
        float: 感知质量分
    """
    avg_vmaf = np.mean(quality_vmaf)

    # 低质量惩罚：低于阈值的段，按超出阈值的程度平方惩罚
    low_quality_penalty = 0.0
    low_quality_segs = quality_vmaf < low_quality_threshold
    if np.any(low_quality_segs):
        low_quality_diff = low_quality_threshold - quality_vmaf[low_quality_segs]
        low_quality_penalty = np.mean(low_quality_diff ** 2) / 100.0

    # 方差惩罚：VMAF的时间标准差
    quality_std = np.std(quality_vmaf)
    variance_penalty = quality_std * 0.1

    perceived_quality = avg_vmaf - low_quality_penalty - variance_penalty

    return float(perceived_quality)


def calculate_temporal_smoothness(quality_vmaf, perception_threshold=1.0):
    """
    计算时序平滑度

    包含：可感知振荡 + 高频振荡惩罚

    参数:
        quality_vmaf: 逐段VMAF数组
        perception_threshold: 感知阈值，小于此值的变化不被惩罚

    返回:
        float: 时序平滑度（负值，表示惩罚）
    """
    num_seg = len(quality_vmaf)
    if num_seg < 2:
        return 0.0

    diffs = np.abs(np.diff(quality_vmaf))

    # 可感知振荡：只计算超过感知阈值的变化
    perceivable_oscillation = np.mean(np.maximum(0, diffs - perception_threshold))

    # 高频振荡惩罚：检测频繁上下波动的模式
    # 模式：q[i-2] < q[i-1] > q[i] < q[i+1] 或反向
    high_freq_penalty = 0.0
    if num_seg >= 4:
        for i in range(2, num_seg - 1):
            # 检测局部极值（峰或谷）
            is_peak = (quality_vmaf[i] > quality_vmaf[i-1] and
                       quality_vmaf[i] > quality_vmaf[i+1])
            is_valley = (quality_vmaf[i] < quality_vmaf[i-1] and
                         quality_vmaf[i] < quality_vmaf[i+1])
            if is_peak or is_valley:
                high_freq_penalty += 0.5
        high_freq_penalty = high_freq_penalty / num_seg

    temporal_smoothness = -perceivable_oscillation - high_freq_penalty

    return float(temporal_smoothness)


def calculate_stalling_experience(freeze_time_per_seg, short_threshold=0.5):
    """
    计算卡顿体验

    包含：非线性卡顿惩罚（短卡顿轻，长卡重重）+ 频率惩罚

    参数:
        freeze_time_per_seg: 逐段卡顿时间数组，单位秒
        short_threshold: 短卡顿阈值，小于此值的为短卡顿

    返回:
        float: 卡顿体验（负值，表示惩罚）
    """
    # 分离短卡顿和长卡顿
    short_stalls = np.minimum(freeze_time_per_seg, short_threshold)
    long_stalls = np.maximum(0, freeze_time_per_seg - short_threshold)

    # 短卡顿惩罚：线性，权重较低
    short_penalty = np.sum(short_stalls) * 0.5

    # 长卡顿惩罚：平方，权重较高
    long_penalty = np.sum(long_stalls ** 2) * 2.0

    # 卡顿频率惩罚：卡顿事件数量
    stall_events = np.sum(freeze_time_per_seg > 0.01)  # 超过10ms算一次卡顿事件
    frequency_penalty = stall_events * 0.3

    stalling_experience = -short_penalty - long_penalty - frequency_penalty

    return float(stalling_experience)


def calculate_md_qoe(quality_vmaf, freeze_time_per_seg,
                      weights=None, low_quality_threshold=80.0,
                      perception_threshold=1.0, short_threshold=0.5):
    """
    计算MD-QoE（多维度QoE）

    参数:
        quality_vmaf: 逐段VMAF数组 (num_seg,)
        freeze_time_per_seg: 逐段卡顿时间数组 (num_seg,)，单位秒
        weights: 各维度权重字典，默认 {"quality": 0.4, "smoothness": 0.2, "stalling": 0.25, "resource": 0.1, "startup": 0.05}
        low_quality_threshold: 低质量阈值
        perception_threshold: 振荡感知阈值
        short_threshold: 短卡顿阈值

    返回:
        dict: 包含md_qoe, 各维度分数, bones_qoe
    """
    if weights is None:
        weights = {
            "quality": 0.4,
            "smoothness": 0.2,
            "stalling": 0.25,
            "resource": 0.1,
            "startup": 0.05,
        }

    # 计算各维度
    perceived_quality = calculate_perceived_quality(quality_vmaf, low_quality_threshold)
    temporal_smoothness = calculate_temporal_smoothness(quality_vmaf, perception_threshold)
    stalling_experience = calculate_stalling_experience(freeze_time_per_seg, short_threshold)

    # 资源效率和启动体验暂时设为0（需要额外数据）
    resource_efficiency = 0.0
    startup_experience = 0.0

    # 综合MD-QoE（加权求和）
    md_qoe = (weights["quality"] * perceived_quality +
              weights["smoothness"] * temporal_smoothness +
              weights["stalling"] * stalling_experience +
              weights["resource"] * resource_efficiency +
              weights["startup"] * startup_experience)

    # 同时计算BONES QoE用于对比
    bones_qoe_result = calculate_bones_qoe(quality_vmaf, freeze_time_per_seg)

    return {
        "md_qoe": float(md_qoe),
        "perceived_quality": float(perceived_quality),
        "temporal_smoothness": float(temporal_smoothness),
        "stalling_experience": float(stalling_experience),
        "resource_efficiency": float(resource_efficiency),
        "startup_experience": float(startup_experience),
        "bones_qoe": bones_qoe_result["qoe"],
        "bones_avg_vmaf": bones_qoe_result["avg_vmaf"],
        "bones_oscillation": bones_qoe_result["oscillation"],
        "bones_avg_freeze": bones_qoe_result["avg_freeze"],
    }


if __name__ == "__main__":
    # 简单测试
    print("MD-QoE计算模块测试")
    print("=" * 50)

    # 测试1：稳定高质量
    q1 = np.array([95.0] * 10)
    f1 = np.zeros(10)
    r1 = calculate_md_qoe(q1, f1)
    print(f"测试1（稳定高质量）: MD-QoE={r1['md_qoe']:.2f}, BONES-QoE={r1['bones_qoe']:.2f}")

    # 测试2：振荡质量
    q2 = np.array([90.0, 95.0, 90.0, 95.0, 90.0, 95.0, 90.0, 95.0, 90.0, 95.0])
    f2 = np.zeros(10)
    r2 = calculate_md_qoe(q2, f2)
    print(f"测试2（振荡质量）: MD-QoE={r2['md_qoe']:.2f}, BONES-QoE={r2['bones_qoe']:.2f}")
    print(f"  平滑度: {r2['temporal_smoothness']:.2f}, 高频振荡惩罚明显")

    # 测试3：低质量段
    q3 = np.array([70.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0, 95.0])
    f3 = np.zeros(10)
    r3 = calculate_md_qoe(q3, f3)
    print(f"测试3（含低质量段）: MD-QoE={r3['md_qoe']:.2f}, BONES-QoE={r3['bones_qoe']:.2f}")
    print(f"  感知质量: {r3['perceived_quality']:.2f}, 低质量惩罚明显")

    # 测试4：卡顿
    q4 = np.array([95.0] * 10)
    f4 = np.array([0, 0, 0.2, 0, 0, 0.8, 0, 0, 0, 0])
    r4 = calculate_md_qoe(q4, f4)
    print(f"测试4（含卡顿）: MD-QoE={r4['md_qoe']:.2f}, BONES-QoE={r4['bones_qoe']:.2f}")
    print(f"  卡顿体验: {r4['stalling_experience']:.2f}, 长卡顿惩罚更重")

    print("\n测试完成")
