"""
帧级内容感知增强调度评估

对比：
1. 段级固定增强（BONES方式）：整段用同一个增强级别
2. 帧级贪心调度：每帧独立选择使边际收益最大的动作
3. 帧级前瞻DP调度：段内全局优化，在GPU预算内最大化总VMAF

数据来源：
- per_frame_quality.json：逐帧VMAF（base/sr_low/sr_medium/sr_high/usm_only）
- 内容特征：运动强度、纹理复杂度、场景切换
- imdn_bbb_3060ti.json：各增强级别的GPU时间
"""
import json
import numpy as np
from collections import defaultdict

# 数据路径
PER_FRAME_FILE = r"E:\dev\Code\Python\bones\data\bbb\per_frame_quality.json"
ENHANCE_FILE = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_3060ti.json"

# 增强级别定义（索引对应）
# 0: 不增强(base), 1: sr_low, 2: sr_medium, 3: sr_high
LEVELS = [0, 1, 2, 3]
LEVEL_NAMES = ["base", "sr_low", "sr_medium", "sr_high"]


def load_data():
    """加载逐帧VMAF数据和内容特征"""
    with open(PER_FRAME_FILE) as f:
        data = json.load(f)

    with open(ENHANCE_FILE) as f:
        enh = json.load(f)

    # 增强FPS表：enhancement_fps[resolution_idx][method_idx]
    # resolution: 0=240p, 1=360p, 2=480p
    # method: 0=low, 1=medium, 2=high, 3=ultra
    fps_table = enh["enhancement_fps"]
    print("增强FPS表:")
    for i, res in enumerate([240, 360, 480]):
        print(f"  {res}p: low={fps_table[i][0]:.1f}, medium={fps_table[i][1]:.1f}, "
              f"high={fps_table[i][2]:.1f}, ultra={fps_table[i][3]:.1f}")

    return data, fps_table


def get_frame_vmafs(data, res, seg_idx):
    """
    获取某段某分辨率下，各增强级别的逐帧VMAF

    Returns:
        vmafs: dict {level: np.array(120,)}  level=0,1,2,3
    """
    vmafs = {}
    for level, name in enumerate(LEVEL_NAMES):
        key = f"{res}_{name}_{seg_idx}"
        if key in data["results"]:
            vmafs[level] = np.array(data["results"][key])
        else:
            return None
    return vmafs


def get_content_features(data, seg_idx):
    """获取某段的内容特征"""
    if str(seg_idx) in data["content_features"]:
        cf = data["content_features"][str(seg_idx)]
        return {
            "motion": np.array(cf["motion"]),
            "texture": np.array(cf["texture"]),
            "scene_change": np.array(cf["scene_change"]),
        }
    return None


def compute_gpu_time_ms(fps_table, res_idx, level, num_frames=120):
    """
    计算某增强级别处理num_frames帧的GPU时间（毫秒）

    level: 1=low, 2=medium, 3=high
    """
    if level == 0:
        return 0.0
    method_idx = level - 1  # low=0, medium=1, high=2
    fps = fps_table[res_idx][method_idx]
    if fps <= 0:
        return float("inf")
    return num_frames / fps * 1000.0  # ms


def segment_fixed_scheduling(vmafs, gpu_budget_ms, fps_table, res_idx):
    """
    段级固定增强：整段用同一个增强级别，选择在GPU预算内VMAF最高的级别

    Returns:
        best_level, total_vmaf, gpu_used
    """
    best_level = 0
    best_vmaf = -1
    best_gpu = 0

    for level in LEVELS:
        gpu_time = compute_gpu_time_ms(fps_table, res_idx, level, 120)
        if gpu_time <= gpu_budget_ms:
            avg_vmaf = np.mean(vmafs[level])
            if avg_vmaf > best_vmaf:
                best_vmaf = avg_vmaf
                best_level = level
                best_gpu = gpu_time

    return best_level, best_vmaf * 120, best_gpu


def frame_greedy_scheduling(vmafs, gpu_budget_ms, fps_table, res_idx, content_features=None):
    """
    帧级贪心调度：每帧独立选择使"单位GPU时间VMAF增益"最大的动作

    策略：
    - 计算每帧每个级别的VMAF和GPU时间
    - 按边际收益（VMAF增益/GPU时间）排序
    - 贪心选择，直到GPU预算用完

    Returns:
        decisions, total_vmaf, gpu_used
    """
    num_frames = 120
    decisions = np.zeros(num_frames, dtype=int)
    gpu_used = 0.0

    # 计算所有可能的(帧, 级别)组合的边际收益
    candidates = []
    for f in range(num_frames):
        base_vmaf = vmafs[0][f]
        for level in [1, 2, 3]:
            enh_vmaf = vmafs[level][f]
            gain = enh_vmaf - base_vmaf
            gpu_time = compute_gpu_time_ms(fps_table, res_idx, level, 1)
            if gpu_time > 0 and gain > 0:
                efficiency = gain / gpu_time  # VMAF点/毫秒
                candidates.append((efficiency, f, level, gain, gpu_time))

    # 按边际收益降序排序
    candidates.sort(key=lambda x: -x[0])

    # 贪心选择
    selected = set()
    for eff, f, level, gain, gpu_time in candidates:
        if f in selected:
            continue
        if gpu_used + gpu_time <= gpu_budget_ms:
            decisions[f] = level
            gpu_used += gpu_time
            selected.add(f)

    # 计算总VMAF
    total_vmaf = sum(vmafs[decisions[f]][f] for f in range(num_frames))
    return decisions, total_vmaf, gpu_used


def frame_lookahead_dp(vmafs, gpu_budget_ms, fps_table, res_idx, content_features=None):
    """
    帧级前瞻DP调度：在GPU预算约束下，最大化段内总VMAF

    状态：dp[f][c] = 前f帧用c毫秒GPU时间的最大总VMAF
    动作：每帧选择level ∈ {0,1,2,3}

    复杂度：O(120 × budget × 4)，budget通常为4000ms（4秒）

    Returns:
        decisions, total_vmaf, gpu_used
    """
    num_frames = 120
    # GPU时间离散化到10ms粒度，减少状态数
    step_ms = 10
    budget_steps = int(gpu_budget_ms / step_ms) + 1

    # 预计算每帧每个级别的GPU时间（步数）和VMAF
    frame_gpu_steps = np.zeros((num_frames, 4), dtype=int)
    frame_vmaf = np.zeros((num_frames, 4))
    for f in range(num_frames):
        for level in range(4):
            frame_vmaf[f][level] = vmafs[level][f]
            gpu_ms = compute_gpu_time_ms(fps_table, res_idx, level, 1)
            frame_gpu_steps[f][level] = int(gpu_ms / step_ms)

    # DP
    # dp[f][c] = 前f帧用c步GPU时间的最大总VMAF
    dp = np.full((num_frames + 1, budget_steps), -np.inf)
    dp[0][0] = 0
    # 记录决策
    choice = np.zeros((num_frames + 1, budget_steps), dtype=int)

    for f in range(num_frames):
        for c in range(budget_steps):
            if dp[f][c] == -np.inf:
                continue
            for level in range(4):
                new_c = c + frame_gpu_steps[f][level]
                if new_c < budget_steps:
                    new_vmaf = dp[f][c] + frame_vmaf[f][level]
                    if new_vmaf > dp[f + 1][new_c]:
                        dp[f + 1][new_c] = new_vmaf
                        choice[f + 1][new_c] = level

    # 找最优终点
    best_c = np.argmax(dp[num_frames])
    total_vmaf = dp[num_frames][best_c]

    # 回溯决策
    decisions = np.zeros(num_frames, dtype=int)
    c = best_c
    for f in range(num_frames - 1, -1, -1):
        level = choice[f + 1][c]
        decisions[f] = level
        c -= frame_gpu_steps[f][level]

    gpu_used = best_c * step_ms
    return decisions, total_vmaf, gpu_used


def run_evaluation():
    """运行完整评估"""
    print("=" * 70)
    print("帧级内容感知增强调度评估")
    print("=" * 70)

    data, fps_table = load_data()
    resolutions = [240, 360, 480]
    num_segments = 158  # 跳过最后一段

    # GPU预算：段时长4秒=4000ms（模拟BONES段级固定增强的GPU时间）
    # 对比不同预算场景
    gpu_budgets = [2000, 4000, 6000]  # 受限/正常/充足

    results = defaultdict(lambda: defaultdict(list))

    for res_idx, res in enumerate(resolutions):
        print(f"\n{'='*70}")
        print(f"分辨率: {res}p")
        print(f"{'='*70}")

        for budget in gpu_budgets:
            seg_fixed_vmafs = []
            greedy_vmafs = []
            dp_vmafs = []
            fixed_gpus = []
            greedy_gpus = []
            dp_gpus = []
            fixed_levels = []
            greedy_enh_ratios = []
            dp_enh_ratios = []

            valid_segs = 0
            for seg_idx in range(num_segments):
                vmafs = get_frame_vmafs(data, res, seg_idx)
                if vmafs is None:
                    continue

                # 1. 段级固定增强
                fixed_level, fixed_vmaf, fixed_gpu = segment_fixed_scheduling(
                    vmafs, budget, fps_table, res_idx)
                seg_fixed_vmafs.append(fixed_vmaf / 120)
                fixed_gpus.append(fixed_gpu)
                fixed_levels.append(fixed_level)

                # 2. 帧级贪心调度
                greedy_dec, greedy_vmaf, greedy_gpu = frame_greedy_scheduling(
                    vmafs, budget, fps_table, res_idx)
                greedy_vmafs.append(greedy_vmaf / 120)
                greedy_gpus.append(greedy_gpu)
                greedy_enh_ratios.append(np.mean(greedy_dec > 0))

                # 3. 帧级前瞻DP
                dp_dec, dp_vmaf, dp_gpu = frame_lookahead_dp(
                    vmafs, budget, fps_table, res_idx)
                dp_vmafs.append(dp_vmaf / 120)
                dp_gpus.append(dp_gpu)
                dp_enh_ratios.append(np.mean(dp_dec > 0))

                valid_segs += 1

            if valid_segs == 0:
                continue

            print(f"\n  GPU预算: {budget}ms ({valid_segs}段)")
            print(f"  {'方法':<16} {'平均VMAF':<10} {'VMAF增益':<10} {'平均GPU':<10} {'增强帧比例':<12}")
            print(f"  {'-'*60}")

            fixed_avg = np.mean(seg_fixed_vmafs)
            greedy_avg = np.mean(greedy_vmafs)
            dp_avg = np.mean(dp_vmafs)

            print(f"  {'段级固定':<16} {fixed_avg:<10.2f} {'0.00':<10} "
                  f"{np.mean(fixed_gpus):<10.0f} {np.mean([1 if l>0 else 0 for l in fixed_levels])*100:<11.1f}%")
            print(f"  {'帧级贪心':<16} {greedy_avg:<10.2f} {greedy_avg-fixed_avg:<10.2f} "
                  f"{np.mean(greedy_gpus):<10.0f} {np.mean(greedy_enh_ratios)*100:<11.1f}%")
            print(f"  {'帧级前瞻DP':<16} {dp_avg:<10.2f} {dp_avg-fixed_avg:<10.2f} "
                  f"{np.mean(dp_gpus):<10.0f} {np.mean(dp_enh_ratios)*100:<11.1f}%")

            # 统计显著性
            diff_dp = np.array(dp_vmafs) - np.array(seg_fixed_vmafs)
            print(f"  DP vs 固定: 平均增益={np.mean(diff_dp):.2f}, "
                  f"正增益段比例={np.mean(diff_dp>0)*100:.1f}%, "
                  f"最大增益={np.max(diff_dp):.2f}, 最大损失={np.min(diff_dp):.2f}")

            results[res][budget] = {
                "fixed_vmaf": fixed_avg,
                "greedy_vmaf": greedy_avg,
                "dp_vmaf": dp_avg,
                "dp_gain": dp_avg - fixed_avg,
                "fixed_gpu": np.mean(fixed_gpus),
                "dp_gpu": np.mean(dp_gpus),
                "dp_enh_ratio": np.mean(dp_enh_ratios),
                "positive_ratio": np.mean(diff_dp > 0),
            }

    # 汇总
    print(f"\n{'='*70}")
    print("汇总：帧级前瞻DP vs 段级固定")
    print(f"{'='*70}")
    print(f"{'分辨率':<8} {'GPU预算':<10} {'固定VMAF':<10} {'DP VMAF':<10} {'增益':<8} {'正增益%':<10} {'DP GPU':<10}")
    print("-" * 70)
    for res in resolutions:
        for budget in gpu_budgets:
            if budget in results[res]:
                r = results[res][budget]
                print(f"{res}p{'':<5} {budget}ms{'':<4} {r['fixed_vmaf']:<10.2f} "
                      f"{r['dp_vmaf']:<10.2f} {r['dp_gain']:<8.2f} "
                      f"{r['positive_ratio']*100:<10.1f} {r['dp_gpu']:<10.0f}")

    # 保存结果
    output = {
        "description": "帧级内容感知增强调度评估结果",
        "resolutions": resolutions,
        "gpu_budgets_ms": gpu_budgets,
        "results": {str(res): {str(b): v for b, v in bd.items()}
                    for res, bd in results.items()},
    }
    with open(r"E:\dev\Code\Python\bones\workspace\frame_level_scheduling_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存到 frame_level_scheduling_results.json")


if __name__ == "__main__":
    run_evaluation()
