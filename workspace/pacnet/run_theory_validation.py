"""
PAC-Net Halt调度理论验证实验
实验1：预测误差敏感性（注入高斯噪声，测竞争比下降曲线）
实验2：缓冲区大小与最大可持续e3比例的关系
"""
import json
import numpy as np

# 加载数据
data = json.load(open(r'E:\dev\Code\Python\bones\workspace\pacnet\per_seg_psnr.json'))
N = len(data)
e2 = np.array([data[str(i)]['e2'] for i in range(N)])
e3 = np.array([data[str(i)]['e3'] for i in range(N)])
delta23 = e3 - e2  # 真实深算增益

DT = 33.3   # 帧间隔 ms
D2 = 26.0   # e2延迟 ms
D3 = 39.0   # e3延迟 ms

def can_play(e3_mask, B):
    """检查给定e3选择序列是否不卡顿"""
    budget = float(B)
    for i in range(N):
        budget += DT
        d = D3 if e3_mask[i] else D2
        if budget < d:
            return False
        budget -= d
    return True

def offline_opt(B):
    """离线最优：按增益降序贪心选择，直到不卡顿约束"""
    sorted_idx = np.argsort(-delta23)
    mask = np.zeros(N, dtype=bool)
    for idx in sorted_idx:
        mask[idx] = True
        if not can_play(mask, B):
            mask[idx] = False
            break
    return mask

def threshold_strategy(pred_gain, gamma, B):
    """纯阈值策略：预测增益>=gamma则选e3，检查不卡顿（如果卡顿则按增益降序去掉一些）"""
    mask = pred_gain >= gamma
    if can_play(mask, B):
        return mask
    # 卡顿了，按预测增益降序去掉一些帧直到不卡顿
    sorted_idx = np.argsort(-pred_gain)
    for idx in sorted_idx:
        if mask[idx]:
            mask[idx] = False
            if can_play(mask, B):
                break
    return mask

def ca_strategy(pred_gain, B, safe_factor=1.5, gain_thresh=1.5):
    """内容感知CA策略（模拟播放器中的halt控制器）"""
    budget = float(B)
    mask = np.zeros(N, dtype=bool)
    for i in range(N):
        budget += DT
        if budget >= D3 * safe_factor:
            mask[i] = True
            budget -= D3
        elif budget >= D3 and pred_gain[i] >= gain_thresh:
            mask[i] = True
            budget -= D3
        else:
            budget -= D2
    return mask

def quality(mask):
    """总质量"""
    return np.sum(e2) + np.sum(delta23[mask])

# ============================================================
# 实验1：预测误差敏感性
# ============================================================
print("=" * 70)
print("实验1：预测误差敏感性")
print("=" * 70)

B = 500  # 初始缓冲区
opt_mask = offline_opt(B)
opt_q = quality(opt_mask)
print(f"离线最优: e3={opt_mask.sum()}/{N} ({opt_mask.sum()/N*100:.1f}%), 质量={opt_q:.2f}")
print()

sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
n_trials = 30  # 每个sigma跑30次取平均

print(f"{'sigma':>6} {'阈值策略竞争比':>14} {'CA策略竞争比':>14} {'阈值e3比例':>12} {'CA e3比例':>12}")
print("-" * 65)

results_noise = []
for sigma in sigmas:
    thresh_ratios = []
    ca_ratios = []
    thresh_e3s = []
    ca_e3s = []
    for trial in range(n_trials):
        if sigma == 0:
            pred = delta23.copy()
        else:
            pred = delta23 + np.random.normal(0, sigma, N)
        # 阈值策略，gamma=0.65（之前验证的最优阈值）
        t_mask = threshold_strategy(pred, gamma=0.65, B=B)
        thresh_ratios.append(quality(t_mask) / opt_q)
        thresh_e3s.append(t_mask.sum() / N)
        # CA策略
        c_mask = ca_strategy(pred, B, safe_factor=1.5, gain_thresh=1.5)
        ca_ratios.append(quality(c_mask) / opt_q)
        ca_e3s.append(c_mask.sum() / N)
    print(f"{sigma:>6.1f} {np.mean(thresh_ratios):>14.4f} {np.mean(ca_ratios):>14.4f} {np.mean(thresh_e3s)*100:>11.1f}% {np.mean(ca_e3s)*100:>11.1f}%")
    results_noise.append({
        'sigma': sigma,
        'thresh_ratio_mean': float(np.mean(thresh_ratios)),
        'thresh_ratio_std': float(np.std(thresh_ratios)),
        'ca_ratio_mean': float(np.mean(ca_ratios)),
        'ca_ratio_std': float(np.std(ca_ratios)),
        'thresh_e3_mean': float(np.mean(thresh_e3s)),
        'ca_e3_mean': float(np.mean(ca_e3s)),
    })

print()
print("关键观察：")
print(f"  sigma=0（完美预测）: 阈值竞争比={results_noise[0]['thresh_ratio_mean']:.4f}, CA={results_noise[0]['ca_ratio_mean']:.4f}")
print(f"  sigma=0.3（段级预测典型误差）: 阈值={results_noise[3]['thresh_ratio_mean']:.4f} (损失{(1-results_noise[3]['thresh_ratio_mean'])*100:.2f}%), CA={results_noise[3]['ca_ratio_mean']:.4f}")
print(f"  sigma=1.0（较大误差）: 阈值={results_noise[6]['thresh_ratio_mean']:.4f} (损失{(1-results_noise[6]['thresh_ratio_mean'])*100:.2f}%), CA={results_noise[6]['ca_ratio_mean']:.4f}")

# ============================================================
# 实验2：缓冲区大小与最大可持续e3比例
# ============================================================
print()
print("=" * 70)
print("实验2：缓冲区大小与最大可持续e3比例")
print("=" * 70)

buffers = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000]
alpha_max_steady = (DT - D2) / (D3 - D2)  # 稳态上界

print(f"稳态上界 alpha_max = {alpha_max_steady:.4f} ({alpha_max_steady*100:.1f}%)")
print()
print(f"{'B(ms)':>8} {'离线最优e3%':>12} {'离线竞争比':>12} {'Greedy e3%':>12} {'CA e3%':>12} {'CA竞争比':>12}")
print("-" * 75)

results_buffer = []
for B in buffers:
    opt_m = offline_opt(B)
    opt_q_B = quality(opt_m)
    # Greedy
    budget = float(B); g_mask = np.zeros(N, dtype=bool)
    for i in range(N):
        budget += DT
        if budget >= D3: g_mask[i] = True; budget -= D3
        else: budget -= D2
    # CA
    c_mask = ca_strategy(delta23, B, safe_factor=1.5, gain_thresh=1.5)

    print(f"{B:>8} {opt_m.sum()/N*100:>11.1f}% {1.0:>12.4f} {g_mask.sum()/N*100:>11.1f}% {c_mask.sum()/N*100:>11.1f}% {quality(c_mask)/opt_q_B:>12.4f}")
    results_buffer.append({
        'B': B,
        'opt_e3_ratio': float(opt_m.sum()/N),
        'opt_quality': float(opt_q_B),
        'greedy_e3_ratio': float(g_mask.sum()/N),
        'ca_e3_ratio': float(c_mask.sum()/N),
        'ca_ratio': float(quality(c_mask)/opt_q_B),
    })

print()
print("关键观察：")
print(f"  B=0（无缓冲）: 离线最优e3={results_buffer[0]['opt_e3_ratio']*100:.1f}% (稳态上界{alpha_max_steady*100:.1f}%)")
print(f"  B=500: 离线最优e3={results_buffer[5]['opt_e3_ratio']*100:.1f}% (超调{(results_buffer[5]['opt_e3_ratio']-alpha_max_steady)*100:.1f}%)")
print(f"  B=5000: 离线最优e3={results_buffer[-1]['opt_e3_ratio']*100:.1f}%")
print(f"  CA策略在所有B下竞争比 >= {min(r['ca_ratio'] for r in results_buffer):.4f}")

# 保存结果
out = {
    'noise_sensitivity': results_noise,
    'buffer_analysis': results_buffer,
    'alpha_max_steady': float(alpha_max_steady),
    'N': N,
    'params': {'DT': DT, 'D2': D2, 'D3': D3, 'n_trials': n_trials},
}
json.dump(out, open(r'E:\dev\Code\Python\bones\workspace\pacnet\theory_validation.json', 'w'), indent=2)
print()
print("结果已保存到 workspace/pacnet/theory_validation.json")
