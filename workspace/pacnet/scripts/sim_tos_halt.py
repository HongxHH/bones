# -*- coding: utf-8 -*-
"""
阶段一：TOS 离线 halt 调度仿真。
隔离"增强算力调度"这一个变量，用 TOS 真实三出口延迟与 VMAF，
扫描算力紧张系数 k，对比三种策略在不同 k 下的掉帧率与有效 VMAF。

模型（离散时间，单线程增强器）：
- 播放端：24fps，帧间隔 41.67ms，第 i 帧在 i*dt 时刻必须可播。
- 增强端：单线程顺序增强，每帧选一个出口 e，耗时 l_e * k。
- lead = 增强完成时刻 - 该帧播放时刻（>0 提前做好，<0 晚了）。
- 初始增强器领先 init_buffer（预缓冲）开始工作。
- 若增强完第 i 帧时已超过其播放时刻+一帧间隔，则该帧 stall（掉帧，用 base VMAF）。

策略：
- e3_only：每帧 PAC e3（死磕画质，不管预算）
- imdn_l2：每帧 IMDN 独立 l2 模型（基线近似，中等画质+快）
- pac_halt：按 lead 自适应选 e1/e2/e3（预算够深、紧张浅）
"""
import numpy as np
from collections import Counter

# ---- TOS 实测数值（240p，毫秒）----
DT = 1000.0 / 24.0          # 帧间隔 41.67ms
N_FRAMES = 184 * 96         # 17664 帧
INIT_BUFFER = 2000.0        # 预缓冲 2s（增强器领先播放开始）

# PAC 三出口延迟 ms（measure_latency_tos）
L_PAC = {1: 11.64, 2: 24.19, 3: 33.79}
# PAC 三出口 VMAF（tos 240p）
V_PAC = {0: 80.43, 1: 86.18, 2: 87.12, 3: 90.22}
# IMDN 基线 l2（独立中等模型）：延迟 5.1ms（实测fps197），VMAF 88.70
L_IMDN_L2 = 5.1
V_IMDN_L2 = 88.70


def simulate(strategy, k, verbose=False):
    B = INIT_BUFFER          # 增强缓冲（毫秒视频量），初始预充 2s
    stalls = 0
    vmaf_sum = 0.0
    usage = Counter()
    l1, l2, l3 = L_PAC[1]*k, L_PAC[2]*k, L_PAC[3]*k
    for i in range(N_FRAMES):
        # 选出口（halt 依据当前缓冲 B：厚则深、薄则浅）
        if strategy == 'e3_only':
            e, lat, vq = 3, l3, V_PAC[3]
        elif strategy == 'imdn_l2':
            e, lat, vq = 2, L_IMDN_L2*k, V_IMDN_L2
        else:  # pac_halt
            if B >= l3 * 1.5:
                e, lat, vq = 3, l3, V_PAC[3]
            elif B >= l2 * 1.3:
                e, lat, vq = 2, l2, V_PAC[2]
            else:
                e, lat, vq = 1, l1, V_PAC[1]
        usage[e] += 1
        # 处理一帧耗时 lat ms：这期间播放消费 lat ms 视频，增强补回一帧=DT ms
        B += DT - lat
        # 缓冲透支 = 该播的帧还没增强完 → 掉帧
        if B < 0:
            stalls += 1
            vmaf_sum += V_PAC[0]   # 掉帧用 base 质量
        else:
            vmaf_sum += vq
    stall_rate = stalls / N_FRAMES * 100
    avg_vmaf = vmaf_sum / N_FRAMES
    return stalls, stall_rate, avg_vmaf, dict(usage)


print('='*92)
print('TOS 离线 halt 调度仿真（24fps，184段x96帧=17664帧，预缓冲2s）')
print('策略：e3_only=PAC全深 / imdn_l2=基线中等模型 / pac_halt=PAC自适应降档')
print('k=算力紧张系数（k=1满血GPU，越大越弱/越紧张）')
print('='*92)
print(f"{'k':>5} {'策略':>10} {'掉帧帧':>8} {'掉帧%':>8} {'有效VMAF':>10}  出口分布")
print('-'*92)

results = {}
for k in [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]:
    results[k] = {}
    for strat in ['e3_only', 'imdn_l2', 'pac_halt']:
        s, sr, av, use = simulate(strat, k)
        results[k][strat] = (sr, av)
        print(f"{k:>5} {strat:>10} {s:>8} {sr:>7.2f}% {av:>10.3f}  {use}")
    # 该 k 下 halt 相对两个固定策略的优劣
    h = results[k]['pac_halt']
    e3 = results[k]['e3_only']
    l2 = results[k]['imdn_l2']
    note = []
    if h[0] < e3[0]: note.append(f"比全e3少掉{ e3[0]-h[0]:.1f}%")
    if h[1] > l2[1]: note.append(f"比固定l2高VMAF+{h[1]-l2[1]:.2f}")
    if note:
        print(f"      >>> halt 在 k={k} 的优势: " + "; ".join(note))
    print('-'*92)

def simulate_halt_tuned(k, safe_e3, safe_e2):
    """可调阈值的 halt：B >= l3*safe_e3 -> e3; B >= l2*safe_e2 -> e2; else e1"""
    B = INIT_BUFFER
    stalls = 0
    vmaf_sum = 0.0
    usage = Counter()
    l1, l2, l3 = L_PAC[1]*k, L_PAC[2]*k, L_PAC[3]*k
    for i in range(N_FRAMES):
        if B >= l3 * safe_e3:
            e, lat, vq = 3, l3, V_PAC[3]
        elif B >= l2 * safe_e2:
            e, lat, vq = 2, l2, V_PAC[2]
        else:
            e, lat, vq = 1, l1, V_PAC[1]
        usage[e] += 1
        B += DT - lat
        if B < 0:
            stalls += 1
            vmaf_sum += V_PAC[0]
        else:
            vmaf_sum += vq
    return stalls / N_FRAMES * 100, vmaf_sum / N_FRAMES, usage[3] / N_FRAMES * 100


print()
print('='*92)
print('halt 阈值调优扫描（在紧张档 k=1.5 / 2.0）：能否做到 0掉帧 且 VMAF > 固定l2(88.70)')
print('='*92)
print(f"{'k':>4} {'safe_e3':>8} {'safe_e2':>8} {'掉帧%':>8} {'有效VMAF':>10} {'e3占比%':>9}  判断")
for k in [1.5, 2.0]:
    for se3 in [1.2, 1.5, 2.0, 2.5, 3.0]:
        for se2 in [1.0, 1.3, 1.6]:
            sr, av, p3 = simulate_halt_tuned(k, se3, se2)
            ok = '✓ 0掉帧且超l2' if (sr < 0.01 and av > 88.70) else ''
            print(f"{k:>4} {se3:>8} {se2:>8} {sr:>7.2f}% {av:>10.3f} {p3:>8.1f}%  {ok}")
    print('-'*92)

