# -*- coding: utf-8 -*-
"""
时变算力下 anytime halt vs 固定档（固定 repr0，下载瓶颈已消除，0921 trace）。

动机：anytime 多出口的价值是应对"随机/时变算力预算"。恒定 K 把问题退化成恒定弱算力，
一个固定快档（IMDN l2）就够，halt 无优势。本脚本让 K 随段波动（模拟多任务抢占/热降频），
对比帧级 halt 与各固定档的有效 VMAF / fast_complete 掉帧。

下载侧固定 repr0（trace 0921 下卡顿 0，见 sim_joint），增强器在该时间线上工作。
K(t) 以段（4s）为粒度变化；halt 帧级实时按 lead 选档（天然跟踪瞬时算力），
固定档不管 K 如何都硬跑同一出口，lead 不足则 fast_complete 剩余帧按 base。
"""
import numpy as np
from collections import Counter
from sim_download_counterfactual import load_seg_bytes, ready_timeline, total_stall, TRACES, N_SEG
from sim_joint_bitrate_enhance import play_timeline, DT, FPF, L, V, L_IMDN_L2, V_IMDN_L2

seg_bytes = load_seg_bytes()
ready = ready_timeline([0]*N_SEG, seg_bytes, TRACES['0921'])  # 固定 repr0
play = play_timeline(ready)
stall, _ = total_stall(ready)
assert stall < 1e-6, stall

def k_trajectories(kind, seed):
    rng = np.random.default_rng(seed)
    K = np.ones(N_SEG)
    if kind == 'const1.5':
        K[:] = 1.5
    elif kind == 'slow_mkv':  # 慢变马尔可夫：负载以 10~30 段为周期切换
        levels = [1.0, 1.3, 1.6, 2.0]
        i = 0; k = 1.3
        while i < N_SEG:
            dur = int(rng.integers(8, 26))
            k = levels[(levels.index(k) + rng.choice([-1, 0, 1])) % len(levels)] if i > 0 else 1.3
            K[i:i+dur] = k; i += dur
    elif kind == 'iid':       # 逐段 i.i.d.
        K = rng.uniform(1.0, 2.0, N_SEG)
    elif kind == 'burst':     # 平时 1.1，20% 段突发重负载 2.2
        K[:] = 1.1
        K[rng.random(N_SEG) < 0.20] = 2.2
    return K

def run(strategy, Kseg, safe=(1.0, 1.0)):
    se3, se2 = safe
    enh = ready[0]
    q = []; ec = Counter(); fc = 0
    for i in range(N_SEG):
        k = Kseg[i]
        l = {e: L[e]*k for e in (1, 2, 3)}
        limdn = L_IMDN_L2*k
        for f in range(FPF):
            frame_play = play[i] + f*DT/1000.0
            if f == 0:
                enh = max(enh, ready[i])
            lead = (frame_play - enh)*1000.0
            rem = FPF - f
            if strategy == 'halt':
                if lead >= l[3]*rem*se3: e = 3
                elif lead >= l[2]*rem*se2: e = 2
                elif lead >= l[1]*rem: e = 1
                else:
                    q.extend([V[0]]*rem); fc += rem; break
                enh += l[e]/1000.0; ec[e] += 1; q.append(V[e])
            elif strategy.startswith('pac_e'):
                e = int(strategy[-1]); lat = l[e]
                if lead >= lat*rem:
                    enh += lat/1000.0; ec[e] += 1; q.append(V[e])
                else:
                    q.extend([V[0]]*rem); fc += rem; break
            elif strategy == 'imdn_l2':
                if lead >= limdn*rem:
                    enh += limdn/1000.0; q.append(V_IMDN_L2)
                else:
                    q.extend([V[0]]*rem); fc += rem; break
    return float(np.mean(q)), fc/(N_SEG*FPF)*100, ec

strats = ['halt', 'pac_e3', 'pac_e2', 'imdn_l2']
scenarios = ['const1.5', 'slow_mkv', 'iid', 'burst']
print('固定 repr0（下载 0 卡顿），时变算力 K(t)，各策略有效 VMAF / fc帧%（10 种子均值）')
print('='*94)
for sc in scenarios:
    agg = {s: [] for s in strats}; fcp = {s: [] for s in strats}
    e3share = []
    for seed in range(10):
        Kseg = k_trajectories(sc, seed)
        for s in strats:
            v, fc, ec = run(s, Kseg)
            agg[s].append(v); fcp[s].append(fc)
            if s == 'halt':
                tot = sum(ec.values()); e3share.append(ec[3]/tot*100 if tot else 0)
    print(f'[{sc:>9}] 平均K={np.mean([k_trajectories(sc,sd) for sd in range(10)]):.2f}  '
          f'halt e3帧占比 {np.mean(e3share):.0f}%')
    for s in strats:
        print(f'    {s:>9}: VMAF {np.mean(agg[s]):6.2f}±{np.std(agg[s]):.2f}  fc帧 {np.mean(fcp[s]):5.1f}%')
    hv = np.mean(agg['halt'])
    best_fixed = max(np.mean(agg['pac_e3']), np.mean(agg['pac_e2']), np.mean(agg['imdn_l2']))
    print(f'    >>> halt 相对最佳固定档 VMAF {hv-best_fixed:+.2f}')
    print('-'*94)
