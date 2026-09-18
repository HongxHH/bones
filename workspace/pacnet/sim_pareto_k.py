# -*- coding: utf-8 -*-
"""
全算力范围帕累托检验（固定 repr0，下载 0 卡顿，0921）。
对比 PAC 帧级 halt / PAC 固定档 / IMDN 帧级 oracle 选档（IMDN 自适应上界）/ IMDN 固定 l2，
扫恒定 K=1..6，确认是否存在任何算力区间让 anytime halt 赢最强 IMDN 基线。
"""
import numpy as np
from collections import Counter
from sim_download_counterfactual import load_seg_bytes, ready_timeline, total_stall, TRACES, N_SEG
from sim_joint_bitrate_enhance import play_timeline, DT, FPF

seg_bytes = load_seg_bytes()
ready = ready_timeline([0]*N_SEG, seg_bytes, TRACES['0921'])
play = play_timeline(ready)

# 满血单帧延迟 ms / repr0 质量
PAC_L = {1: 11.64, 2: 24.19, 3: 33.79}
PAC_V = {1: 86.18, 2: 87.12, 3: 90.22}
IMDN_L = {1: 3.6, 2: 5.1, 3: 9.8}
IMDN_V = {1: 87.02, 2: 88.70, 3: 89.64}
BASE = 80.43

def run(strategy, K):
    pl = {e: PAC_L[e]*K for e in PAC_L}
    il = {e: IMDN_L[e]*K for e in IMDN_L}
    enh = ready[0]; q=[]; ec=Counter(); fc=0
    for i in range(N_SEG):
        for f in range(FPF):
            fp = play[i]+f*DT/1000.0
            if f==0: enh=max(enh, ready[i])
            lead=(fp-enh)*1000.0; rem=FPF-f
            if strategy=='pac_halt':
                e = 3 if lead>=pl[3]*rem else 2 if lead>=pl[2]*rem else 1 if lead>=pl[1]*rem else 0
                if e==0: q.extend([BASE]*rem); fc+=rem; break
                enh+=pl[e]/1000; ec[e]+=1; q.append(PAC_V[e])
            elif strategy=='imdn_oracle':
                e = 3 if lead>=il[3]*rem else 2 if lead>=il[2]*rem else 1 if lead>=il[1]*rem else 0
                if e==0: q.extend([BASE]*rem); fc+=rem; break
                enh+=il[e]/1000; ec[e]+=1; q.append(IMDN_V[e])
            elif strategy.startswith('pac_fix'):
                e=int(strategy[-1])
                if lead>=pl[e]*rem: enh+=pl[e]/1000; ec[e]+=1; q.append(PAC_V[e])
                else: q.extend([BASE]*rem); fc+=rem; break
            elif strategy=='imdn_l2':
                if lead>=il[2]*rem: enh+=il[2]/1000; q.append(IMDN_V[2])
                else: q.extend([BASE]*rem); fc+=rem; break
    return float(np.mean(q)), fc/(N_SEG*FPF)*100, ec

strats=['pac_halt','imdn_oracle','imdn_l2','pac_fix3','pac_fix2']
print('固定 repr0（0 下载卡顿），全 K 扫描：有效 VMAF（fc帧%）')
print('='*88)
print(f"{'K':>4} | {'PAC halt':>16} | {'IMDN oracle':>16} | {'IMDN 固定l2':>14} | halt-oracle")
print('-'*88)
for K in [1.0,1.2,1.5,1.8,2.0,2.5,3.0,4.0,5.0,6.0]:
    r={}
    for s in strats:
        v,fc,ec=run(s,K); r[s]=(v,fc,ec)
    ph=r['pac_halt'][0]; io=r['imdn_oracle'][0]
    print(f"{K:>4.1f} | {ph:7.2f} ({r['pac_halt'][1]:4.1f}%) | {io:7.2f} ({r['imdn_oracle'][1]:4.1f}%) | "
          f"{r['imdn_l2'][0]:7.2f} ({r['imdn_l2'][1]:4.1f}%) | {ph-io:+.2f}")
print('-'*88)
# K=1 满血时 halt 出口分布（算力充足是否全程 e3）
v,fc,ec=run('pac_halt',1.0)
print('K=1.0 PAC halt 出口分布:', dict(ec), 'fc帧%', round(fc,2))
v,fc,ec=run('imdn_oracle',1.0)
print('K=1.0 IMDN oracle 出口分布:', dict(ec))
