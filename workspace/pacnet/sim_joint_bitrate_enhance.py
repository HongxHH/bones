# -*- coding: utf-8 -*-
"""
下载-增强联合反事实仿真（0921，K=1.5）。

在 sim_download_counterfactual 的下载时间线上，叠加单线程帧级 halt 增强库存：
- 段 i 下载完成时刻 ready_dl[i]（trace 限速串行下载，增强在旁路并行、不拖慢下载，
  已由同码率段配对 +3s 证实）。
- 播放时刻 play[i] 由下载 ready 递推（零起播缓冲），卡顿只来自下载。
- 增强器按段顺序、段 i 不早于 ready_dl[i] 开始；帧级 halt 可调度性前瞻
  lead_ms >= l_exit*remaining_frames 选最高出口，连 e1 都不可行则 fast_complete
  剩余帧按 base（复刻 pacnet_enhancer._halt_decide，安全系数 1.0）。
质量用全片平均质量表（出口常量 VMAF），先看趋势与角点。
"""
import json, os, math
from collections import Counter
import numpy as np
from sim_download_counterfactual import (
    load_seg_bytes, TraceClock, ready_timeline, total_stall,
    real_repr_seq, TRACES, PLAYER, N_SEG)

DT = 1000.0/24.0
FPF = 96
# K=1.5 乘法后单帧延迟 ms
L = {1: 11.64*1.5, 2: 24.19*1.5, 3: 33.79*1.5}
V = {0: 80.43, 1: 86.18, 2: 87.12, 3: 90.22}
L_IMDN_L2 = 5.1*1.5
V_IMDN_L2 = 88.70

def play_timeline(ready, seg=4.0):
    play = [ready[0]]
    for i in range(1, len(ready)):
        play.append(max(ready[i], play[-1]+seg))
    return play

def halt_enhance(ready, play, safe=(1.0, 1.0), verbose=False):
    """返回每帧质量列表、出口计数、fc 帧数、增强完成晚于播放点的帧数。"""
    se3, se2 = safe
    enh = ready[0]            # 增强器墙钟，第一段下载完才开始
    quality = []
    exit_cnt = Counter()
    fc_frames = 0
    late_frames = 0
    fc_segs = 0
    for i in range(len(ready)):
        seg_fc = False
        for f in range(FPF):
            frame_play = play[i] + f*DT/1000.0
            if f == 0:
                enh = max(enh, ready[i])
            lead_ms = (frame_play - enh)*1000.0
            remaining = FPF - f
            # 逐级选最高可行出口
            e = 0
            if lead_ms >= L[3]*remaining*se3:
                e = 3
            elif lead_ms >= L[2]*remaining*se2:
                e = 2
            elif lead_ms >= L[1]*remaining:
                e = 1
            else:
                # fast_complete：剩余帧全部 base
                fc_frames += (FPF-f)
                seg_fc = True
                quality.extend([V[0]]*(FPF-f))
                break
            enh += L[e]/1000.0
            if enh > frame_play:
                late_frames += 1
            exit_cnt[e] += 1
            quality.append(V[e])
        if seg_fc:
            fc_segs += 1
    return quality, exit_cnt, fc_frames, late_frames, fc_segs

def imdn_enhance(ready, play):
    """IMDN 段级：增强近瞬时(l2)，假设全帧 l2（质量 88.70），段下载完即可增强完。"""
    quality = []
    late = 0
    for i in range(len(ready)):
        need = L_IMDN_L2*FPF/1000.0
        start = ready[i]
        if start + need > play[i]:
            late += 1
        quality.extend([V_IMDN_L2]*FPF)
    return quality, late

def report(name, ready, quality, stall, extra=''):
    vmaf = float(np.mean(quality))
    # 粗略 QoE：VMAF - 卡顿总时长/10（固定码率无切换，oscillation≈0）
    qoe = vmaf - stall/10.0
    print(f'{name:>22}: 卡顿 {stall:6.1f}s | VMAF {vmaf:6.2f} | 粗QoE {qoe:6.2f} {extra}')
    return vmaf, qoe

seg_bytes = load_seg_bytes()
tag = '0921'
trace_fn = TRACES[tag]

print('='*100)
print(f'联合反事实 {tag} K=1.5（固定码率无 oscillation；粗QoE=VMAF-卡顿/10）')
print('='*100)

# 校准：真实 halt 序列
halt_fn = 'bones_pacnet_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-121806.json'
d, seq = real_repr_seq(halt_fn)
seq = [0 if x is None else x for x in seq]
ready = ready_timeline(seq, seg_bytes, trace_fn)
stall, _ = total_stall(ready)
play = play_timeline(ready)
q, ec, fc, late, fc_segs = halt_enhance(ready, play)
dist = f'e1{e1p:3.0f}/e2{e2p:3.0f}/e3{e3p:3.0f}%' if False else ''
e1p = ec[1]/sum(ec.values())*100 if sum(ec.values()) else 0
e2p = ec[2]/sum(ec.values())*100 if sum(ec.values()) else 0
e3p = ec[3]/sum(ec.values())*100 if sum(ec.values()) else 0
print(f'[校准] 真实 halt：真实卡顿84.5/真实VMAF89.13/fc段15；仿真卡顿{stall:.1f} fc段{fc_segs} '
      f'出口 e1{e1p:.0f}/e2{e2p:.0f}/e3{e3p:.0f}% fc帧{fc} 晚成帧{late}')
report('  真实BONES序列+halt', ready, q, stall)
print('-'*100)

# 反事实：固定各码率 + halt
for r in [0, 1]:
    seqr = [r]*N_SEG
    rd = ready_timeline(seqr, seg_bytes, trace_fn)
    st, _ = total_stall(rd)
    pl = play_timeline(rd)
    qh, ech, fch, lateh, fcsh = halt_enhance(rd, pl)
    tot = sum(ech.values())
    dist = f"出口 e1{ech[1]/tot*100:.0f}/e2{ech[2]/tot*100:.0f}/e3{ech[3]/tot*100:.0f}% fc段{fcsh} fc帧{fch}"
    report(f'固定repr{r}+halt', rd, qh, st, dist)
    # 同码率 IMDN
    qi, _ = imdn_enhance(rd, pl)
    report(f'固定repr{r}+IMDNl2', rd, qi, st)
    print()

# 角点参照（满血 e3，不可持续上界）
print(f'角点参照：固定repr0+满血e3 卡顿0 VMAF上界 {V[3]:.2f}；固定repr0+IMDNe3档质量89.64')
