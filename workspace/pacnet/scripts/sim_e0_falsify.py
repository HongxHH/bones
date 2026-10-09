# -*- coding: utf-8 -*-
"""
轻量保命档 e0 离线证伪（在 sim_tos_halt.py 库存模型上加第 4 个更快出口）。

问题：阶段二实测 halt 卡顿追不平 IMDN，安全系数扫描证明主瓶颈是最浅档 e1
太慢（K=1.5 后 17.5ms，IMDN 主力 l2 约 7.7ms）。本脚本只隔离"增强算力供给"
这一个变量（与阶段一相同，不含网络下载），回答：
  若新增一个比 e1 更早/更轻的出口 e0（延迟 L0、质量 V0），halt 能否在 k=1.5
  及更紧算力下把掉帧率压到 0（追平 IMDN l2 的流畅度），同时厚库存吃 e3 红利、
  平均 VMAF 仍高于固定 l2(88.70)？

模型与 sim_tos_halt.py 完全一致：B 为增强缓冲(ms 视频量)，初始预充 2s；
每帧选档耗时 lat，B += DT - lat；B<0 记一帧掉帧（按 base 质量）。
e0 是"能实时完成的低质量增强"，不是掉帧：最薄库存时选 e0，lat<DT 即可回血。
"""
import numpy as np
from collections import Counter

DT = 1000.0 / 24.0
N_FRAMES = 184 * 96
INIT_BUFFER = 2000.0

# TOS 240p 实测
L_PAC = {1: 11.64, 2: 24.19, 3: 33.79}     # ms
V_PAC = {0: 80.43, 1: 86.18, 2: 87.12, 3: 90.22}
L_IMDN_L2 = 5.1
V_IMDN_L2 = 88.70


def run(strategy, k, l0_ms=None, v0=84.0, safe=1.0):
    """strategy: e3_only / imdn_l2 / halt_e1(现状,最浅e1) / halt_e0(加虚拟e0)。"""
    B = INIT_BUFFER
    stalls = 0
    vsum = 0.0
    use = Counter()
    l1, l2, l3 = L_PAC[1]*k, L_PAC[2]*k, L_PAC[3]*k
    l0 = (l0_ms * k) if l0_ms is not None else None
    for _ in range(N_FRAMES):
        if strategy == 'e3_only':
            e, lat, vq = 3, l3, V_PAC[3]
        elif strategy == 'imdn_l2':
            e, lat, vq = 2, L_IMDN_L2*k, V_IMDN_L2
        elif strategy == 'halt_e1':
            if B >= l3*safe:
                e, lat, vq = 3, l3, V_PAC[3]
            elif B >= l2*safe:
                e, lat, vq = 2, l2, V_PAC[2]
            else:
                e, lat, vq = 1, l1, V_PAC[1]
        elif strategy == 'halt_e0':
            if B >= l3*safe:
                e, lat, vq = 3, l3, V_PAC[3]
            elif B >= l2*safe:
                e, lat, vq = 2, l2, V_PAC[2]
            elif B >= l1*safe:
                e, lat, vq = 1, l1, V_PAC[1]
            else:
                e, lat, vq = 0, l0, v0
        use[e] += 1
        B += DT - lat
        if B < 0:
            stalls += 1
            vsum += V_PAC[0]
        else:
            vsum += vq
    sr = stalls / N_FRAMES * 100
    av = vsum / N_FRAMES
    pct = {e: round(100*use[e]/N_FRAMES, 1) for e in sorted(use)}
    return sr, av, pct


print('='*104)
print('e0 证伪：TOS 24fps 纯增强库存模型，预缓冲2s。l0 列为 e0 满血延迟(ms)，K=1.5 后实际=l0*k')
print('对照：e3_only / imdn_l2(固定,7.65ms@1.5,VMAF88.70) / halt_e1(现状最浅17.5ms) / halt_e0(各 l0)')
print('='*104)

for k in [1.2, 1.5, 1.8, 2.0]:
    print(f"\n--- k={k} ---")
    print(f"{'策略':<16}{'l0满血':>7}{'加压后':>7}{'掉帧%':>9}{'有效VMAF':>10}   出口占比 e0/e1/e2/e3")
    sr, av, p = run('e3_only', k)
    print(f"{'e3_only':<16}{'-':>7}{'-':>7}{sr:>9.2f}{av:>10.3f}   {p}")
    sr, av, p = run('imdn_l2', k)
    print(f"{'imdn_l2(基线)':<16}{'5.1':>7}{5.1*k:>7.1f}{sr:>9.2f}{av:>10.3f}   l2固定")
    sr, av, p = run('halt_e1', k)
    print(f"{'halt_e1(现状)':<16}{'-':>7}{'-':>7}{sr:>9.2f}{av:>10.3f}   {p}")
    # l0 为满血延迟；5.3 加压后≈8ms（对齐 IMDN l2 加压 7.65），11.64 加压后≈e1(17.5) 作对照
    for l0 in [3.5, 5.3, 7.7, 11.64]:
        sr, av, p = run('halt_e0', k, l0_ms=l0, v0=84.0)
        tag = '  <-- 目标(加压≈8ms)' if abs(l0-5.3) < 1e-6 else ('  (=e1对照)' if abs(l0-11.64) < 1e-6 else '')
        print(f"{'halt_e0':<16}{l0:>7.1f}{l0*k:>7.1f}{sr:>9.2f}{av:>10.3f}   {p}{tag}")

# e0 质量敏感性（满血 l0=5.3ms→加压≈8ms，k=1.5）：V0 在 base~e1 之间
print('\n' + '='*70)
print('e0 质量敏感性（k=1.5，l0满血5.3ms/加压8ms，V0 从 base80.43 到 e1=86.18）')
for v0 in [80.43, 82.0, 84.0, 86.18]:
    sr, av, p = run('halt_e0', 1.5, l0_ms=5.3, v0=v0)
    print(f"  V0={v0:>6.2f}: 掉帧 {sr:>6.2f}%  有效VMAF={av:.3f}  占比={p}")

# 安全系数敏感性（l0满血5.3ms,V0=84,k=1.5）
print('\n' + '='*70)
print('halt_e0 安全系数敏感性（k=1.5，l0满血5.3ms/加压8ms，V0=84）')
for sf in [1.0, 1.2, 1.5]:
    sr, av, p = run('halt_e0', 1.5, l0_ms=5.3, v0=84.0, safe=sf)
    print(f"  safe={sf}: 掉帧 {sr:>6.2f}%  有效VMAF={av:.3f}  占比={p}")
