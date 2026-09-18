# -*- coding: utf-8 -*-
import io
p = r'E:\dev\Code\Python\bones\docs\pacnet_progressive_enhancer.md'
t = io.open(p, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in t else '\n'
s = t.index('- 下一步先用含真实下载时间线')
e = t.index('系数 1.2 可作默认捎带。', s) + len('系数 1.2 可作默认捎带。')
new = ('- 含下载时间线的反事实（2026-09-18，脚本 analyze_loop_blocking / sim_download_counterfactual / '
'sim_joint_bitrate_enhance / sim_pareto_k / sim_variable_compute，明细见 current_state.md）已定位杠杆并给出结构性判决：'
'①增强与下载基本并行（配对同码率段下载墙钟仅 +3s），卡顿唯一硬杠杆是码率下载量（固定 repr0 卡顿 0、repr1 卡 512s、repr2 卡 1096s）；'
'②固定 repr0+halt 卡顿 0、VMAF 88.79（当前 89.13 但卡 84.5s），但仅比 IMDN 固定 l2 高 0.09；'
'③全 K 帕累托检验决定性负面——halt 仅 K≤1.2（全程 e3、anytime 不换档）赢 IMDN 0.58，K≥1.5 紧张区（anytime 本该发力）反输 0.8~6.4 且越紧张越崩，时变算力同样输。'
'根因是共享 nf=32 骨干的浅出口 e1(11.64ms,86.18)/e2(24.19,87.12) 被专用轻模型 IMDN_RTC l2(5.1ms,88.70) 严格 dominate，e0 更浅也仍被压制。'
'anytime 在轻量 CNN、小倍率单帧超分的低算力甜区无帕累托空间；要成立需换到轻模型不够用、深浅质量差显著拉大的设定（4x / 强混合退化 / 视频时序 VSR），下一步先离线测轻重模型质量差再决定是否改设定重训。')
out = t[:s] + new + t[e:]
io.open(p, 'w', encoding='utf-8', newline='').write(out)
print('done, nl', repr(nl))
