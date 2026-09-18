# -*- coding: utf-8 -*-
import io, os
p = r'E:\dev\Code\Python\bones\docs\current_state.md'
text = io.open(p, encoding='utf-8').read()
start_anchor = '**下一步（主推荐）：e0 不做'
end_anchor = '不单独作结论。'
s = text.index(start_anchor)
e = text.index(end_anchor, s) + len(end_anchor)

new = '''**含下载时间线的反事实仿真（2026-09-18，`workspace/pacnet/`，结论性）。**

1. 事件循环归因（`analyze_loop_blocking.py`）：增强在 asyncio 单线程内同步阻塞，但与下载基本并行——配对到相同码率的段（halt vs IMDN 155 段）下载墙钟仅差 +3.1s（中位 −0.02s），增强多花 368s 却几乎没拉长同码率下载（在下载 chunk sleep 间隙搭车）。halt 多卡 15.5s 主要因差网选了更多 repr1（下载字节多），不是增强抢 GPU。
2. 下载反事实（`sim_download_counterfactual.py`，严格复刻 `trace.py` 分块限速）：用三策略真实码率序列复现卡顿 65.7/81.3/60.9s（真实 84.5/102/69，系统性低估 15-20s，未建模增强关键路径/解码/起播，相对顺序一致）。固定码率：**全程 repr0（400k）卡顿 0（下载墙钟 615s < 视频 736s）；repr1 卡 512s、repr2 卡 1096s**。卡顿唯一硬杠杆是码率下载量，BONES 在烂网仍选 30+ 段 repr1 是卡顿根源。
3. 联合仿真（`sim_joint_bitrate_enhance.py`，下载时间线 + 帧级 halt 增强库存，K=1.5）：固定 repr0+halt 卡顿 0、VMAF 88.79（真实 halt 89.13 但卡 84.5s），质量仅低 0.34、粗 QoE（VMAF−卡顿/10，固定码率无切换）反超约 2；但固定 repr0 下 halt(88.79) 仅比 IMDN 固定 l2(88.70) 高 0.09。
4. **全 K 帕累托检验（`sim_pareto_k.py`，决定性负面）**：固定 repr0 消除下载瓶颈后扫 K=1..6，halt 对 IMDN 帧级 oracle 选档（其自适应上界）——K≤1.2 时 halt 全程 e3、VMAF 90.17 赢 0.58，但 anytime 不换档（退化为固定深模型）；**K≥1.5 紧张区（anytime 本该发力）halt 大量 fc 掉帧、质量 88.79→82.67，IMDN l2 靠 5.1ms（8 倍余量）稳在 88.66，halt 反输 0.8~6.4 且越紧张崩越狠**；时变算力（slow_mkv/iid/burst，`sim_variable_compute.py`）同样输 1~2.7。根因：PAC 共享 nf=32 重骨干的浅出口是“深网络半成品”，e1(11.64ms,86.18)/e2(24.19,87.12) 在质量-延迟上被专用轻模型 IMDN_RTC l2(5.1ms,88.70) 严格 dominate，仅 e3(33.79,90.22) 质量最高但只在算力过剩时可持续。**e0 也救不了**：更浅出口质量更低、仍被 l2 dominate。anytime 想覆盖的紧张区恰是共享骨干浅档最弱区。

**方向含义与下一步（待拍板）**：轻量 CNN 单帧超分（IMDN 系、240p 小倍率）的低算力甜区已被专用轻模型占满，共享骨干 anytime 无帕累托空间；anytime 成立需“轻模型不够用、深浅质量差显著拉大”的设定。先做一两天离线测量：在更高超分倍率（4x）、强混合退化（压缩+模糊+噪声）、视频时序超分（VSR 多帧）上测 IMDN_RTC 轻模型 vs 重 PAC 深档的质量差。若轻模型显著塌陷（差 ≥3~5 分），anytime 价值区间出现、值得改任务设定重训；若仍追平，则该任务族 anytime 不成立，需止损或转异构快慢分支 / 系统成本（单模型替代 9 模型池）角度。'''

out = text[:s] + new + text[e:]
io.open(p, 'w', encoding='utf-8', newline='').write(out)
print('replaced bytes', e-s, '->', len(new))
