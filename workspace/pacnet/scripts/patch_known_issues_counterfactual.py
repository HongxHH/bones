# -*- coding: utf-8 -*-
import io
p = r'E:\dev\Code\Python\bones\docs\known_issues.md'
text = io.open(p, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in text else '\n'
anchor = 'K=1.5 可能高估深档流畅度惩罚。'
idx = text.index(anchor) + len(anchor)

blocks = [
('### 增强在 asyncio 单线程同步阻塞，但与下载基本并行、卡顿杠杆在码率（阶段二反事实）', [
 '全部 enhancer 的 torch forward + cuda.synchronize + gpuburn 在 asyncio run 内同步执行，全工程无 run_in_executor/线程池，每帧仅 sleep(0) 让出；trace 下载在同一事件循环按 20000 字节分块 sleep 限速。',
 '0921 配对到相同码率的段（halt vs IMDN 155 段）下载墙钟仅差 +3.1s（中位 −0.02s），增强多花 368s 却几乎没拉长同码率下载——增强在下载 chunk sleep 间隙“免费搭车”。halt 多卡 15.5s 主要来自差网选了更多 repr1（下载字节多）。',
 '复刻 trace.py 限速的下载反事实（sim_download_counterfactual.py）：固定 repr0(400k) 卡顿 0（下载 615s < 视频 736s），repr1 卡 512s、repr2 卡 1096s。卡顿唯一硬杠杆是码率下载量；仿真系统性低估真实卡顿 15-20s（未建模增强关键路径/解码/起播），但相对顺序正确。',
]),
('### 共享骨干浅出口被专用轻模型帕累托压制（阶段二全 K 检验，决定 anytime 适用设定）', [
 '固定 repr0 消除下载瓶颈后扫 K=1..6（sim_pareto_k.py）：halt 仅 K≤1.2（全程 e3、anytime 不换档）赢 IMDN 0.58；K≥1.5 紧张区 halt 掉帧崩到 82.7，IMDN l2（5.1ms、8 倍余量）稳在 88.66，halt 反输 0.8~6.4；时变算力（sim_variable_compute.py）同样输 1~2.7。',
 '根因：PAC 共享 nf=32 重骨干的 e1(11.64ms,86.18)/e2(24.19,87.12) 是“深网络半成品”，被专用轻模型 IMDN_RTC l2(5.1ms,88.70) 严格 dominate；e0 更浅但质量更低仍被压制。轻量 CNN 单帧、小倍率超分的低算力甜区已被轻模型占满，anytime 需换到轻模型不够用的设定（高倍率/强混合退化/视频 VSR），见 current_state 下一步。',
]),
]
ins = ''
for title, items in blocks:
    ins += nl + nl + title + nl
    for it in items:
        ins += '- ' + it + nl
out = text[:idx] + ins + text[idx:]
io.open(p, 'w', encoding='utf-8', newline='').write(out)
print('inserted', len(ins), 'chars, nl=', repr(nl))
