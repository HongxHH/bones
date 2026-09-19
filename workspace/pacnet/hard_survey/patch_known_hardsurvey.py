# -*- coding: utf-8 -*-
import io
p = r'E:\dev\Code\Python\bones\docs\known_issues.md'
t = io.open(p, encoding='utf-8', newline='').read()
nl = '\r\n' if '\r\n' in t else '\n'
anchor = 'anytime 需换到轻模型不够用的设定（高倍率/强混合退化/视频 VSR），见 current_state 下一步。'
idx = t.index(anchor) + len(anchor)
title = '### 单帧压缩/倍率难度扫描：轻-重质量阶梯 ≤2.5、极端退化归零（阶段二难任务测量）'
items = [
 '`workspace/pacnet/hard_survey/measure_hard_survey.py` 在 7 档退化（426/320 宽即 4.5x/6x × CRF26/34/42，加 426/400k 锚点，每档 32 帧）上测 l1/l2/l3 与 PAC e1/e2/e3：l2(5.1ms)→e3(33.8ms) 的 VMAF 差全程 0~2.52，从不达 3~5 的价值阈值；差距呈倒 U（中等退化 crf26 最大 2.52，极端压缩 crf42 信息被抹除后所有模型一起崩、差距归零 0~0.5）；6x 不大于 4.5x；e3 比 l3(9.8ms) 仅高 0.6~1.7 分却慢 3.4 倍。',
 '含义：换更重压缩或更高倍率救不出 anytime 浅档价值区间，单帧 IMDN 系、压缩+下采样任务族的共享骨干 anytime 不成立。视频时序超分（VSR）这一“换信息源”维度尚未排除，但本地缺匹配退化的 VSR 代码/权重（anovsr、CARN-video 仅权重无推理代码，STDF 仅模型定义无权重且不放大），公平对比需训练，见 current_state 下一步。',
]
ins = nl + nl + title + nl + ''.join('- ' + x + nl for x in items)
io.open(p, 'w', encoding='utf-8', newline='').write(t[:idx] + ins + t[idx:])
print('inserted', len(ins))
