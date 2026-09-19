# -*- coding: utf-8 -*-
import io
p = r'E:\dev\Code\Python\bones\docs\pacnet_progressive_enhancer.md'
t = io.open(p, encoding='utf-8', newline='').read()
old = ('anytime 在轻量 CNN、小倍率单帧超分的低算力甜区无帕累托空间；要成立需换到轻模型不够用、'
'深浅质量差显著拉大的设定（4x / 强混合退化 / 视频时序 VSR），下一步先离线测轻重模型质量差再决定是否改设定重训。')
new = ('anytime 在轻量 CNN、小倍率单帧超分的低算力甜区无帕累托空间。2026-09-19 离线难度扫描（hard_survey，'
'426/320 宽即 4.5x/6x × CRF26/34/42 共 7 档、32 帧）进一步证伪“换更难单帧任务”：l2(5.1ms)→e3(33.8ms) 的 VMAF 差全程 0~2.52、'
'从不达 3~5 阈值，且呈倒 U（中等退化最大、极端压缩归零），6x 不大于 4.5x——更重压缩/更高倍率也救不出浅档价值区间。'
'唯一未排除的是视频时序超分（多帧提供单帧拿不到的亚像素信息，属“换信息源”而非同信息加深）；但本地缺匹配退化的 VSR 代码/权重，'
'公平对比需训练。下一步先用光流多帧融合 oracle（不训练、不 OOD）估时序信息上限，再决定是否训练轻/重 VSR 或转系统成本角度，详见 current_state。')
assert old in t, 'anchor not found'
io.open(p, 'w', encoding='utf-8', newline='').write(t.replace(old, new))
print('pacnet doc patched')
