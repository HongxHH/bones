# 高频先验旁路（side-channel）信息论上界探针

## 假设

传统 H.264 主码流（1080p crf34 = 726kbps）之外，独立编码一条仅占 1~2% 带宽（7~15kbps）的学习式高频先验旁路，端侧轻量条件 SR 以该旁路为条件重建 4K。go/no-go = 1~2% 旁路能否把救糊区间 VMAF 增益从 <1 拉到 ≥+2。

## 方法

做信息论上界 oracle：旁路传的不是学习式 token，而是**真残差** D = 真 4K − bicubic(1080p 放大)，用 libx264 CRF 阶梯编码成旁路；客户端 F = bic + 解码残差。任何学习式先验都不可能比"直接传真残差"更省码率，所以这是天花板。

- 素材：BBB 4K60，clip0 起点 60s，200 帧冒烟
- 主码流：`modern_sr/ladder/lr1080_crf34.mp4`（1920×1080@60，726kbps）
- 旁路编码：`ffmpeg blend=all_mode=grainextract`（= ref−bic+128），libx264 veryfast yuv420p
- 重建：`blend=all_mode=grainmerge`（= bic+side−128）
- VMAF：4K 模型 vmaf_4k_v0.6.1，n_subsample=2，连续时间序
- 脚本：`workspace/pacnet/side_channel/side_channel_probe.py`

## 结果（2026-10-09，clip0，200 帧）

| 旁路 CRF | 旁路码率 | 占主码流 | F 的 4K VMAF | ΔVMAF vs bicubic |
|---|---|---|---|---|
| 38 | 489 kbps | 67% | 83.72 | +9.10 |
| 42 | 278 kbps | 38% | 79.26 | +4.64 |
| 46 | 188 kbps | 26% | 74.82 | +0.20 |
| 50 | 144 kbps | 20% | 73.94 | −0.67 |
| 62 | 143 kbps | 20% | 73.95 | −0.67 |
| 0（纯 bicubic） | 0 | 0% | 74.62 | 基线 |

## 判决：NO-GO

- +2 VMAF 拐点在旁路 ~220kbps ≈ **主码流 30%**，不在 1~2%。
- 目标 1~2%（7~15kbps）落在曲线最右端，旁路被量化成噪声，ΔVMAF ≈ 0 或负。
- crf50/62 码率钉死在 144kbps、ΔVMAF 钉死在 −0.67：残差里的稀疏高频尖峰吃光 bits，再提高 CRF 也减不了码率，只会更糊。
- 学习式先验（VQ token/edge map/wavelet）可能比原始残差省 2~5 倍，但那是把需求从 30% 降到 6~15%——仍是"再加半条码流"，与直接请求更高档 ABR 无差别，不构成"便宜旁路"卖点。

## 工程坑（复现注意）

- ffmpeg `blend=all_expr=...` 在本机 Windows 构建上**静默失效**：连 `all_expr='X'` 直通都输出 mean≈129/std≈83 的无关信号（与真残差 corr≈0）。必须用内置模式 `all_mode=grainextract`（=X−Y+128）和 `grainmerge`（=X+Y−128），不要用自定义表达式。
- numpy 逐帧 pipe 方案（dec_popen + np.subtract + enc_popen）在第二个 CRF 必现管道死锁 40 分钟无进展，原因是 encoder stderr 管道未排空；即使加 `stderr=DEVNULL` 仍偶发卡死。纯 ffmpeg filter_complex 路线稳定。
- 真残差统计（clip0 f50 Y）：mean≈0.07、std≈3.6、范围 [−87,+112]，稀疏小信号带边缘尖峰。
