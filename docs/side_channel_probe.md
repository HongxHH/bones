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

## 2026-10-10 档案复核补记（#15）

- **bic 基线来源已查实**：tmp/bic_c0.mkv 不由探针脚本生成（原脚本假设其预存在，复现缺口已由 `ensure_bic()` 补上）。帧级 PSNR 比对确认它 = crf34 主码流经 **ffmpeg 原生 YUV 域 bicubic** 放大 4K 后 crf0 落盘（与现重建 ffmpeg bicubic 帧 PSNR 47.6dB；同帧段重测 pooled 73.33 vs 归档 74.62，−1.3 来自帧数/seek 对齐差异，不影响曲线形态）。
- **跨档案测量路径发现（重要）**：modern_sr 探针的 bicubic 走 cv2 解码→RGB 域 INTER_CUBIC→再转回 YUV，同帧段（frameNum 0~214）仅 64.36 VMAF，比 ffmpeg 原生 YUV 域放大低约 9~10 分。modern_sr 中 bic 与三个 IMDN 臂共用该路径，其 Δ 比较对称成立、NO-GO 结论不受影响，但绝对 VMAF 系统性偏低；本 oracle 以"免费且最强的原生放大"为基线是**正确且保守**的选择（基线越强，残差旁路越难刷出增益）。
- **实际范围澄清**：全部产物（side mp4 nb_frames=200、vmaf 108 scored frames）均为 clip0 200 帧冒烟，4 片段×1200 帧全量从未运行；结果 json 于 2026-10-10 由 tmp 原始 vmaf json 与 side mp4 码率聚合复原（补齐 q42/q46 两个点）。
- **目标码率带不可达即判决**：4K veryfast 编码残差存在 143~144kbps 地板（crf50/62 码率与 ΔVMAF 双钉死），目标 1~2%（7~15kbps）在该编码路径下物理不可表达，地板处 F 已略差于 bic。
- **边界（诚实记录）**：本探针是**信号上界**（真残差，任何学习式先验不可能携带更多像素信息），但不是**码率下界**——4K 分辨率 + libx264 veryfast 非最优编码器；换 AV1/慢速/学习式特征压缩或低分辨率 latent 或有 20~50% BD-rate 改善，拐点最好情况从 30% 移到约 15~25%，与 1~2% 仍有近一个数量级差距；像素重建率失真受同一 RD 物理约束，不随编码方式消失。
