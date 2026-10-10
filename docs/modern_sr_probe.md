# 现代分辨率超分生死探针（1080p→4K x2）

## 结论（2026-10-08，NO-GO）

零样本 IMDN x2 超分在当代 1080p→4K（x2）区间，即使在压缩最重的 crf34（726kbps）下，也无法同时带来 VMAF 稳定增益与 LPIPS 不恶化：VMAF 增益 med 档中位仅约 +1.2 分，三档 LPIPS 在全部 4 个片段全部恶化。端侧神经后处理超分作为独立增强手段的大前提在本设定下不成立。

## 实验设定

- 源：`data/bbb/bbb_2160p_60fps.mp4`（3840×2160@60，参考帧 4K crf0 无损）。
- LR：4K bicubic 降 1080p 后 libx264 crf34（实测 726kbps，`workspace/pacnet/modern_sr/ladder/lr1080_crf34.mp4`）。
- 片段：`tmp/ref_c0..3.mkv`，起点 60/220/380/540s，各 20s=1200 帧@60。
- 对照：bicubic x2 vs IMDN low（nf6/3 模块）/med（nf12/5）/high（nf32/6），DIV2K 零样本 x2 权重。
- 指标：连续全片 libvmaf 4K 模型（n_subsample=2）逐帧分取均值；LPIPS（VGG）每 200 帧抽 1 帧取均值。脚本 `workspace/pacnet/modern_sr/probe_modern_sr.py`，流式双 ffmpeg 管道（内存恒定）。
- 延迟（RTX 5080，1080p 输入，`latency_modern_sr.json`）：low GPU fp32 10.7ms/e2e 16.2ms 可 60fps；med 30.9/36.2ms 仅 fp16 可 30fps；high 115ms 且 fp16 出 NaN，不可实时。

## crf34 四片段结果（`modern_sr_probe_c0123.json`，2026-10-08）

| 片段 | bic VMAF/LPIPS | low ΔV/ΔL | med ΔV/ΔL | high ΔV/ΔL |
|---|---|---|---|---|
| c0 | 61.00 / 0.331 | +0.29 / +0.005 | +0.49 / +0.005 | +0.39 / +0.004 |
| c1 | 56.87 / 0.332 | +0.81 / +0.005 | +1.12 / +0.005 | +0.98 / +0.005 |
| c2 | 60.14 / 0.325 | +0.96 / +0.006 | +1.33 / +0.004 | +1.13 / +0.004 |
| c3 | 80.87 / 0.129 | +1.70 / +0.056 | +2.71 / +0.070 | +2.45 / +0.061 |

ΔV=SR−bic VMAF；ΔL=SR−bic LPIPS（正=更差）。

## 判读

1. **VMAF 增益小且不稳**：med 最佳，ΔV 为 +0.49/+1.12/+1.33/+2.71，中位约 +1.2；三片段 <1.5。唯一明显增益的 c3（+2.7）同时 LPIPS 从 0.129 恶化到 0.199（+0.070，劣化 54%），VMAF 增益来自过锐化伪影，不是可感知细节恢复。
2. **LPIPS 全段全档恶化**：c0-c2 ΔL +0.004~+0.006（小但方向一致为负收益），c3 大幅恶化。
3. **规律与既往一致**：med 优于 low、high 反降——模型加大不带来增益，通用 DIV2K 权重在本任务已到顶；x2 温和放大区间 bicubic 本身已很强，客户端"凭空猜细节"无信息来源。
4. crf34 是理论上最该有增益的压缩档（最轻质量），此档已死，crf18/26 不必补跑。

## 对课题的含义

纯客户端后处理超分（426 宽 x4 救糊、1080p→4K x2）两区间均证伪，根因同为"输入码流不含被重建的信息"。下一候选方向：传统主码流之外加一条极小学习式高频先验旁路（1~2% 带宽），端侧轻量网以该旁路为条件增强；文献坐标与选题见飞书调研 MJpIdy0d4oaFDixEX8hcq2OXnne（2026-10-08）。

## 复现注意

- Windows 下需限线程（OMP_NUM_THREADS=4、libvmaf n_threads=2），双 worker 并行会把 16 核打满导致桌面卡死（worker_A.log 即双 worker 抢同一 tmp 文件名、rawvideo 读 0 字节死亡的现场；固定为单 worker 串行）。
- 长片 VMAF 必须连续时间序逐帧（n_subsample=2 为时间抽样），禁止把间隔帧打包成连续 clip（虚高 20~28 分，口径教训见 sr_aware_alloc_falsification.md）。

## 2026-10-10 档案复核补记（#14）

- **隔帧 frameNum 之谜已查明，非 bug**：libvmaf `n_subsample=2` 每 2 帧打 1 帧，json frameNum 记原始偶数帧号（0,2,…,1198），1200 帧输入→600 个分，四臂同参数、pooled 均值与结果 json 一致；4K VMAF v0.6.1 为逐帧空间模型无时间项，均匀时间抽样不偏均值。
- **参考/LR 帧对齐原为假设、已实测验证**（`audit_alignment.py`）：8 个抽检点（4 片段×片头/片中）下采样 4K 参考帧与同时间戳 crf34 LR 帧 PSNR 26.5~37.2 dB（差值仅含 crf34 压缩失真）；故意偏移 ±1 帧 PSNR 下降 2.5~14 dB（c3 高运动段对齐点 26.49 vs 偏移帧 11.9~13.1），offset=0 处处为最大值，双管道独立 seek 帧精确对齐。
- **逐帧配对分布**（`modern_sr_frame_vmaf.json`，每片段 600 帧）：med 档帧胜率 c0~c3 = 87.8%/98.7%/100%/100%，但 ΔVMAF p90 仅 +1.77/+2.82/+1.84/+3.03、逐帧中位 +0.22/+0.74/+1.23/+2.73——SR 确实在多数帧上小胜 bicubic，NO-GO 的含义是"增益量级过小（≤约 1~3）且 LPIPS 16/16 全恶化"，不是"SR 无效"。
- **冒烟 crf26 证据残缺说明**：clip0 120 帧 VMAF 已从 tmp 原始 json 复现（+0.44/+0.75/+0.55）；该轮 LPIPS（0.197/0.199/0.200 vs bic 0.188，仅 2 抽样帧）未落盘（smoke 输出 json 已不存在），不可复算，但 crf34 全量 LPIPS 才是判决证据。
- **fp16**：high 档 fp16 输出含非有限值（latency json `fp16_finite=false`），不可用；med fp16 22.4ms 仍超 60fps 帧预算 16.7ms，仅 low（9.8ms）可实时。
- 参考片为 crf0 x264（视觉无损而非数学无损），四臂共用不影响 Δ 比较；ladder mp4（57~440MB）、ref mkv（335~954MB）、tmp 不入库。
