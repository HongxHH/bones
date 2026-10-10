# 档案：1080p→4K 现代分辨率超分探针（modern_sr）

- 状态：已封板（NO-GO，2026-10-08；2026-10-10 复核补证）
- 时间：实验 2026-10-08；复核 2026-10-10
- 最近复核：2026-10-10（读码 + 隔帧之谜查明 + 帧对齐实测 + 逐帧配对统计 + 冒烟数字复现）

叙事文档：`docs/modern_sr_probe.md`。

## 1. 要回答的问题

426 宽 x4 救糊区间已被 PAC-Net/hard_survey 系列证伪后，换一个对当代用户更相关的新区间：
原生真 4K60 素材 bicubic 降到 1080p 再压（模拟"4K 屏播 1080p 流"），零样本现代轻量 SR
（IMDN low/med/high x2）能否在 4K 屏上稳定优于 bicubic x2？这是纯客户端后处理超分在
"现代分辨率"下的生死探针。

## 2. 方法概述

- 参考：`data/bbb/bbb_2160p_60fps.mp4`（3840×2160@60），按 60/220/380/540s 起点切
  4 个 20s 片段（1200 帧/段），crf0 x264 视觉无损参考（tmp/ref_c*.mkv，不入库）。
- LR：4K bicubic 降 1080p 后 libx264 CRF18~34（gen_1080p_ladder.ps1，GOP=240）；
  判决档 crf34 实测 726kbps。
- 四臂：bicubic x2 / IMDN low(nf6,m3) / med(nf12,m5) / high(nf32,m6)，DIV2K 零样本 x2 权重。
- 测量：ref+四臂竖排 rawvideo 堆叠、单次 ffmpeg 四路 libvmaf（4K 模型，n_subsample=2）
  连续时间序打分；LPIPS-VGG 每 200 帧抽 1 帧（6 帧/段）；单帧延迟 40 次计时（fp32/fp16）。
- 双 ffmpeg 流式管道，内存恒定；单 worker 串行（限 OMP=4、libvmaf n_threads=2）。

## 3. 文件清单

| 文件 | 作用 | 复核定性 |
|---|---|---|
| probe_modern_sr.py | 主探针（切参考/双管道/四臂/延迟/LPIPS） | 正确；docstring 抽样率已勘误 |
| gen_1080p_ladder.ps1 | 4K→1080p 五档 CRF 阶梯 | 正确，ladder mp4 不入库可再生成 |
| bench_lat_recheck.py / bench_tiles.py | 延迟二轮复测、分块推理可行性 | 只产出控制台数；结论 med 不可 60fps、high fp16 NaN |
| audit_alignment.py | **2026-10-10 新增**：ref/LR 帧对齐实测核查 | 8 点对齐成立（见 §6 N2） |
| modern_sr_probe_c0123.json | crf34 四片段均值结果 | 现算逐值复现 |
| modern_sr_frame_vmaf.json | **2026-10-10 新增**：四臂 600 帧/段逐帧 VMAF（+crf26 冒烟 60 帧） | 配对统计证据，105KB |
| latency_modern_sr.json | 三模型 1080p 输入 fp32/fp16 延迟 | 复算引用 |
| models/vmaf_4k_v0.6.1.json | 4K VMAF 模型（22.9KB） | 已跟踪 |
| ladder/*.mp4（57~440MB）、tmp/ref_c*.mkv（335~954MB）、tmp/vmaf_*.json、worker_*.log | 数据/临时/日志 | 不入库 |

## 4. 复现方式

SR 解释器 `E:\dev\env\miniconda3\envs\SR\python.exe`，cwd 为本目录，OMP_NUM_THREADS=4。

1. `powershell -ExecutionPolicy Bypass -File gen_1080p_ladder.ps1`（约数小时编码）。
2. `python probe_modern_sr.py`（全量 4 片段×5 CRF；判决只需 `MODERN_SR_CRFS=34`，
   单片段 21~29 分钟）；冒烟 `set MODERN_SR_SMOKE=1`。
3. 对齐核查：`python audit_alignment.py`（秒级）。
4. 必须单 worker；models/vmaf_4k 需在 cwd（tmp）下，Windows libvmaf 不认盘符路径。

## 5. 结论与关键数字

crf34（726kbps）四片段，ΔVMAF（SR−bic）/ΔLPIPS（正=更差）：

| 片段 | bic VMAF/LPIPS | low | med | high |
|---|---|---|---|---|
| c0 | 61.00 / 0.331 | +0.29 / +0.005 | +0.49 / +0.005 | +0.39 / +0.004 |
| c1 | 56.87 / 0.332 | +0.81 / +0.005 | +1.12 / +0.005 | +0.98 / +0.005 |
| c2 | 60.14 / 0.325 | +0.96 / +0.006 | +1.33 / +0.004 | +1.13 / +0.004 |
| c3 | 80.87 / 0.129 | +1.70 / +0.056 | +2.71 / +0.070 | +2.45 / +0.061 |

- med 档 ΔVMAF 中位约 +1.2；**16/16（4 段×3 模型）LPIPS 全部劣于 bicubic**。
- 逐帧配对（600 帧/段）：med 帧胜率 87.8/98.7/100/100%，但 ΔVMAF p90 仅 +1.77/+2.82/+1.84/+3.03；
  c3 最大 VMAF 增益伴随 LPIPS 恶化 54%（过锐化假纹理，与 hard_survey G2/VIF 同族机理）。
- crf26 冒烟（clip0 120 帧）VMAF +0.44/+0.75/+0.55，LPIPS 三档同样全部差于 bicubic。
- 延迟（RTX 5080，1080p 输入）：low GPU 10.7ms/e2e 16.2ms（可 60fps）；
  med 30.9/36.2ms、fp16 22.4ms 仍超 16.7ms 帧预算；high 115ms 且 fp16 出 NaN。

## 6. 复核记录

| # | 发现 | 严重度 | 处置 |
|---|---|---|---|
| N1 | "隔帧 frameNum=0,2,4… 600/1200"之谜：libvmaf `n_subsample=2` 每 2 帧打 1 帧、frameNum 记原始偶数帧号，1200→600 分；四臂同参数、pooled 均值与结果 json 逐值一致；4K 模型为逐帧空间模型，均匀时间抽样不偏均值 | 疑问消除（非 bug） | docstring 与 docs 已写明机制 |
| N2 | ref 与 LR 靠两条 ffmpeg 独立 seek 对齐，此前从未验证。新增 audit_alignment.py 实测 8 点：对齐 PSNR 26.5~37.2dB，故意偏 ±1 帧降 2.5~14dB（c3 高运动 26.49 vs 11.9~13.1），offset=0 处处最大 | 重要隐患排除 | 脚本与数字入档 |
| N3 | docstring 写"LPIPS 每 100 帧取 1"，全量实际每 200 帧（6 帧/段），仅冒烟为 100 | 低（文档错） | 已改 |
| N4 | crf26 冒烟 LPIPS（2 抽样帧）未落盘、smoke json 已不存在，不可复算；VMAF 部分已从 tmp json 复现 | 低（证据残缺） | docs 注明；判决依据为 crf34 全量 LPIPS，不受影响 |
| N5 | 逐帧分布补证：SR 多数帧小胜（med 胜率 88~100%）但量级 ≤1~3 且感知全损——NO-GO 是"增益太小+伪影"而非"SR 完全无效"，定性更精确 | 正面补强 | modern_sr_frame_vmaf.json 入档 |
| N6 | 双 worker 死亡现场（worker_A.log：rawvideo 短读 0 字节）：两 worker 共用 tmp 输出文件名互相覆盖 + 打满 16 核 | 工程教训 | docs 复现注意注明；铁律=单 worker |
| N7 | 参考为 crf0 x264 视觉无损而非数学无损；四臂共用同一参考，Δ 比较不受影响 | 极低 | 记录 |
| N8 | 边界：仅 BBB 单一内容、仅 IMDN 家族零样本 DIV2K 权重、crf34 一档全量、LPIPS 6 帧/段、未测生成式/专用 x2 大模型（重，违背端侧前提） | 边界 | docs 与 §8 记录 |

## 7. 影响面：历史结论逐条重新定性

| 历史结论 | 复核定性 |
|---|---|
| "1080p→4K x2 零样本 IMDN 无稳定增益、LPIPS 全恶化，NO-GO" | **成立且证据增强**（对齐已验证、逐帧分布已补、冒烟 VMAF 已复现） |
| "c3 +2.7 增益是过锐化假纹理" | **成立**（LPIPS +0.070/恶化 54%，逐帧 VMAF p90 仅 +3.03） |
| "med 已到顶、high 反降" | **成立**（四片段 med 均最优，high ΔVMAF 均低于 med） |
| "crf34 已死不必补 crf18/26" | 维持：crf34 是压缩最重、SR 理论空间最大档；crf26 冒烟方向一致。补全矩阵价值低 |
| 延迟"仅 low 可 60fps" | **成立**，并补记 high fp16 NaN、med fp16 仍超时 |

## 8. 是否重做：判断与方案

**封板，不补全 CRF 矩阵、不换更大模型重测。** 复活条件（均需新前提，不在原命题内）：

1. 出现**时域视频超分**或针对压缩失真训练的轻量 x2 模型（DIV2K 通用权重的天花板已实测）；
   重测时直接复用本探针管道，加 LPIPS 逐帧密度（建议每 50 帧 1 帧）。
2. 有旁路/侧信息输入（命题已转为档案 #15 side_channel，信息论上界亦已 NO-GO）。
3. 若未来主张"现代分辨率仍有收益"，必须同时满足：多内容（≥3）、逐帧 LPIPS 不恶化、
   端侧延迟进帧预算——三条缺一不可，单一 VMAF 均值不再作为证据。

## 9. 踩坑铁律

- 4K/长片测量：连续 rawvideo 堆叠管道 + 帧数/短读硬断言；禁止抽样帧打包（#9 G1）。
- 任何"独立 seek 对齐"都必须用偏移灵敏度实测验证（本档案 N2 范本）。
- libvmaf `n_subsample` 会改变 json 帧数与 frameNum 编号，引用前先核对，勿当丢帧。
- 单 worker 串行；多进程不得共用同一 tmp/输出文件名。
- fp16 可行性必须检查输出有限性（high 档 NaN 教训）。

## 10. 上下游关系

- 上游：#9 hard_survey（426 宽 x4 区间证伪与 LPIPS 判决方法）、#8 PAC-Net（延迟 dominate 结论）。
- 本档案关闭"纯客户端后处理超分"第二区间（现代分辨率 x2），与 #9 共同构成
  "无额外信息→客户端猜不出细节"的完整证据链，直接催生 #15 旁路方向。
- 测量管道（4K 模型、竖排多路 libvmaf、流式双 ffmpeg）是后续任何高分辨率 SR 评测的范本。

## 11. 复盘备注（复活检查清单）

1. 新 SR 模型声称 1080p→4K 有效？先过本探针：真 4K 参考、连续时间序、VMAF+LPIPS 双指标、延迟进预算。
2. 引用本档案数字时是否区分了"VMAF 小胜"与"感知收益"？帧胜率高不等于可发布增益。
3. 测量管道的 seek 对齐是否做了偏移核查？
4. LPIPS 抽样密度是否足够（本档案仅 6 帧/段，重测时加密）？
5. 是否误用 crf0 x264 参考做绝对分声称？Δ 比较才是本档案口径。
