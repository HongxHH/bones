# 档案：增强算子增益普查 + leave-segments-out 对症微调闭环（enhancer_survey）

**状态：已封板（现成算子证伪成立、证据已补齐；对症微调"跨块泛化"定性保留但 +7~14 大数字含 checkpoint 选择泄漏，严谨引用口径以本档 §5/§7 为准）**
**时间：2026-09-11 ~ 2026-09-12（阶段17 及续）**
**最近复核：2026-10-10（7 个 py 全读、5 个结果 json 现算、STDF/NAFNet/cascade 重跑补证、训练日志核对、泄漏口径重算）**

## 1. 要回答的问题

阶段14（段级选择）、15（帧级 DP）、16（服务端引导）三轮证伪后，回到增强地基本身：

1. 换一个更强/更新/更对症的增强算子（去模糊、去压缩块、轻量视频超分 VSR），逐段 VMAF 能否显著高于 IMDN 单帧超分？
2. 不同算子的相对优势是否**内容非单调**（各段各有胜负）？这是"按内容选算子/多算子协同"能否成立的前提。
3. （续）IMDN-BBB（在目标视频上微调）相对通用 IMDN-DIV2K 的 +6.11 "主场红利"，是可跨段泛化的"对症真实退化"能力，还是单纯逐段/逐场景记忆？

## 2. 方法概述

- 源：`data/video_mp4box/output_{240,360,480}p.mp4`（真实 H.264 链路），GT 为 `output_1080p.mp4`，同帧号对齐；每段 120 帧（30fps，BBB 159 段）。
- 普查（`step1_operator_gain_survey.py`，算子可插拔注册表）：base=cv2 bicubic、yuv_up（YUV 分通道中性对照）、NAFNet-GoPro-width32（1080p 去模糊）、STDF（MFQEv2，7 帧 LR 尺度去压缩再 bicubic 放大）、IMDN 三档（BBB 微调/DIV2K 通用）、STDF→IMDN 级联、NanoVSR-644k（T=15 分窗双向循环 VSR，固定 4x 后 bicubic 对齐 1080）。
- 打分：段内 120 **连续帧**逐帧过 ffmpeg libvmaf（enh/ref 双臂都先 cv2 VideoWriter mp4v 打包成临时 mp4），取逐帧 VMAF 均值；IMDN 对照优先读 `data/bbb/per_frame_quality.json` 并做同管道自检。
- 样本设计：小样本 seg 2/60/129 冒烟；正式为全片均匀 16 段 seg 0,10,…,150。
- 对症微调（`train_leaveout.py`）：从 IMDN-high-DIV2K x4 warm start，训练对取真实链路退化（LR=output_240p、GT=output_1080p，按真实缩放比 sy=4.5/sx=4.507 裁 GT 再 resize 到 4x 网格，LR patch 48/HR 192），L1、Adam lr2e-4、StepLR(15,0.5)、batch16、40 epoch、翻转/转置同步增广；每非留出段每 8 帧取 1 训练帧，留出段仅 j∈{30,60,90} 三帧做 val。两种留出口径：
  - uniform：测试段 UNIFORM_TEST=0,10,…,150（16 段），val=[30,90,140]；
  - block：连续块 28-32、88-92、**135-159 整块**（含相邻段）不训练，val=[30,90,140,150]，测试 {30,90,140,145,150,155}。
- best checkpoint 按 val 段 RGB PSNR 选（train_leaveout.py:131-133）。

## 3. 文件清单

| 文件 | 作用 | 复核定性 |
|---|---|---|
| `step1_operator_gain_survey.py` | 算子注册表 + 同口径普查主脚本（327 行） | 可复现；**`*_ms` 字段含模型推理+mp4v 写盘+ffmpeg VMAF 全程墙钟（:295-299），不是模型延迟（Q5）**；VMAF 打包口径见 Q4 |
| `step2_analyze.py` | NanoVSR vs IMDN-BBB：非单调、异构 oracle、内容 Spearman | 现算复现 |
| `step3_analyze.py` | 公平口径：加 IMDN-DIV2K 泛化对照 | 现算复现 |
| `step4_analyze.py` | leave-out 闭环判决 | **原对 block json 必崩（KeyError 240_145，inner join step2），本次已修为 .get 可选连接 + NaN 安全均值（Q3）** |
| `train_leaveout.py` | 两种留出口径对症微调（145 行） | 可复现；**val 段与 block 测试段重叠 → checkpoint 选择泄漏（Q1）** |
| `stdf_model.py` | TV 版 STDF 网络定义（3.7KB） | 权重不在本目录、在 `enhance/stdf-pytorch/exp/MFQEv2_R3_enlarge300x/ckp_290000.pt`（存在） |
| `dcn_nocompile.py` | grid_sample 免编译分组调制 DCNv2 替代（4.5KB），offset=0/mask=1 等价 F.conv2d 单测 | strict 加载零缺失、重跑出数正常，适配有效 |
| `step1_survey.json` | 3 段冒烟（仅 base/imdn/nano） | 已入库 |
| `step1b_stdf_nafnet.json` | **本次新增**：seg2/60/129 的 base/yuv_up/stdf/nafnet/imdn/cascade 重跑 | 补历史只在 docs 没落盘的证据（Q2） |
| `step2_survey16.json` | 16 段 base/IMDN-BBB/Nano 逐段 | 已入库，现算复现 |
| `step3_fair16.json` | 加 IMDN-DIV2K 的 16 段公平对照 | 已入库，现算复现 |
| `step4_leaveout16.json` | uniform 留出 16 段（含 imdn_lo） | 已入库，现算复现 |
| `step4b_block6.json` | block 留出 6 段 | 已入库，现算复现 |
| `weights/IMDN_high_bbbLO_x4{,_best}.pth` + `_trainlog.json` | uniform LO 权重（best=ep40，val 25.886） | 已入库（各约 762KB） |
| `weights/IMDN_high_bbbLO_block_x4{,_best}.pth` + `_trainlog.json` | block LO 权重（best=ep36，val 23.237） | 已入库 |
| `vmaf_tmp/` | 临时 mp4 空目录 | 无文件 |

外部依赖（不在本目录）：`enhance/NAFNet/experiments/pretrained_models/NAFNet-GoPro-width32.pth`、`enhance/stdf-pytorch/.../ckp_290000.pt`、`enhance/nanovsr/checkpoints/nanovsr_644k.pth`、播放器 IMDN 权重目录、`data/video_mp4box/output_*.mp4`、`data/bbb/per_frame_quality.json`。

## 4. 复现方式

SR 环境、单卡（脚本内 libvmaf n_threads=4，限 OMP=4）：

```powershell
cd E:\dev\Code\Python\bones\workspace\enhancer_survey
$env:OMP_NUM_THREADS=4
# 普查（算子可插拔；正式 16 段把 --segs 换为 0,10,...,150，约数十分钟）
E:\dev\env\miniconda3\envs\SR\python.exe step1_operator_gain_survey.py --res 240 --segs 2 60 129 --ops base yuv_up stdf nafnet imdn cascade --out step1b_stdf_nafnet.json
# 分析
E:\dev\env\miniconda3\envs\SR\python.exe step2_analyze.py
E:\dev\env\miniconda3\envs\SR\python.exe step3_analyze.py
# 对症微调（各约 3~5 分钟/40ep，RTX 5080）
E:\dev\env\miniconda3\envs\SR\python.exe train_leaveout.py --holdout uniform --epochs 40
E:\dev\env\miniconda3\envs\SR\python.exe train_leaveout.py --holdout block   --epochs 40
# 闭环对照（step4 json 由 step1 同管线带 imdn_lo 算子生成；分析器已支持 block）
E:\dev\env\miniconda3\envs\SR\python.exe step4_analyze.py step4_leaveout16.json
E:\dev\env\miniconda3\envs\SR\python.exe step4_analyze.py step4b_block6.json
```

## 5. 结论与关键数字

### 5.1 现成算子普查（240p，16 uniform 段，同管道配对差）

| 方案 | 均值 VMAF | vs base |
|---|---|---|
| base bicubic | 40.27 | — |
| NanoVSR-644k（ECCV'26 泛化 VSR） | 51.94 | +11.67 |
| IMDN-high-DIV2K（通用 ISR） | 54.99 | +14.72 |
| IMDN-high-BBB（目标视频微调） | 61.10 | +20.83 |

- NanoVSR − IMDN-BBB：均值 **−9.16**、中位 −5.16、范围 [−36.40,−1.52]，**0 赢/16 输**；逐段异构 oracle=纯 IMDN，选择上限 Δ=**0.000**（非单调选择空间不存在）。
- NanoVSR − IMDN-DIV2K（公平零样本）：均值 **−3.05**、范围 [−12.3,+0.4]，**1 赢/15 输**；IMDN 主场红利（BBB−DIV）= **+6.11**。
- 内容相关性：Δ(Nano−BBB) 与段运动/纹理 Spearman **−0.606/−0.517**（越复杂输越多）；Δ(Nano−DIV) 为 −0.206/−0.209。
- 小样本 seg2/60/129（2026-10-10 重跑落盘 `step1b_stdf_nafnet.json`）：NAFNet vs base **+4.64/−9.36/−0.49**；STDF vs base **−8.10/−5.13/−2.01**；STDF→IMDN 级联 vs 直接 IMDN **−8.55/−8.87/−3.03**；yuv_up 对照 ≈0（−0.06/−0.01/−0.02）。冒烟期 seg2 Nano +1.07 小胜，扩到 16 段确认噪声。

### 5.2 leave-segments-out 对症微调

- uniform 16 段：base 40.27 / DIV 54.99 / **LO 57.92** / BBB 61.10；LO−DIV 均值 +2.93、中位 **−0.73**、范围 [−5.2,+28.4]、5 赢/10 输；LO−BBB −3.18。
- block 6 段（连相邻块都不训练）：LO−DIV 逐段 seg30 **−0.51**、seg90 **−1.48**、seg140 **+13.98**、seg145 **+3.34**、seg150 **+7.26**、seg155 **+0.27**；均值 +3.81、中位 +1.80。
- 训练日志：uniform 40ep best=ep40 val PSNR 25.886；block 40ep best=ep36 val 23.237（每 epoch 4.4~5.2s）。

**定性结论**：现成预训练增强算子（去模糊/去压缩块/通用 VSR）在真实 H.264 低清链路上全部跑输对症单帧 IMDN，且不产生内容非单调选择空间；根因是退化不匹配（NAFNet 训于运动模糊、STDF 训于 HEVC、NanoVSR 训于干净 bicubic），时序模型在高运动压缩段沿时间轴传播噪声、失效最重。对症真实退化训练的增益严格集中在通用模型塌陷的高运动/强压缩段，普通段持平到轻微遗忘。

## 6. 复核记录

- **Q1（重要，结论定量弱化）checkpoint 选择泄漏**：best epoch 按 val 段 RGB PSNR 选取（train_leaveout.py:103-106,131-133），而 block 测试段 {30,90,140,145,150,155} 中 30/90/140/150 正是 val 段。历史标题数字 +13.98（seg140）、+7.26（seg150）落在 val 段上，属间接选择泄漏（按 PSNR 选 epoch、按 VMAF 报告，val 每段仅 3 帧）。**真正零泄漏的块内段只有 seg145（+3.34）、seg155（+0.27），n=2**。uniform 口径排除 val 段（30/90/140）后干净 13 段：均值 +1.72、中位 **−0.71**、4 赢/8 输（seg130 +8.49、seg150 +25.49 对 uniform 协议是非 val 干净段，但带相邻段近邻记忆；seg140 +28.40 为 val 段）。
- **Q2（证据缺口，已补）**：STDF/NAFNet/级联的负数历史只写在 docs、无 json 落盘。本次重跑 `step1b_stdf_nafnet.json`，与 docs 数字在 ±0.03 内一致（临时编码波动），证据补齐。
- **Q3（代码 bug，已修）**：`step4_analyze.py` 原 `b=bb[k]` 对 block-only 段（145/155）KeyError 崩溃；改为 `.get` 可选连接、BBB/Nano 列 NaN 安全，block 汇总现可直接跑。
- **Q4（测量口径债，程度轻于 #9 G1）**：段内 120 帧为**连续帧**、时间序真实（不同于 hard_survey 的全片非相邻抽帧），但 enh/ref 双臂都经 cv2 mp4v 重编码打包，mp4v 对不同高频内容失真不对称，绝对 VMAF 非金标准；self-check base diff=0、imdn diff≤0.2 只证明与 `per_frame_quality.json`（同 mp4v 管道）自洽，属循环验证。同管道配对 Δ 可用；金标准复核（双 rawvideo 连续 libvmaf）并入帧级指标体系 A 阶段。
- **Q5（字段误导）**：`*_ms` 计时区间为"模型推理+mp4v 写盘+ffmpeg libvmaf"全程（step1:295-299 t0 在算子前、ms 在 VMAF 后），不能当模型延迟引用；真实模型延迟以 PAC-Net 延迟表/modern_sr `latency_modern_sr.json` 为准。
- **Q6（文档小错，待改）**：`docs/current_state.md:143` 称 stdf"仅模型定义无权重"，实际权重在 `enhance/stdf-pytorch/.../ckp_290000.pt`（Test-Path=True）；`docs/enhancer_operator_survey.md:62` 把 Spearman −0.21 归给"Nano−IMDN"，实际 −0.21 是 Nano−DIV2K，Nano−BBB 为 −0.61/−0.52。
- **Q7（边界）**：仅 BBB、仅 240p、仅 VMAF（无 LPIPS/PSNR 配对）；跨视频零样本 leave-out 从未验证（TOS 未做）；STDF 设计上只做 LR 尺度去压缩不放大。

## 7. 影响面（历史结论逐条重新定性）

| 历史结论 | 复核定性 |
|---|---|
| NAFNet/STDF/级联小样本全部为负 | **成立**，本次重跑补证（±0.03 复现） |
| NanoVSR 0/16 输 BBB、1/15 输 DIV、异构 oracle Δ=0 | **成立**，数字现算复现；非单调选择空间判决不动 |
| 主场红利 +6.11、BONES 质量表部署假设脱节 | **成立**（同管道口径） |
| "对症训练可跨连续块泛化，困难段 +7~+14" | **弱化但不反转**：+7~14 段（140/150）是 val 段带 checkpoint 选择泄漏；零泄漏块内仅 seg145 +3.34/seg155 +0.27（n=2，方向为正）。定性"能跨块、只在困难段为正"保留，定量数字在论文中不得按 +7~14 引用 |
| "uniform 口径高估约一半、普通段持平到轻微为负" | **成立且更强**：干净 13 段中位 −0.71、8/13 为负 |
| PAC-Net 阶段0 复用本档案管线（uniform16+DIV2K warm+真实退化） | 范式复用见 `docs/current_state.md:215`；其 imdn_lo 对照继承同一 val 协议，若泄漏使 imdn_lo 略乐观，则"e3 仅逊 imdn_lo 0.287dB"是保守方向（对照被抬高），PAC 判决不受损 |

## 8. 是否重做

- 普查部分：**不重做**。结论多通道互证（VMAF 配对差 + #9 LPIPS 逐帧 + #13 连续管道），金标准绝对值复核统一并入帧级指标 A 阶段，单独立项重复测量无收益。
- 对症微调：**不整轮重做**，但该结论若未来要进入论文，必须先做一次协议修正小实验（见 §11 复活清单 R1，约 15 分钟），替换 +7~14 的泄漏数字。

## 9. 踩坑铁律

1. 非整数缩放比（240p→1080p 为 4.5/4.507x）训练必须按真实比裁 GT 再落到模型整数倍网格；按 4x 裁导致 LR/GT patch 仅 10.7dB 错位、训崩（val 15dB、VMAF=0），修正后 27dB。
2. leave-out 实验的 checkpoint 选择段（val）与报告段（test）必须不相交，否则 early-stop 选择本身就是泄漏，尾部大数字会被显著抬高。
3. 评测管道里的计时字段先看计时区间再命名；把 VMAF ffmpeg 时间算进模型延迟会得出错误的速度对比。
4. 双臂都重编码不等于金标准：self-check 只能验证与同管道历史文件自洽。
5. STDF 官方 DCNv2 需编译，`dcn_nocompile.py` 的 grid_sample 调制变形卷积可直接替代（offset=0/mask=1 单测 <1e-5、strict 加载零缺失）。

## 10. 上下游关系

- **上游**：`workspace/measure_per_frame_vmaf.py`（逐帧 VMAF 打包口径，本档案复用）；`data/bbb/per_frame_quality.json`（BONES 质量表，IMDN 对照来源）；播放器 IMDN 权重与三个外部预训练权重。
- **下游**：PAC-Net 阶段0（档案 #8）直接复用"uniform 16 段留出 + DIV2K warm + 真实链路退化"训练/评测范式；hard_survey（档案 #9）沿用"多档退化×多方法"普查结构但改为全片非相邻抽帧（其 G1 口径事故与本档 Q4 同族、更严重）；"必须在目标退化上训练、困难段才有增益"是 PAC stage2 多退化混合训练的动机之一。
- **叙事文档**：`docs/enhancer_operator_survey.md`（本档对应长文，需按 Q1/Q6 修正）、`docs/current_state.md:143,227-257,428`、`docs/exploration_log.md:437`（阶段17）。

## 11. 复盘备注（复活检查清单）

- **R1（若重启"对症微调"方向，必做）**：val/test 完全不相交的 block 协议重跑——block 模型 val 改用块外固定段（如 0/60/120），测试只用块内段（135-159 中除 val 外的段，如 135/145/155/158），训练命令同 §4；以零泄漏数字替换 +7~14。预计训练 2×约 5 分钟 + 测量约 10 分钟。
- **R2**：跨视频零样本 leave-out（第二内容，如 TOS 或 Tears of Steel）是本档案从未覆盖的最大边界；若未来方向回到"内容自适应增强"，先做这个再谈泛化。
- **R3**：金标准 VMAF（双 rawvideo 连续 libvmaf，范本 `workspace/pacnet/sr_aware_alloc/measure_seg_rd.py:69-145`）统一在帧级指标体系 A 阶段对本档案关键配对（Nano vs IMDN-DIV、LO vs DIV 困难段）复核绝对值。
- **R4**：普通段轻微遗忘可用更小 lr（如 5e-5）或 DIV2K 混合回放缓解——仅在 R1 证实值得继续后再做。
- 已废弃的中间假象：step2b 跨口径比较曾得 +7.24（整段 libvmaf vs 逐帧均值混用），docs 已标注废弃，勿再引用。
