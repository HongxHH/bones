# 增强算子离线增益普查（阶段17）

## 要回答的问题

阶段14（段级）、15（帧级）、16（服务端引导）三轮证伪后，判断"选择层无差异可挖"的根因是否在增强地基：现有 IMDN 单帧超分的增益曲线对级别单调、段平均鲁棒。本普查在**完全同口径**下离线测量多种替代增强算子的逐段 VMAF，回答两个决定方向生死的问题：

1. 换一个更强/更对症的增强算子，整体增益能否显著高于 IMDN；
2. 不同算子的相对优势是否**内容非单调**（不同段各有胜负）——这是"按内容选算子/多算子协同"能否成立的前提。

不跑完整视频流、不碰播放器，只做离线测量。

## 同口径测量方法

复用 `workspace/measure_per_frame_vmaf.py` 的口径，脚本 `workspace/enhancer_survey/step1_operator_gain_survey.py`：

- 源帧从 `data/video_mp4box/output_{240,360,480}p.mp4` 按段定位（120 帧/段，30fps），GT 为 `output_1080p.mp4`；
- base = 双三次放大到 1920×1080；所有算子输出统一 bicubic 对齐到 1080；
- 逐帧 ffmpeg libvmaf（enh/ref 都经相同 cv2 mp4v 临时编码），120 帧取均值；
- IMDN 对照优先读 `data/bbb/per_frame_quality.json`，需要时在同一管线现算。

**口径自检（结果可信的前提，全部通过）**：本管线重测 base 与 json 内 base 逐段 diff=0.000；现算 IMDN-high 与 json sr_high diff≈0（≤0.2，临时编码波动）。跨文件比较存在系统偏差——`data/bbb/measure.py` 的整段 libvmaf 口径会系统性低于逐帧均值口径，**禁止跨口径直接比大小**（step2b 曾因此得到 +7.24 的假象，已废弃，以 step3 同口径结果为准）。

样本：240p 全片均匀 16 段（seg 0,10,…,150），另用 seg 2/60/129 做小样本快速验证。

## 被测算子

| 算子 | 类型 | 权重/来源 | 训练退化 | 时序 |
|---|---|---|---|---|
| IMDN-high-bbb | 轻量单帧超分 | `player/.../imdn_weight/IMDN_high_bbb_x*.pth` | DIV2K 后在 BBB 微调 | 否 |
| IMDN-high-DIV2K | 轻量单帧超分 | `IMDN_high_DIV2K_x*.pth` | DIV2K 通用 | 否 |
| NAFNet-width32 | 同尺寸复原 | `enhance/NAFNet/.../NAFNet-GoPro-width32.pth`（17.1M） | GoPro 运动模糊 | 否 |
| STDF（MFQEv2） | 压缩视频复原 | `enhance/stdf-pytorch/exp/.../ckp_290000.pt`（0.365M） | HEVC HM16.5 LDP QP37 | 7 帧 |
| NanoVSR-644k | 轻量视频超分 | `enhance/nanovsr/checkpoints/nanovsr_644k.pth`（0.644M，ECCV'26） | REDS+Vimeo bicubic | 双向循环 |

适配实现：STDF 官方依赖需编译的 DCNv2，用 `workspace/enhancer_survey/dcn_nocompile.py` 的 grid_sample 免编译分组调制可变形卷积替代，单测验证 offset=0、mask=1 时与 `F.conv2d` 误差 <1e-5，官方权重 strict 加载零缺失；NanoVSR 固定 4×，T=15 分窗。

## 结果（240p，同口径）

整体均值（16 段）：

| 方案 | 均值 VMAF | 相对 base |
|---|---|---|
| base（bicubic） | 40.27 | — |
| NanoVSR-644k（泛化 VSR） | 51.94 | +11.67 |
| IMDN-high-DIV2K（泛化 ISR） | 54.99 | +14.72 |
| IMDN-high-BBB（目标视频微调） | 61.10 | +20.83 |

两两对比：

| 对比 | 均值差 | 逐段胜负 | 判定 |
|---|---|---|---|
| NanoVSR − IMDN-BBB | −9.16 | 0 赢 / 16 输 | 现成最新 VSR 打不过微调版 |
| NanoVSR − IMDN-DIV2K | −3.05 | 1 赢 / 15 输 | 公平泛化下仍略输单帧 ISR |
| IMDN-BBB − IMDN-DIV2K（主场红利） | +6.11 | — | 同一架构仅因见过 BBB 就高 6 分 |

小样本（seg 2/60/129，相对各自对照）：

- NAFNet 去模糊单独 vs base：+4.64 / −9.35 / −0.48，输出值域一度到 ±51（分布外），全面跑输 IMDN；
- STDF 去压缩块单独 vs base：−8.08 / −5.11 / −2.01；
- STDF→IMDN 级联 vs 直接 IMDN：−8.5 / −9.0 / −3.0（前置去块平滑改变 SR 输入分布，反而劣化）。

内容相关性：NanoVSR−IMDN-DIV2K 的差值与段运动强度 Spearman −0.21、与纹理复杂度 −0.21；NanoVSR−IMDN-BBB 为 −0.61/−0.52（越复杂输得越多）。片尾高运动高纹理段（seg140/150，motion≈25、texture 3000~5000）NanoVSR 输 31~36 分，时序隐式对齐在压缩+快运动下失效最严重。

## 根因

1. **退化不匹配是主因，不是模型新旧**。NAFNet 训于运动模糊、STDF 训于 HEVC、NanoVSR 训于干净 bicubic 下采样，而真实链路是 H.264 压缩 + 低分辨率 + 实时；通用预训练模型把压缩噪声当细节，时序模型还会沿时间轴传播噪声。ECCV'26 的 NanoVSR 同样跑输 2019 的 IMDN-DIV2K，证明"换更新的现成模型"不解决问题。
2. **同尺寸复原不补分辨率**。去模糊/去块只在原尺度修伪影，再 bicubic 放大依然缺高频，在"低清→高清"链路里天然弱于直接超分；前置去块再超分也因改变输入分布而失败。
3. **高运动段是时序方法的灾难区**，单帧 ISR 不受运动影响反而稳，因此跨算子也未出现稳定的"各有胜负段"，非单调选择空间不成立（3 段小样本曾见 NanoVSR 赢 seg2，扩到 16 段后确认是噪声）。
4. **BONES 质量表存在部署假设脱节**。其增强质量用"在目标视频上微调过的 IMDN-BBB"测得，同口径主场红利 +6.11；真实流媒体无法对每个未知视频预先微调，零样本泛化才是实际工况，该口径下增强增益被系统性高估。

## 结论与边界

**结论**：直接接入现成预训练增强算子（去模糊/去压缩块/通用 VSR）无法抬升增益地基，也不产生内容非单调的选择空间。对症微调闭环进一步证明：IMDN-BBB 的主场红利**不是纯逐段记忆**，其中"对症真实 H.264 压缩退化"的部分可跨段、跨连续块泛化（block 口径困难段 +7~+14 VMAF），但增益严格集中在通用模型塌陷的高运动/强压缩段，普通段持平。由此得到两条确定结论：其一，增强地基要立得住，必须在目标链路退化上训练，零样本通用模型在困难段必然塌陷；其二，"哪里值得增强"存在稳定、由内容（运动×压缩难度）驱动的信号——困难段才是增强产生价值之处，这为后续内容自适应的选择性增强/质量预算分配提供了实证地基。

**边界**：仅 240p、BBB 单视频；跨块泛化已验证，跨视频（inter-video）零样本泛化未验证，需第二视频素材；未测 360/480p 与非 4× 缩放路径；VMAF 单指标。

## 对症微调闭环：跨段泛化是真泛化还是逐段记忆

脚本 `workspace/enhancer_survey/train_leaveout.py`，从 IMDN-DIV2K warm-start，训练对取**真实链路退化**（LR=`output_240p.mp4` 真实 H.264 流，GT=`output_1080p.mp4`，同帧号对齐），L1、Adam lr2e-4、HR patch192、40 epoch。两种留出口径：

- uniform：每 10 段留 1（seg 0,10,…,150），测试段不训练；
- block：连续块留出（seg 28-32、88-93、135-159 整块不训练），测试段的相邻段也不进训练集，用于排除"时间相邻帧近邻记忆"。

**关键工程教训（训练对齐）**：240p(426×240)→1080p(1920×1080) 的真实缩放比是 sy=4.5、sx=4.507，不是整数 4。最初按 `4×` 坐标裁 GT patch，LR patch 与 GT patch 仅 10.7dB（严重错位），模型学错位映射被训崩（val PSNR 15dB、VMAF 精确为 0）；按真实 4.5× 裁 GT 再 resize 到模型 4× 输出网格后对齐度 27dB，训练恢复正常（train L1 0.12→0.020，留出段 PSNR 单调上升）。两视频同帧号对齐本身正确（同帧号 bicubic PSNR 最高，±1 帧明显更低）。

uniform 留出 16 段均值：base 40.27 / DIV2K 54.99 / **LO 57.92** / BBB 61.10。LO−DIV2K 均值 +2.93，但中位 −0.73、赢 5 输 10，增益几乎全部来自三个高运动塌陷段（seg130 +8.5、seg140 +28.4、seg150 +25.5），其余普通段 LO 普遍小幅输 DIV2K（−0.04~−2.04，轻微通用能力遗忘）。

block 连续块留出（连相邻段都不训练）揭示真泛化与近邻记忆的拆分：

| 段（类型） | DIV2K | LO-uniform | LO-block | BBB | block−DIV（真泛化） |
|---|---|---|---|---|---|
| seg30 普通 | 60.96 | 58.93 | 60.46 | 61.89 | −0.51 |
| seg90 中运动 | 47.43 | 45.61 | 45.95 | 50.89 | −1.48 |
| seg140 高运动塌陷 | 35.57 | 63.97 | 49.55 | 65.06 | **+13.98** |
| seg145 块内 | 53.69 | — | 57.02 | — | +3.34 |
| seg150 高运动塌陷 | 35.56 | 61.05 | 42.82 | 65.72 | **+7.26** |
| seg155 块内 | 64.32 | — | 64.59 | — | +0.27 |

判决：

1. **对症训练可跨段、跨连续块泛化，不是纯逐段记忆**。片尾 seg135-159 整块（含相邻段）都不训练，模型在最困难的 seg140/150 仍比通用 DIV2K 涨 +14/+7.3，这部分来自"学会该链路 H.264 压缩退化"，可迁移到没见过的同分布段。
2. **uniform 口径把困难段增益高估约一半**：seg140 的 +28.4 中约 +14 来自时间相邻段近邻记忆，真泛化 +14；seg150 同理（+25.5 中真泛化 +7.3）。严谨数字以 block 口径为准。
3. **增益严格内容选择性**：只在通用模型塌陷的高运动/强压缩困难段为正，普通段持平到轻微为负。对症微调的价值是补齐通用模型的塌陷短板，而非全面抬升。
4. 距整片过拟合上界 BBB 仍有差距（seg140 block 49.55 vs BBB 65.06），剩余红利确需见过该具体场景。

**未消除的边界**：block 仍只证明同一视频 BBB 内跨块泛化（intra-video），不是跨视频零样本（inter-video）；真实流媒体遇到全新视频能否保持需第二个视频验证（外网直链多次失败，待手动提供素材）。仅 240p；普通段轻微遗忘可用更小 lr 或混入 DIV2K 数据缓解。

> **2026-10-10 档案复核补注（checkpoint 选择泄漏）**：block 训练 best epoch 按 val 段 [30,90,140,150] 的 RGB PSNR 选取（`train_leaveout.py:105-106,131-133`），而上表 seg140/150 正是 val 段，+13.98/+7.26 含间接选择泄漏（按 PSNR 选 epoch、按 VMAF 报告；val 每段仅 3 帧）。零泄漏的块内段只有 seg145（+3.34）、seg155（+0.27，n=2）；uniform 口径排除 val 段（30/90/140）后干净 13 段 LO−DIV 均值 +1.72、中位 −0.71、4 赢 8 输。"能跨块、增益只在困难段"定性保留，+7~14 不得直接引用；严谨重测见 `workspace/enhancer_survey/README.md` §11 R1。同期复核：STDF/NAFNet/级联小样本负数已重跑落盘 `step1b_stdf_nafnet.json`（与上文 ±0.03 一致）；`step4_analyze.py` 对 block json 的 KeyError 已修复；`*_ms` 字段含 VMAF 全程墙钟、非模型延迟；VMAF 经 cv2 mp4v 双臂打包，绝对分待 rawvideo 金标准复核（帧级指标 A 阶段）。

## 产物索引

- `workspace/enhancer_survey/step1_operator_gain_survey.py`：同口径普查主脚本（算子可插拔）
- `workspace/enhancer_survey/dcn_nocompile.py`：免编译 DCNv2（含等价性单测）
- `workspace/enhancer_survey/stdf_model.py`：TV 版 STDF 网络
- `step2_analyze.py`/`step3_analyze.py`：非单调与同口径公平对比分析
- `step2_survey16.json`/`step3_fair16.json`：16 段原始测量结果
- `step1b_stdf_nafnet.json`：seg2/60/129 的 STDF/NAFNet/级联小样本（2026-10-10 复核重跑落盘）
- `train_leaveout.py`：leave-segments-out 对症微调（`--holdout uniform|block`，按真实 4.5× 对齐裁 GT）
- `step4_analyze.py`：闭环泛化判决分析
- `step4_leaveout16.json`（uniform）、`step4b_block6.json`（block）：留出段对照结果
- `weights/IMDN_high_bbbLO_x4*.pth`（uniform）、`IMDN_high_bbbLO_block_x4*.pth`（block）及各自 `_trainlog.json`
