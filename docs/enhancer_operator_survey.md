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

内容相关性：NanoVSR−IMDN 的差值与段运动强度 Spearman −0.21、与纹理复杂度 −0.21；片尾高运动高纹理段（seg140/150，motion≈25、texture 3000~5000）NanoVSR 输 31~36 分，时序隐式对齐在压缩+快运动下失效最严重。

## 根因

1. **退化不匹配是主因，不是模型新旧**。NAFNet 训于运动模糊、STDF 训于 HEVC、NanoVSR 训于干净 bicubic 下采样，而真实链路是 H.264 压缩 + 低分辨率 + 实时；通用预训练模型把压缩噪声当细节，时序模型还会沿时间轴传播噪声。ECCV'26 的 NanoVSR 同样跑输 2019 的 IMDN-DIV2K，证明"换更新的现成模型"不解决问题。
2. **同尺寸复原不补分辨率**。去模糊/去块只在原尺度修伪影，再 bicubic 放大依然缺高频，在"低清→高清"链路里天然弱于直接超分；前置去块再超分也因改变输入分布而失败。
3. **高运动段是时序方法的灾难区**，单帧 ISR 不受运动影响反而稳，因此跨算子也未出现稳定的"各有胜负段"，非单调选择空间不成立（3 段小样本曾见 NanoVSR 赢 seg2，扩到 16 段后确认是噪声）。
4. **BONES 质量表存在部署假设脱节**。其增强质量用"在目标视频上微调过的 IMDN-BBB"测得，同口径主场红利 +6.11；真实流媒体无法对每个未知视频预先微调，零样本泛化才是实际工况，该口径下增强增益被系统性高估。

## 结论与边界

**结论**：直接接入现成预训练增强算子（去模糊/去压缩块/通用 VSR）无法抬升增益地基，也不产生内容非单调的选择空间。"挑模型/调选择算法"是死路，真正杠杆是让增强**对症 H.264 压缩流退化**——在压缩数据上训练/微调，或做压缩感知（显式建模 QP/码率/压缩噪声）的增强。

**边界**：仅 240p、BBB 单视频、16 段抽样、VMAF 单指标；未测在 H.264 压缩对上训练后的模型（这正是下一步）；未测 360/480p 与非 4× 缩放路径。

## 下一步（待确认）

对症训练证伪实验：用 BBB 的 1080p 当 GT、ffmpeg 按目标码率压出 H.264 LR 构造训练对，**leave-segments-out**（训练段/测试段严格隔离，避免重蹈整片微调的虚高），微调一个轻量 ISR/VSR，测跨段泛化增益能否稳定超过 IMDN-DIV2K、并逼近或超过 IMDN-BBB。若能，则"压缩对症增强地基"成立，再谈选择/协同；若不能，增强地基路线需重新评估。

## 产物索引

- `workspace/enhancer_survey/step1_operator_gain_survey.py`：同口径普查主脚本（算子可插拔）
- `workspace/enhancer_survey/dcn_nocompile.py`：免编译 DCNv2（含等价性单测）
- `workspace/enhancer_survey/stdf_model.py`：TV 版 STDF 网络
- `step2_analyze.py`/`step3_analyze.py`：非单调与同口径公平对比分析
- `step2_survey16.json`/`step3_fair16.json`：16 段原始测量结果
