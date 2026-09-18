# PAC-Net 可渐进神经增强网络

面向流媒体"随机算力预算 + 未知压缩退化 + 内容难度不均"的端侧增强网络，与 BONES 选择层解耦的新主线。核心是**单套权重、算力可伸缩**：同一网络在第 2/4/6 个 block 后各设一个递进出口，时间够就深算、到点必出当前最优结果（anytime/可中断）；退化条件化模块让一套权重覆盖任意压缩程度。代码在 `workspace/pacnet/`。

## 与既有结论的关系

- 阶段17（见 `enhancer_operator_survey.md`）证明：对症增强增益严格集中在高运动/强压缩塌陷段，普通段持平。由此需要"困难段深算、简单段浅算、时间不够先出保命档"的单网络多档能力，而非多个固定模型切换。
- 文献空白：anytime/early-exit 动态深度几乎只用于分类/视觉推理，SR/增强侧未被开发；流媒体神经增强系统的创新都在网络外部选帧/选块/选模型，无人改网络内部使其算力可伸缩。

## 网络结构（`pacnet_model.py`）

- 共享骨干：`fea_conv` + 6 个串联 `IMDModule`（IMDB 本身残差），在第 2/4/6 个 block 后各接一个 `ExitHead`（1×1 融合 + LeakyReLU + 3×3 + 浅层残差 + pixelshuffle×4），记 e1/e2/e3。
- `forward(exit_idx=k)` 只跑到第 k 个出口即返回，真正省算力；`exit_idx=None` 返回三出口用于深监督训练。
- `CondEncoder`：从 LR 提 5 维在线统计（RGB 均值 3 + 高频能量 + 下采样残差能量），经 MLP 生成 FiLM gamma/beta 调制浅层特征，末层零初始化保证开局恒等、不破坏 warm。**阶段0先关闭，阶段2再启用验证跨退化。**
- `PlainIMDN(num_modules)`：参数化 block 数的标准 IMDN，用于造与 e1 同算力的固定浅模型对照。

## 阶段0最小证伪（纯离线，不上播放器）

用最小实验验证承重墙假设，数据管线复用 `workspace/enhancer_survey/train_leaveout.py`：真实 4.5× 裁 GT 对齐、uniform 16 段留出、DIV2K 权重 warm、Adam 2e-4、StepLR(15,0.5)、30ep、patch192、batch16、L1。测试为 16 留出段（含困难段 130/140/150）每段 4 帧共 64 帧整帧 PSNR（×4 后 bicubic 到 1080 对 GT）。脚本：`selftest_model.py` 结构自检、`train_pacnet.py` 训练、`eval_pacnet.py` 评测、`analyze_pacnet.py` 判决、`forward_equiv_check.py` 前向等价核验。

### 关键坑：warm 迁移键名（复用价值高，排查耗时最长）

播放器 IMDN 里 `conv_layer` 返回**裸 `nn.Conv2d`**（state_dict 键为 `fea_conv.weight`，无 `.conv`），`conv_block` 返回 **`Sequential`**（融合层键为 `c.0.weight`），`pixelshuffle_block` 内卷积键为 `upsampler.0.weight`。首版按"都带 `.conv`"臆测键名，导致**入口 `fea_conv` 与最深出口重建头全部漏迁、保持随机初始化**，只有 6 个 block 共 84 项迁移成功（恰好等于 84，造成"warm 成功"假象）。随机入口上 block 白算：e3 仅 24.2dB、三出口挤在一起、整体≈bicubic，一度误判 NO-GO。修正后 92 项权重全迁，`forward_equiv_check.py` 证实 PAC e3 与标准 IMDN 同权重下**逐元素 diff=0**。另修两处：`ExitHead.fuse` 必须用带 LeakyReLU 的 `conv_block` 对齐 IMDN 的 `c`（用无激活 `conv_layer` 会让 warm 失配）；`CondEncoder` 不得用 batch 维标准化（单帧推理 batch=1 时统计恒 0、条件机制失效），改固定尺度。

### 结果（16 段 64 帧整帧 PSNR，dB；主版出口损失权重 0.3/0.6/1.0、单调 margin 0.5）

| 方法 | 均值 | 说明 |
|---|---|---|
| base bicubic | 28.407 | 无增强 |
| e1（2-block 出口） | 28.546 | 保命档 |
| e2（4-block 出口） | 28.664 | 中档 |
| **e3（6-block 出口）** | **29.630** | 最深档 |
| plain2（独立 2-block 固定模型） | 29.311 | 与 e1 同算力对照 |
| imdn_div（通用 6-block） | 28.365 | DIV2K 零样本 |
| imdn_lo（对症 6-block 单模型） | 29.917 | 与 e3 同算力上界 |

判决（`eval_pac16_verdict.json`）：

- **H1b 最深档不掉队 PASS**：e3 仅比对症单模型 imdn_lo 低 0.287dB，反超通用 imdn_div 1.265dB——一套权重兼顾三档，深档几乎不付代价。
- **H1c 越深越单调 PASS**：逐帧 e3≥e2≥e1 链比例 98.4%，e3≥e1 为 100%。
- **H2 困难段值得深算 PASS**：困难段 e3−e1 均值 +2.506dB（seg140 +3.17、seg150 +3.47、seg130 +0.88），简单段仅 +0.756。算力花在困难段回报高、简单段浅算即可，anytime 按内容分配算力的核心动机成立。
- **H4 时延单调**：240p 整帧中位 e1=14/e2=26/e3=39ms；e1/e2 在 30fps 的 33ms 帧预算内，e3=39ms 与标准 6-block IMDN 相同（非多出口引入的开销）。
- **H1a 浅出口弹性税（观察项，不否决方向）**：e1 比同算力独立 plain2 低 0.765dB。改等权损失（1/1/1）后 e1 仅回升 0.05、e3 反降 0.12，证明弹性税来自共享骨干要同时服务深出口的结构本质、非损失权重可解；e1 仍优于 bicubic 与通用 6-block。阶段1用自蒸馏/深度课程收敛。

### 阶段0判决：GO

核心承重墙（越深越单调、困难段深算有效、深档基本不掉队）全部成立。**遗留与边界**：浅出口 0.7dB 弹性税待自蒸馏；仅 240p、BBB 单视频（跨视频零样本需第二素材）；CondEncoder 尚未验证跨 QP/跨分辨率；e3 距对症单模型仍差 0.29dB。

## 阶段1：弹性税消除尝试与结构定型（2026-09-12）

目标是把浅出口 e1 相对同算力独立浅模型 plain2 的约 0.77dB 弹性税收窄到 ≥−0.2dB。系统试了六条路径，全部用同一 16 段口径验证：

| 方案 | 做法 | 结果 | 结论 |
|---|---|---|---|
| 等权损失 | 出口权重 1/1/1（阶段0已试） | e1 仅 +0.05、e3 反降 0.12 | 无效 |
| 像素自蒸馏 | e1/e2 向 stop-grad 的 e3 输出对齐，扫 α=0.1/0.3/0.5 | α=0.1 最好；多种子均值仅 +0.065±噪声、e3 微降、α 越大越差 | 边际、噪声内 |
| 深度课程 | 前 10 轮只训 e1 再全开 | e1 纹丝不动、e3 反降 | 无效 |
| 课程+冻结浅层 | 课程后冻结 block1/2 | e1/e3 双输、单调降到 0.64 | 有害 |
| 残差级联（三段隔离） | s2=s1.detach+Δ、段边界截断梯度，三段各只被本出口训 | e3 崩到 28.22、困难增益 2.51→0.24 | 有害 |
| 级联（只保护 e1） | 仅 block1/2 隔离，e2/e3 仍共享 block3–6 | e1=27.55、e3=28.27、单调 0.58 | 有害 |

**关键负结果与机理**：任何梯度隔离都会切断端到端协同——深层被迫在"为浅出口优化、却不利于继续深算"的冻结浅层特征上学习，输入分布漂移、优化不稳，深档质量与困难段增益大幅塌陷，连被保护的浅出口也训不好。多种子（seed 1/2/3，同种子配对 plain2）确认弹性税稳定在 −0.78±0.06dB，是**共享多出口 anytime 结构不可无损消除的固有代价**；要清零只能退回多个独立模型，即失去单网络共享前向、可平滑早退、省三套权重/显存的核心意义。

**结构定型**：采用原始 PACNet（不挂自蒸馏/级联）。理由：深档最高（e3 29.63）、单调最好（链 0.984）、困难增益最大（+2.51）；自蒸馏收益在种子噪声内且微降 e3，不值得引入复杂度。弹性税按阶段0校准口径属观察项（门槛 −1.0），实测 −0.75 在界内，且 e1 仍优于 bicubic(28.41) 与通用 6-block(28.37)。跨种子最稳的正面结论：困难段深算增益 **+2.46±0.05dB**、深档 e3≈29.55–29.60（仅逊对症单模型约 0.3dB）。脚本：`run_stage1.py`（α 扫描）、`run_stage2_curriculum.py`、`run_stage3_cascade.py`、`run_stage3b_protect.py`、`run_stage4_multiseed.py`、`selftest_cascade.py`（级联梯度隔离自检）；结果 `stage1_sweep/stage2_sweep/stage3_sweep/stage3b_sweep/stage4_multiseed.json`。两个实现陷阱已记入 `exploration_log.md`：增量头零化最后一层会切断梯度（应小尺度 ×0.01）；ExitHead 拼接全部历史 bis，段隔离须同时对历史 bis 与 fea detach。

## 阶段2：跨退化泛化与条件化验证（2026-09-13，负结果+设计准则）

**目标**：验证一套权重跨分辨率/压缩强度泛化（H3），以及 CondEncoder 在线退化统计条件化是否有用。用播放器已有的 5 档分辨率视频（240/360/480/720/1080p，帧数一致、天然携带不同压缩强度），训练混合 240/360/480p，留出 720p 作未见退化。三配置对照：A 单退化专用（240p cond=False，复用阶段1主版权重）、B 多退化无条件（混合 cond=False）、C 多退化+条件化（混合 cond=True，CondEncoder 5 维在线统计 FiLM）。各 30ep，16 段每段 1 帧（j=75）整帧 PSNR。脚本 `workspace/pacnet/run_stage2_degradation.py`，结果 `stage2_cross_res.json`。

**e3 深档 PSNR（行=退化档，训练档 240/360/480，未见 720）**：

| 退化档 | A 单退化 | B 混合无条件 | C 混合+条件化 | base bicubic |
|---|---|---|---|---|
| 240p（训练） | 29.494 | **29.633** | 29.526 | 28.226 |
| 360p（训练） | 32.139 | **32.725** | 32.634 | 30.674 |
| 480p（训练） | 34.002 | **34.616** | 34.498 | 32.549 |
| 720p（未见） | 36.326 | **36.925** | 36.709 | 35.784 |

e1 保命档趋势一致（B≥C≥A，720p e1 C 36.677 与 B 36.656 噪声内持平）。

**结论**：
1. **多退化混合训练（B）在所有档都优于单退化专用模型（A）**，包括训练过的 240p（+0.14dB）和未见的 720p（+0.60dB）。一套权重混合训练足以跨分辨率/压缩泛化，不需要为每档存一个模型——这是 anytime 单网络落地的关键前提。
2. **在线统计条件化（C）未带来额外增益，所有档略低于 B**（训练档 −0.09~−0.12，未见 720p −0.22）。CondEncoder 在这个退化范围（240–720p 四档）和表示（RGB 均值/高频能量/下采样残差能量，固定尺度 ×8.0）下是多余的。
3. **负结果价值**：混合训练已提供足够泛化；条件化可能需要更大退化范围（多 QP × 多分辨率）或更好的条件表示（码率/QP 标签而非在线统计）才会显现。当前定型采用 B（多退化混合、无条件化），省掉 CondEncoder 的复杂度与优化难度。

**工程踩坑**：720p 大输入（1280×720→模型 ×4 输出 5120×2880）在多模型共存或评测环境下触发 cuDNN 慢速路径（诊断 A 11s/帧、B 108s/帧，干净环境单模型仅 62ms）。解决：逐个模型加载评测（GPU 只留一个模型）+ `forward_tiled` 分块推理（tile=640、重叠 16 取平均，小分辨率自动直通，720p 切 4 块稳定在 ~170ms/帧）。另修：val_pac 二元组解包（多退化缓存为五元组，验证集转二元组）、翻转增强一行方向写反。

## 后续路线

1. ~~结构定型~~（阶段1已完成）：弹性税为 anytime 共享结构固有代价，定型原始 PACNet。
2. ~~退化条件化~~（阶段2已完成）：多退化混合训练已足够跨分辨率泛化；在线统计条件化负消融，定型不挂 CondEncoder。
3. ~~halt 控制器 + 接播放器~~（阶段3已完成）：离线仿真验证低预算内容感知+0.49有效质量；真实播放器单trace QoE+2.46(+4.2%)，fast_complete降为0。
4. **多轨迹验证 + halt阈值调优**（进行中）：3G/4G各3-5条轨迹验证增益稳定性，safe_factor/gain_thresh网格搜索。
5. 时域一致性（插帧/抗闪烁，第二曲线）。
6. 系统对照与跨视频验证（第二视频需手动提供）。

## 阶段3：halt控制器离线仿真 + 播放器集成（2026-09-13~14，方向成立+首次正向）

### 3.1 离线halt调度仿真

**目标**：在接播放器前，用真实增强时延+全段内容难度快速验证"帧级halt调度在随机时间预算下能否比固定出口/greedy更优"。

**数据**：先生成全159段per-seg e1/e2/e3 PSNR（`workspace/pacnet/per_seg_psnr.json`，每段j=75帧）。深算增益gain=e3-e1：mean=1.166 std=1.004 min=0.150 max=4.702 p25=0.546 p75=1.127，内容差异显著（困难段深算增益是简单段的5-10倍）。

**仿真**（`run_stage3_halt_sim.py`）：1590帧（10周期），3预算水平（low=20ms/帧、mid=28ms、high=36ms），对比fixed_e1/e2/e3、greedy（能e3就e3否则e2否则e1）、内容感知多阈值（CA_t1.0/1.5/2.0、激进CAv2_t1.5/2.0）、自适应（ADA/ADAv2）。时延模型e1=14/e2=26/e3=39ms（阶段0实测）加±15%噪声；账户模型每帧+=预算，选出口后-=时延，账户<−50ms掉帧（有效质量=PSNR×(1−drop_rate)）。

**结果**（`stage3_halt_sim.json`）：

| 预算 | 最优策略 | 有效质量 | vs greedy | vs fixed_e3 | 掉帧率 |
|---|---|---|---|---|---|
| low 20ms | CAv2_t2.0 | 29.321 | +0.494 | +7.601 | 1.89% |
| mid 28ms | CA_t1.0 | 30.132 | +0.134 | +5.712 | 0.13% |
| high 36ms | greedy（全e3） | 30.518 | — | +0.005 | 0% |

**结论**：方向成立。预算紧张时内容感知halt调度有明确收益（低预算+0.49有效质量、掉帧减半）；预算充足时全深算即可（内容感知没必要，自适应退化为greedy）。这验证了"anytime网络+内容感知halt"的核心价值——不是在所有情况下都更好，而是在资源紧张时避免fixed_e3的大量掉帧、同时比greedy更聪明地分配深算预算。

### 3.2 播放器集成与首次对比

**集成**：新增`player/istream_player/modules/enhancer/pacnet_enhancer.py`（注册模块名`pacnet`），继承IMDNEnhancerAlways。核心重写：
- `load_model`：单PACNet模型（多退化混合权重stage2_mix_nocond_x4_best.pth），替代原来三个独立IMDN模型
- `measure_latency`：遍历所有分辨率测e1/e2/e3延迟，返回5×5表（缓存pacnet_latency.json）
- `enhance_one_frame`：支持exit_idx参数，model(tensor, exit_idx=k)只跑到指定出口省算力
- `run`：帧级halt控制器——根据剩余时间预算+per-seg内容难度gain决定出口。策略：预算充足全e3；够e3但不充足时困难段e3/简单段e2；够e2时困难段e2/简单段e1；极少保命e1。阈值safe_factor=1.5、gain_thresh=1.5
- `get_quality_table`：返回pacnet_quality.json（三出口校准VMAF表，用原IMDN no_enh作基准+测量增益校准）
- headless模式下不存完整CPU tensor（result.append(None)），解决内存积压从23GB降到1.3GB

**工程踩坑**：
1. 首次运行因task_total未设置（父类run里设置但重写版漏了）导致remain_task()报TypeError
2. 首次覆盖get_latency_table/get_quality_table返回全e2延迟/全30质量，导致BONES全选最低码率(avg_bitrate=0)、VMAF=-Inf、QoE=NaN——删除覆盖方法复用父类实现后修复，后改为返回正确的PAC-Net表
3. headless模式下每段增强结果(120帧×24MB≈2.9GB CPU tensor)积压导致内存23GB进程被OOM杀掉，改为不存完整tensor后内存1.3GB
4. VMAF测量绝对值比原IMDN表高30分（ffmpeg libvmaf默认模型版本差异），用原表no_enh列作基准+测量增益校准

**VMAF精确测量**（`workspace/pacnet/measure_pacnet_vmaf.py`，16段均匀采样，ffmpeg libvmaf）：

| 码率档 | no_enh(原表) | e1 | e2 | e3 | 原IMDN level2 | 原IMDN level3 |
|---|---|---|---|---|---|---|
| 240p | 39.30 | 45.94 | 48.91 | 54.42 | 54.0 | 60.8 |
| 360p | 64.02 | 68.32 | 70.63 | 73.05 | 75.9 | 79.9 |
| 480p | 76.80 | 79.31 | 79.87 | 80.25 | 85.2 | 87.2 |

PAC-Net e3（6-block nf=32）与原IMDN level2（同结构）VMAF接近，符合预期；e1/e2提供渐进的质量-算力折衷。

**首次对比**（3G trace report.2010-09-13_1003CEST，BBB视频，headless，BONES码率选择）：

| 指标 | 基线 imdn_always | PACNet | 差值 |
|---|---|---|---|
| QoE | 58.50 | **60.96** | **+2.46 (+4.2%)** |
| avg_vmaf | 66.82 | **68.72** | **+1.91** |
| oscillation | 8.30 | 7.75 | -0.55（更稳） |
| dur_stall | 19.7s | 18.7s | -1.0s |
| num_switches | 55 | 48 | -7 |
| fast_complete段 | 3 | **0** | halt控制器让增强跟上播放 |

出口使用e3=74%、e1=9%（段最后一帧统计），平均增强FPS=101远超实时30。GPU功耗76.6W vs 47.9W（e3使用更多）。

**直观解释**：原来的IMDN增强器在时间紧张时触发fast_complete（剩余帧直接不增强，质量跳崖），而PAC-Net的halt控制器主动降级到e1/e2省时间——既保证不卡顿，又没有完全放弃增强，所以质量更高、波动更小、fast_complete降为0。这正是anytime网络的核心价值。

**注意事项**：①单条trace结果，需多轨迹验证；②halt控制器阈值未调优；③VMAF用校准后的PAC-Net表，BONES决策更准确。

### 3.3 跨视频验证：Tears of Steel（2026-09-17）

用未参与训练的开源真人电影 Tears of Steel（Blender，1920×800，24fps，184 段×4s，2.40:1）检验泛化，与 BBB 完全独立。源视频 `E:\dev\Data\tears_of_steel_1080p.mp4`，5 档分辨率（426×176/640×268/854×356/1280×532/1920×800）用 ffmpeg dash muxer 一步编码+分段（`-g 96 -keyint_min 96 -sc_threshold 0`，96 帧/段），脚本 `workspace/pacnet/encode_tos_dash.bat`；DASH 产物在 `player/data/tears_of_steel/`，编码源 mp4 在项目根 `data/tears_of_steel/`。

**多数据集支持（播放器改造）**：`PlayerConfig` 新增 `video_dataset`（默认 bbb）；基类 `setup/get_quality_table/measure_latency` 与 PACNetEnhancer 的 `get_quality_table/measure_latency/load_model`（per-seg gain）均按该字段切换文件，TOS 对应 `imdn_tos_quality.json`、`pacnet_tos_quality.json`、`*_latency_tos.json`；修复 `get_frame_rate` 不支持分数帧率（`24/1`）的崩溃。运行必须显式传 `--input data/tears_of_steel/tos.mpd`（命令行默认值会覆盖 YAML）。

**离线 VMAF（TARGET 1920×800，16 帧均匀采样，libvmaf）**：

| 码率档 | bicubic | PAC e1 | PAC e2 | PAC e3 | IMDN l1 | l2 | l3 |
|---|---|---|---|---|---|---|---|
| 240p | 80.43 | 86.18 | 87.12 | **90.22** | 87.02 | 88.70 | 89.64 |
| 360p | 94.26 | 96.51 | 97.62 | **99.02** | 96.43 | 97.26 | 98.11 |
| 480p | 97.92 | 99.70 | 99.80 | 99.84 | 99.53 | 99.84 | 99.84 |

两个关键判断：①TOS 上 PAC e3 与通用 IMDN l3 打平（240p 仅 +0.58），而 BBB 上 PAC e3 领先 l3 达 8 分——BBB 的大差距主要是 PAC 在 BBB 数据训练过的数据偏置，TOS 才是公平泛化对比；②浅出口弹性税跨视频仍在（e1<l1、e2<l2），三出口间差异小（e1→e2 仅 0.94，BBB 为 2.97），halt 内容感知选择空间更小。

**播放器 3G 三轨迹对比（headless，BONES，fast_complete 开；汇总 `summarize_tos.py`）**：

| trace | PAC QoE | IMDN QoE | ΔQoE | ΔVMAF | Δ卡顿(s) |
|---|---|---|---|---|---|
| 09-13 | 91.49 | 90.19 | +1.30 | +0.41 | +13.1 |
| 09-14 | 90.98 | 88.99 | +1.98 | +1.35 | −1.3 |
| 09-21 | 88.68 | 87.06 | +1.61 | +1.64 | −9.9 |
| 平均 | 90.38 | 88.75 | **+1.63** | +1.13 | +0.6（打平） |

3/3 胜，质量切换 48 vs 69，PAC fast_complete 全 0。PAC 几乎全程 e3（172/177/183 段），基线主要 l2（108/135/156 段）。

**结论与边界**：跨视频泛化成立（新视频、未训练仍稳定占优），但 TOS 上 halt 的 anytime 算力伸缩未被激发——24fps 帧间隔 41.7ms > 240p e3 端到端约 3.8s/段（<4s 段时长），算力相对充足，halt 几乎不降档，PAC 退化为“更深的固定模型”靠 e3>l2 质量取胜。要验证 halt 的核心卖点（时间紧张时主动降档避免卡顿/质量跳崖），需算力紧张场景：更高帧率（30/60fps）、真 1080p 及以上目标、弱 GPU 或紧算力预算。另：`measure_latency` 仅测 model forward，360p（宽 640 对齐）cuDNN 走快速路径 e3=5.6ms，远快于 240p 33.8ms，实际播放 fps≈126 印证为真实硬件特性；halt/BONES 用的延迟未含解码、resize、H2D/D2H 与调度开销，在“e3 延迟逼近帧间隔”的临界场景偏乐观，是后续可改的测量一致性问题。

### 3.4 阶段二：算力紧张下的 halt 验证（2026-09-18，gpuburn + 可调度性前瞻 + 两个埋点 bug 修复）

TOS 满血时 halt 不降档，阶段二用每帧 dummy matmul 抢占 GPU 人为制造算力紧张，检验 halt 能否“既流畅又比固定档高清”。最新三组数据见 `current_state.md`，本节只沉淀机制与判据。

**算力负载注入（`_gpuburn.py`）必须用乘法系数，不能用固定加法。** 环境变量 `PACNET_BURN_K=1.5` 时，enhance_one_frame 用 cuda 同步实测该帧模型前向 `fwd_ms`，`burn(fwd_ms)` 额外烧 `fwd_ms×(K−1)` 的 2048² matmul，端到端≈×K。出口越深、前向越久烧得越多，三出口延迟比值保持 1.397 不变。旧的 `PACNET_EXTRA_MS` 对每帧固定加 N ms，会把延迟比压到 1.177，使 halt 阈值 l3×1.2 反低于 l2×1.6、e2 档永远选不到（e2 死档），且与离线 `sim_tos_halt.py` 的乘 k 口径不一致。乘法对 PAC/IMDN 同公式天然公平；halt 用的静态延迟表在 `measure_latency` 缓存分支同步 ×K。

**帧级 VMAF 埋点（否则 halt 降档画质代价完全不可见）。** PACNet 重写 run 后段统计必须回传 `exit_frame_counts=[e1,e2,e3]`（段内三出口实际增强帧数）与 `fast_complete_triggered`；`analyzer.py` 按帧占比算段 VMAF：`real_vmaf = (Σ cnt[i]·q[i+1]·n_enh + base·n_non)/(n_enh+n_non)`，截断未增强帧按 base 计。否则 analyzer 用 BONES 下发动作（恒 3）查表，降 e1 的帧也按 e3 打分，VMAF 虚高约 4 分/段，“0 卡顿+画质反超全 e3”是假象。

**halt 判据三版迭代（`pacnet_enhancer.py:_halt_decide`），前两版均失败：**
1. 段内平均预算：`budget=(S−elapsed)/剩余帧`，要求剩余帧全程跑某档才选它。开局 S≈3.94s 均分 96 帧仅 41ms < e3 门槛 60.8ms，过度保守，87% 帧误降 e1（实测 dist 84/4/8）。
2. 库存水位（对齐离线 sim）：只比全局库存 lead 与单帧门槛（≈0.06s）。但真实播放器有独立的 fast_complete，其动态阈值（≈0.8s）远高于 halt 降档门槛，lead 下降时 fast_complete 先触发、整段截断（只做 60/96 帧），halt 的 e2/e1 根本来不及执行——离线 sim 没有 fast_complete 这层，不能照搬其库存判据。
3. **可调度性前瞻（采用，EDF 式）**：逐级做可行性检查，选满足 `lead ≥ l_exit × 剩余帧 × 安全系数` 的最高出口，安全系数默认 1.0（`PACNET_E3_THRESH/E2_THRESH` 可覆盖），fast_complete 退为“连 e1 都不可行”时的最终兜底。开局判全程 e3 来不及（4.87s>S）但全程 e2 来得及（3.49s），于是 e1/e2 混合且 96 帧全增强不截断；中段库存厚（lead≈21s）全程 e3。

**K=1.5 / trace 09-14 单轨迹结论**：halt 前瞻 QoE 90.50、VMAF 92.61、卡顿 2.46s，显著优于强制全 e3（QoE 88.32、卡顿 8.67s、77 段截断），也略优于 IMDN 基线（QoE 89.12、VMAF 92.12、卡顿 0.06s）。anytime 降档保命价值相对全 e3 明确；相对 IMDN 仅 +0.49 VMAF 且多 2.4s 卡顿，优势薄，需多轨迹定胜负。残留卡顿与安全系数扫描、多轨迹胜率是下一步。

