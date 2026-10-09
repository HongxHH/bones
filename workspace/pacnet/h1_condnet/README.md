# 档案：H1 codec 元数据条件网 / H2 光流 warp 邻帧融合 / H3 选择性原生 tile（三路证伪批次）

- 状态：已封板（三路 NO-GO，数字 2026-10-10 全部现算复核）
- 时间：实验 2026-09-19；复核 2026-10-10
- 最近复核：2026-10-10（全量读码 + 三个 eval json / warp_oracle.json / roi_oracle.json 现算）

> 本档案合并三个独立但同日、同动机、同口径的证伪探针。详细叙事在 docs 四篇：
> `docs/h1_codec_metadata.md`（数据通路）、`docs/h1_condnet_falsification.md`（条件网）、
> `docs/h2_warp_oracle.md`、`docs/h3_roi_tile_oracle.md`。本文件只做档案化：文件清单、
> 复核发现、影响面、是否重做、复活条件，不重复叙事。

## 1. 要回答的问题

PAC-Net（档案 #8）与 hard_survey（档案 #9）把"加深/多出口/换更难单帧退化"三条路证伪后，
瓶颈被定位为**压缩/下采样后幸存的输入像素不足**。本批次依次测试四个"加信息"杠杆：

- H1：零带宽的解码器元数据（QP、前/后向运动矢量）作为条件通道喂轻量 SR，能否提升质量？
- H2：用相邻帧经稠密光流 warp 做非学习时序融合（借邻帧像素），能否赢单帧 l2？
- H3：把关键空间区域换成 1080p 原生 tile、其余仍低码率+l2，等码率下能否赢"均匀提分辨率+l2"？

## 2. 方法概述

- H1 数据通路（`h1_codecmeta/`）：PyAV `flags2|=1<<28` 读 MOTION_VECTORS side data（40 字节
  AVMotionVector，位移=motion/4）；`ffmpeg -threads 1 -debug qp` 文本解析逐 16×16 块 QP；
  结论 h264 只导出 MV（+SEI），**残差不可得**。
- H1 条件网（`h1_condnet/`）：l2 骨干 IMDN_RTC（nf=12/modules=5，24114 参数），两种注入
  （add 特征相加 +2076 参数；cat 首层加 7 通道零初始化 +756 参数，零条件与热启动逐位相等），
  三臂 base/cond/shuf（shuf 乱序条件训练、评测喂真条件，容量对照）。BBB 6 退化视频训练、
  TOS 7 档全程留出。
- H2（`h2_warp_oracle/`）：cv2 DIS 稠密光流（优于解码器块级 MV，慷慨上界）把 ±1/±3 邻帧
  warp 到中心帧，mean/median/gated（Y 残差门限 τ=6/12）融合后过 stock l2，9 种方法。
- H3（`roi_tile/`）：1920×800 切 12×5=60 个 160px tile；ROI 面积 0/5/10/20/40%，
  原生 tile crf26/30/34；perframe（上帝选法）/fixed/center；真实 GOP 实测 tile 码率 +
  ffprobe 整流码率，与均匀阶梯（320/426/640/854 × crf22-42 + l2）前沿做等码率插值对比。
- 三路 VMAF/LPIPS 全部沿用 hard_survey 口径：TOS 7 档 × 32 帧、x4 后 bicubic 对齐 1920×800。

## 3. 文件清单

### h1_codecmeta/（数据通路，已入库）

| 文件 | 作用 | 复核定性 |
|---|---|---|
| extract_mv.py | PyAV 读 MV side data、40 字节 struct、建 mvx/mvy/cover 场 | 正确，docs 已记全部坑 |
| extract_qp.py | 单线程 -debug qp 解析逐块 QP | 正确，482 帧 0 错位 |
| combine_features.py | f275 帧 rgb+mv+qp 对齐 npz/三联图；审计 side data 类型 | 正确，残差不可得结论出处 |
| codec_features_f275.npz/.png、qp_check_*.png、mv_check_*.png | 对齐/箭头/热力核对证据 | 有效 |
| *.txt（3 个，已跟踪） | 早期 stdout 日志 | 历史记录，勿引用为结论 |

### h1_condnet/（条件网，代码与 3 个 eval json 已入库；weights/cache/deg/log/txt 不入库）

| 文件 | 作用 | 复核定性 |
|---|---|---|
| codec_lib.py | MV/QP 提取与 7 通道条件场（scan_mv:57、parse_qp:86、build_frame_cache:112） | 正确；QP 缺失静默补零（J6） |
| cond_model.py | CondIMDN_RTC(add:26)、CondIMDNCat(cat:54) | 正确；cat 零初始化等价性 probe_cat 验证 maxdiff=0 |
| build_cache.py | 6 退化各抽 K=180 帧存 cache npz（每视频独立 RNG） | 正确；deg 视频无生成脚本（J5） |
| train_cond.py | 三臂训练，翻转同步且 MV 分量变号（:90-93） | 正确；默认 lr2e-4/100ep 对微调过激（J1） |
| eval_cond.py | TOS 7 档 VMAF(打包)+LPIPS(逐帧)+配对符号检验 | VMAF 属 G1 打包口径（J3） |
| probe_cache.py / probe_cat.py | 缓存/零初始化自检 | 辅助 |
| h1_eval.json | warm/add 协议（weights/，540 样本、30ep） | 协议不健康，仅存档 |
| weights_scratch_eval.json | scratch/add（1080 样本、100ep） | 协议不健康，仅存档 |
| weights_cat_warm_eval.json | warm/cat（1080 样本、20ep，最干净对照） | **判决依据** |
| weights/、weights_cat_warm/、weights_scratch/（各 3 pth+trainlog） | 三协议权重 | 不入库 |
| cache/（6 npz，19~34MB）、deg/（6 mp4） | 训练缓存/退化视频 | 不入库，可再生成 |
| C（18.9KB） | 误重定向残片 | 垃圾，勿提交、可删 |

### h2_warp_oracle/（已入库 2 py + json；txt/log 不入库）

| 文件 | 作用 | 复核定性 |
|---|---|---|
| measure_warp_oracle.py | DIS 光流 warp、mean/median/gated 融合、9 方法×7 档评测 | 正确，邻帧为真实连续帧（K3） |
| smoke_warp.py | 验证 warp 后 Y 残差下降（配准本身有效） | 排除"光流全错"解释（K2） |
| warp_oracle.json | 9 方法×7 档逐帧 VMAF/LPIPS 与配对检验 | 2026-10-10 现算核实 |

### roi_tile/（已入库 2 py+gen_ladder.ps1+json+2 png；ladder mp4/ratetmp/log 不入库）

| 文件 | 作用 | 复核定性 |
|---|---|---|
| measure_roi_oracle.py | tile 真实码率测量、三选法、等码率前沿插值（:156/:255/:266） | 正确，口径对 ROI 偏慷慨（L2） |
| gen_ladder.ps1 | 640/854 均匀阶梯（bicubic+libx264 -g48） | 正确 |
| plot_roi_pareto.py | 帕累托图 | 与 json 一致 |
| roi_oracle.json / roi_pareto.png / smoke_strip.png | 216 配置结果与图 | 现算核实 |
| ratetmp/（2 mp4） | 冒烟临时产物 | 垃圾，可删 |

## 4. 复现方式

SR 解释器 `E:\dev\env\miniconda3\envs\SR\python.exe`，系统 ffmpeg 在 PATH；cwd 为各脚本目录。

- H1 数据通路核对：`python extract_mv.py`、`python extract_qp.py`、`python combine_features.py`
  （输入 hard_survey 的 lr_426_crf34.mp4）。
- H1 deg 视频（生成脚本缺失，2026-10-10 ffprobe 重建；源 data/video_mp4box/output_1080p.mp4，
  1920×1080@30fps，19039 帧；GOP 用 x264 默认值，未逐字节核实）：
  `ffmpeg -i output_1080p.mp4 -vf "scale=426:240:flags=bicubic" -an -c:v libx264 -crf 26 -pix_fmt yuv420p deg/bbb_426_crf26.mp4`
  （320 换 scale=320:180；crf 26/34/42 共 6 个）
- H1 缓存/训练/评测：`python build_cache.py` →
  `python train_cond.py --model cat --init warm --epochs 20 --lr 2e-5`（cat 协议实际口径，
  train_cat.log 核实；warm-add 为 30ep、scratch 为默认 100ep）→
  `python eval_cond.py --model cat --wdir weights_cat_warm`。
- H2：`python measure_warp_oracle.py`（约十几分钟，CPU 光流+GPU l2）；`python smoke_warp.py`。
- H3：先 `gen_ladder.ps1` 生成 640/854 阶梯（不入库），再 `python measure_roi_oracle.py`
  （全量约 1~2 小时，60 tile×3 crf 连续帧编码）；`SMOKE=1` 为 4 帧冒烟。

## 5. 结论与关键数字（2026-10-10 现算，VMAF 为 32 帧打包口径、LPIPS 逐帧）

**H1**：cat-warm 干净对照下 cond−base ΔVMAF 七档为 +0.02/+0.03/+0.02/−0.02/+0.04/+0.01/+0.01
（全部 |Δ|≤0.04），cond−shuf −0.07/−0.03/−0.02/−0.02/0.00/−0.01/+0.01（真条件≈乱序），
ΔLPIPS 为 +0.0007~+0.0015（cond 反而略差，部分 p<0.01）。另两套协议：warm-add cond−base
−0.12~−0.92（7/7 负），scratch-add −0.48~−3.12。**零带宽元数据条件在端侧小样本下无增益。**

**H2**：8 种多帧法在 7 档上 VMAF、LPIPS 全部差于 single_l2（绝大多数 p<0.001）。最保守的
gated3_t6 仍是 −1.60/−0.93/−0.60/−0.35/−0.90/−0.52/−0.30 VMAF、ΔLPIPS −0.0016~−0.0085；
朴素 mean7 最差达 −17.42。单调性：窗口越大越差、mean 比 median 差、门控越紧越接近单帧但
永不越过。**非学习时序借像素证伪。**

**H3**：6 底档×3 crf×3 选法×4 面积 = **216 个配置 ΔVMAF、ΔLPIPS 全部为负**。最接近的一点
（426crf26 底、perframe crf26、40% 面积）仍 −0.24；中重档 perframe 普遍 −10~−40，
fixed/center 再差 5~15。连每帧上帝选 tile 的上限都越不过均匀前沿。**全画幅 ROI tile 证伪。**

## 6. 复核记录

| # | 发现 | 严重度 | 位置 | 处置 |
|---|---|---|---|---|
| J1 | warm-add（lr2e-4/30ep）微调破坏 DIV2K 泛化：base anchor VMAF 88.92(stock)→79.76；scratch base 59.72 低于 bicubic 76.85。两套协议本身不健康，不能作判决依据 | 重要 | train_cond.py:153 默认 lr；h1_eval.json / weights_scratch_eval.json | 判决只采用 cat-warm（base 仅低 stock 2~3 分、三臂同构），docs 结论口径与此一致 |
| J2 | warm-add 训练样本 540（train.log），而现行 build_cache K=180×6=1080（cat/scratch 日志为 1080）；warm-add 所用早期缓存快照未存档 | 低 | train.log vs build_cache.py:24 | warm-add 非判决依据，仅记录 |
| J3 | H1 的 VMAF 走 32 非相邻帧 cv2 mp4v 打包（eval_cond.py:66-84），属档案 #9 G1 同族口径，绝对值虚高；run 内配对差方向可参考；**LPIPS 逐帧直算不受影响且同样为 null** | 重要（口径） | eval_cond.py:66-84 | 不单独重测；并入帧级指标 A 阶段，优先级低（LPIPS 已独立封死） |
| J4 | shuf 设计核实无误：乱序条件训练、评测喂真条件；cond≈shuf 说明条件编码器学会忽略条件，容量对照成立 | — | train_cond.py:101-103,199 | 维持 |
| J5 | deg/ 6 个 BBB 退化视频无生成脚本与来源记录 | 中（复现） | deg/ | §4 已按 ffprobe 重建命令 |
| J6 | QP 解析失败的帧静默补全零 QP（codec_lib.py:120-121），miss 只统计 MV 侧 | 低 | codec_lib.py:120 | 实测未出现缺失，记录 |
| J7 | 翻转增广 MV 分量变号正确（hflip 反 mvx 通道 1/4，vflip 反 mvy 通道 2/5） | —（核对通过） | train_cond.py:88-93 | 无需处理 |
| K1 | H2 VMAF 同 G1 打包口径；**LPIPS 逐帧直算，8 法×7 档一致为负，证伪不依赖 VMAF 口径** | 重要（口径） | measure_warp_oracle.py:125-141,144-149 | 结论稳健，不重测 |
| K2 | smoke_warp 证实 warp 后 Y 残差显著下降（配准有效），融合仍输——排除"光流失效"解释，坐实亚像素误差 x4 放大鬼影的机理 | —（增强证据） | smoke_warp.py | 维持 |
| K3 | 邻帧取自退化视频真实连续帧 t±1/t±3（read_windows 顺序解码），不是抽样帧；DIS 实例跨帧复用无状态问题 | —（核对通过） | measure_warp_oracle.py:66-86 | 无需处理 |
| K4 | 学习型轻量 VSR（可变形对齐+学习融合）未测；重、需多内容训练，属端侧前提外 | 边界 | docs/h2 §未否定边界 | 不复活，见 §8 |
| L1 | H3 两侧同走打包 VMAF，配对公平；LPIPS 逐帧且 216 点全负，证伪不依赖 VMAF 口径 | 重要（口径） | measure_roi_oracle.py:122-145 | 结论稳健 |
| L2 | ROI 码率口径对 ROI **偏慷慨**：贴回质量用 all-intra 编码（优于计费的 gop48 质量）、perframe 是上帝选法；B_low/60 面积均摊返还是近似（假设 tile 化基底）。慷慨上限仍大败 → 判决稳健 | —（增强证据） | measure_roi_oracle.py:186-198,255 | 维持 |
| L3 | 冒烟期曾因 4 帧简单样本+小 tile GOP≈全 I 出现 +4~15 假阳性，全量真实 GOP 后反转；教训已写进 docs/h3 | —（已记录） | docs/h3_roi_tile_oracle.md:54-56 | 维持 |
| L4 | docs/h3 结尾"稳定杠杆只剩 VBR（131 轨迹 100% 胜率）与视口"已**过时**：VBR 经档案 #7 E2 坐实为物理不可实现反事实（−0.06/−0.71），视口/360 用户已明确不做 | 中（陈旧引用） | docs/h3_roi_tile_oracle.md:60 | 本档案标注作废，见 §7 |
| L5 | docs/h3 正文 "crf20≈11.2k bits/帧" 为冒烟期数字，代码 NATIVE_CRFS=26/30/34 | 低（文档） | docs/h3:48 vs measure_roi_oracle.py:46 | 记录，不动 docs |
| L6 | ratetmp/ 2 个临时 mp4、h1 残片文件 C 为垃圾 | 低（卫生） | roi_tile/ratetmp/、h1_condnet/C | 可手动删，均不入库 |

## 7. 影响面：历史结论逐条重新定性

| 历史结论 | 复核定性 |
|---|---|
| H1"端侧便宜版证伪：ΔVMAF≤0.04、真条件≈乱序" | **成立**，数字逐字复现；但仅由 cat-warm 三臂同构对照支撑，含义是"此设置下元数据无可用增益"，不是"元数据方向整体死刑"（大规模多内容训练、残差通道、结构性 warp 注入未测） |
| warm-add / scratch 两套协议的负面数字 | 数字真实但协议不健康（微调反伤/从零欠拟合），只能作旁证，禁止单独引用为"H1 证伪证据" |
| H2"非学习时序融合 7 档全负" | **成立**，VMAF+LPIPS 双指标、符号检验一边倒、单调性完整；学习型 VSR 不在否定范围 |
| H3"216 配置等码率全负、上帝上限越不过均匀前沿" | **成立**，且口径对 ROI 偏慷慨，判决强度高；仅在视口/360/只显示局部场景可能复活（用户已排除 360） |
| H3 docs"后续杠杆：VBR 平方根 131 轨迹 100% 胜率" | **作废**：档案 #7 E1/E2 查明非等码率（+16~18% 字节）+ 决策后反事实，物理可实现口径为 −0.06/−0.71 |
| 三路绝对 VMAF 数值 | 带 G1 打包口径虚高（约 20~28 分量级，见档案 #9），跨档案比较只用相对差与 LPIPS |
| "客户端事后增强触顶、瓶颈在输入像素"的合流判断 | 作为**当时**的推理链成立；后续 #13（SR 感知码率分配）、#14（1080p→4K）、#15（引导旁路）继续在输入端/编码侧试探并各自封板，本批次不是终局结论 |

## 8. 是否重做：判断与方案

**三路均不原样重做，不复活为当前主线。**

- H1 复活条件（需同时满足，否则结论不变）：①多内容训练集（非单 BBB 1080 帧）；
  ②健康微调配方（lr≤2e-5、短 epoch，先断言 base≥stock−0.5 才允许评测 cond）；
  ③连续 rawvideo 口径；④条件参与结构性运算而非通道拼接。其中④的廉价形态就是 H2
  （已死），残差通道需改解码器——综合成本高、且与"端侧不大规模重训"前提冲突，
  属 PnP-VCVE 类文献工程，不建议在本项目做。
- H2 复活条件：仅"学习型轻量 VSR（可变形对齐）"一条缝，需要多内容训练 + 新增推理时延
  预算（与档案 #8 延迟侧结论冲突：轻量甜区已被专用小模型占满），不建议做；若未来重启，
  必须在连续 rawvideo 口径下先做 3~5 片段小样本门控，不过 +1 VMAF/LPIPS 同向即止损。
- H3 复活条件：显示形态改变（视口/缩放/360，用户已明确排除）；全画幅场景不复活。
- 测量债（J3/K1/L1 的连续口径重测）统一并入帧级指标体系 A 阶段，优先级排在 PAC vs IMDN
  质量表复核（#8 M1）与 hard_survey 7×7（#9 G1）之后；三路均有独立 LPIPS 证据，即使重测
  也不改变判决，仅作投稿前加固。

## 9. 踩坑铁律

- 评测微调类方案，先验证无条件 base 不劣于热启动 stock（容忍 ≤0.5）；微调把 stock 调崩后
  的 cond/base 对比没有意义（J1）。
- 容量对照（shuf）必须与处理臂同构、同初始化、同预算；本批次 cat 三臂同构是干净对照的范本。
- "省码率/送原生像素"类结论必须：真实 GOP 实测码率、与均匀加钱对手严格等码率比较
  （H3 冒烟假阳性、档案 #7 E1 是同族教训）。
- 时序融合的成败判据看 LPIPS 逐帧 + 符号检验，不要只看打包 VMAF。
- 解码器 side data 能力以实测为准：h264 只有 MOTION_VECTORS（+SEI），QP 走文本、残差没有；
  规划算法前先跑 combine_features 式的能力审计。

## 10. 上下游关系

- 上游：档案 #8 PAC-Net（加容量证伪，本批次动机）、档案 #9 hard_survey（提供 7 档退化
  DEG、measure_hard_survey 测量模块、REF_VIDEO 与 G1 口径）；h1_codecmeta 直接消费
  hard_survey/lr_426_crf34.mp4。
- 数据：TOS `E:\dev\Data\tears_of_steel_1080p.mp4`（评测留出）；BBB output_1080p（训练）。
- 下游：#13 SR 感知码率分配、#14 1080p→4K 探针、#15 高频引导旁路（均在输入端/编码侧继续
  试探，见各自档案）；"瓶颈在输入像素"的判断直接催生了 #15 旁路 oracle。
- 叙事文档：docs/h1_codec_metadata.md、docs/h1_condnet_falsification.md、
  docs/h2_warp_oracle.md、docs/h3_roi_tile_oracle.md（h3 结尾 VBR 引用按本档案 §7 L4 作废看待）。

## 11. 复盘备注（复活检查清单）

重启本批次任一路线前，逐项确认：

1. 是否有了多内容训练数据与健康微调配方？（没有→H1 不重启）
2. 条件是否参与结构性运算（warp/仿射调制/注意力偏置）而非拼接通道？warp 形态先看 H2 判决。
3. 测量是否已切连续 rawvideo 双管道、双臂对称？（否则数字不入结论）
4. 时序方案是否用真实连续帧、是否有逐帧 LPIPS 与配对检验？
5. ROI 类方案是否处于"只显示局部"的显示形态？全画幅直接引用 H3 216 点全负封板。
6. 码率结论是否在真实 GOP/ffprobe 整流口径下与均匀前沿等码率比较？
7. 引用本批次数字时是否区分了 cat-warm（判决依据）与另两套不健康协议？
