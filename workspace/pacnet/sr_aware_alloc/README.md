# 档案：SR 感知率失真码率分配（sr_aware_alloc）

- 状态：已封板（NO-GO，TOS+BBB 跨内容；2026-10-10 现算复核）
- 时间：实验 2026-09-20；复核 2026-10-10
- 最近复核：2026-10-10（读码 + verify_oracle 双内容重跑 + alloc_results 现算，数字逐值复现）

叙事文档：`docs/sr_aware_alloc_falsification.md`（本文件只做档案化，不重复表格）。

## 1. 要回答的问题

把 l2 超分的"按内容恢复能力"写进服务端逐段码率分配的率失真目标（难恢复段多给码率、
SR 补得回的段少给），在严格等总码率下能否相对等比特 CBR / 平方根 VBR / 经典 R-D oracle
带来稳定 ≥1 post-SR VMAF 增益？

## 2. 方法概述

- TOS（426×178@24，96 帧段，183 整段）与 BBB（426×240@30，120 帧段，158 整段），
  libx264 CRF 22/26/30/34/38/42 强制 GOP、scenecut=0、段间独立（gen_crf_ladder*.ps1）。
- 逐段 R-D：ffprobe 逐帧 pkt_size 聚合段比特；bicubic 与 l2 两画面纵向堆叠经 rawvideo
  管道送**单次 ffmpeg 双 libvmaf 连续时间序打分**（measure_seg_rd.py:69-87），帧数断言。
- 五策略严格等预算（花费比全部 =1.000）：恒定 CRF、等比特 CBR、平方根权重、
  原始 R-D oracle（凹包络+全局边际斜率注水）、SR 感知 oracle；所有策略统一用 post-SR VMAF 评估。
- 独立 LP 复核（verify_oracle_lp2.py，HiGHS 累计变量、原始 6 点、允许非凹曲线）。

## 3. 文件清单

| 文件 | 作用 | 复核定性 |
|---|---|---|
| gen_crf_ladder.ps1 / bbb/gen_crf_ladder_bbb.ps1 | 426 宽六档 CRF 阶梯（强制 GOP） | 正确；ladder mp4 不入库，可再生成 |
| measure_seg_rd.py / bbb/measure_seg_rd_bbb.py | 逐段比特+连续双 libvmaf R-D 测量 | **金标准管道，帧级指标 A 阶段范本** |
| alloc_compare.py | 五策略等预算对比、凹包络注水、出图 | 正确；--rd/--tag 参数化双内容复用 |
| verify_oracle.py | LP 复核 + Spearman 机制分析 | 正确 |
| verify_oracle_lp2.py | 独立 LP（HiGHS，非凹允许） | 正确，与贪心结论一致 |
| seg_rd.json / bbb/seg_rd_bbb.json | 六档逐段 {bits,bic_vmaf,sr_vmaf} | 183/158 段齐全，现算核实 |
| alloc_results*.json、alloc_pareto*.png | 五策略结果与帕累托图 | 现算逐值复现 |
| ladder/*.mp4（12 个，3~33MB）、*.log、tmp/ | 阶梯/日志/临时 vmaf json | 不入库 |

## 4. 复现方式

SR 解释器 `E:\dev\env\miniconda3\envs\SR\python.exe`，cwd 为本目录；OMP/MKL=4。

1. `gen_crf_ladder.ps1`（BBB 在 bbb/ 下跑对应 ps1）生成 ladder。
2. `python measure_seg_rd.py`（BBB：`python bbb/measure_seg_rd_bbb.py`，全片约 1~2 小时/内容）。
3. `python alloc_compare.py`；`python alloc_compare.py --rd bbb/seg_rd_bbb.json --tag bbb`。
4. `python verify_oracle.py`；BBB 加同样 `--rd/--tag`。

## 5. 结论与关键数字（2026-10-10 重跑复现）

- **SR 新颖性（SR oracle 在 SR 目标最优解上的 post-SR 分 − 在原始 RD 最优解上的 post-SR 分，
  LP 口径）**：TOS 0.00/0.20/0.25/0.23/0.12，BBB 0.00/0.07/0.09/0.32/0.41（CRF42→26），
  两内容 10 个预算点全部 ≤0.41，远低于 1 VMAF 阈值。
- **SR oracle 对等比特 CBR 上限**：TOS −0.44/+1.09/+1.07/+0.91/+0.74，
  BBB +0.00/+2.08/+1.97/+1.38/+0.96——两趟编码上帝上界仅约 1~2，可部署预测器只能吃到一部分。
- **平方根等码率下不成立**：TOS 4/5 输 CBR（−0.13~−1.28），BBB 3/5 输、2/5 持平。
- **机制闭合**：工作点逐段边际 VMAF/bit 排序，bic 目标 vs post-SR 目标
  Spearman TOS 0.987（p≈1e-145）/ BBB 0.949（p≈7e-80）；平方根复杂度权重 vs 真实 SR 边际
  斜率 TOS −0.93 / BBB −0.85（按复杂度加码率方向做反）。
- 所有策略预算花费比 1.000（CRF42 端点 1.004，插值舍入）；贪心凹包络相对 LP 真值微高估
  ≤0.15，方向不变。

## 6. 复核记录

| # | 发现 | 严重度 | 位置 | 处置 |
|---|---|---|---|---|
| M1 | docs 判决第 3 条把历史平方根 VBR 虚高部分归因于"抽样帧打包 VMAF 虚高"，属**过度归因**：VBR 线从不做非相邻帧打包（档案 #7 E3 grep 0 处 cv2/VideoWriter，720p 全片 19037 帧连续 libvmaf 与逐段均值一致）；打包事故代码在 hard_survey/measure_hard_survey.py（档案 #9 G1）。历史 VBR 假象的真实原因是 E1 非等码率（+16~18% 字节）与 E2 决策后反事实 | 重要（归因纠错） | docs/sr_aware_alloc_falsification.md 判决 3 | 2026-10-10 已改写该条，明确两因素与本目录无关 |
| M2 | 本目录测量管道是全工作区**最干净的 VMAF 口径**：连续时间序、双臂同一次 ffmpeg、rawvideo 无重编码、帧数断言、ffprobe 真实逐帧比特 | 正面资产 | measure_seg_rd.py:69-145 | 列为帧级指标体系 A 阶段范本；CRF30 bic 39.5/sr 50.7 即 known_issues 连续口径交叉数 |
| M3 | LP 独立复核建模正确（累计变量有序填充、允许非凹、HiGHS），与贪心两实现独立且结论一致；新颖性/上限不依赖求解器 | —（核对通过） | verify_oracle_lp2.py | 维持 |
| M4 | 贪心注水沿凹包络弦线性计收益、再在原始曲线重评估，微高估 ≤0.15；docs 已披露并以 LP 为准 | 低 | alloc_compare.py:82-103 | 引用统一用 LP 口径 |
| M5 | 边界：仅 426 宽、仅 VMAF、两内容、6 个离散 CRF、oracle 为连续插值上界；320/640 宽与 LPIPS 未测 | 边界 | docs 边界节 | 机制（Spearman 0.95/0.99）闭合，反转可能性低 |
| M6 | doc 表 CRF42"−0.45"为贪心四舍五入，LP 真值 −0.444 | 极低 | docs 结果表 | 以 LP 列为准，不改表 |

## 7. 影响面：历史结论逐条重新定性

| 历史结论 | 复核定性 |
|---|---|
| "SR 感知码率分配无新颖性（≤0.41），NO-GO" | **成立**，双内容、双独立求解器、机制 Spearman 闭合 |
| "oracle 对 CBR 上限约 1~2 VMAF" | **成立**（BBB 低预算端最高 2.08）；这是两趟编码上帝上界，不是可部署增益 |
| "平方根内容自适应方向做反" | **成立**，两内容 Spearman −0.93/−0.85；与档案 #7 的等码率粗测（E0 −6.64/−4.02）相互印证 |
| 历史"平方根 VBR +1.70 QoE/100% 胜率" | **作废**（档案 #7 E1/E2）；本目录判决第 3 条归因已纠正（M1） |
| 经典逐段 VBR（非 SR 感知）本身 | 未被否证，上限 +1~2 随内容异质性浮动；属编码工程，不是神经增强论文点 |

## 8. 是否重做：判断与方案

**封板，不重做。** 复活/加固仅在以下情形，且优先级低：

1. 未来论文需要编码侧基线数字：补 320/640 宽与 LPIPS（同管道，约半天/内容）；
   预期不反转——边际排序 Spearman 0.95/0.99 已给出机制级证据。
2. 若出现时域 SR 模型（增益随段内容分布不同于单帧 l2），需重跑本探针：SR 新颖性的前提
   "增强近似对各段 R-D 做均匀仿射缩放"可能被打破；这是唯一有理论依据的复活点。
3. 任何"内容自适应码率分配"新想法，必须先过本目录三关：严格等预算（±0.5% 断言）、
   连续口径测量、与 CBR 而非恒定 CRF 对比。

## 9. 踩坑铁律

- 等码率对比必须归一化总字节并断言花费比（本目录全部 1.000；档案 #7 E1 是反例）。
- "决策后改质量、不下载对应字节"只能当显式标注的 oracle，禁止作方案结论（档案 #7 E2）。
- 率失真最优结论以独立 LP/全局最优为准，贪心凹包络只作可视化与交叉验证。
- 测量管道用 rawvideo 堆叠双 libvmaf，禁用 cv2 mp4v 临时重编码（档案 #9 G1）。

## 10. 上下游关系

- 上游：档案 #7 服务端 VBR（本探针是对"内容自适应码率分配"的严格等码率再审，
  并纠正其测量归因）；档案 #9 hard_survey（G1 打包口径的反面对照与连续口径来源）。
- 测量范本被帧级指标体系 A 阶段复用（PAC vs IMDN 质量表连续重测、hard_survey 7×7 重测）。
- 结论"编码侧无 SR 感知杠杆"与 #14（1080p→4K 后处理 NO-GO）、#15（旁路 NO-GO）共同
  关闭了"编码/分配侧"方向簇。

## 11. 复盘备注（复活检查清单）

1. 新想法是否声称"按内容/按 SR 增益分配码率"？是→先在本目录框架内跑 oracle 上界，
   新颖性 ≤0.5 直接止损。
2. 对比是否严格等预算、连续口径、以 CBR 为对手？
3. 增强模型是否换过（尤其时域模型）？换过才需要重测 seg_rd。
4. 引用历史 VBR 数字时是否区分了物理可实现口径与反事实口径？
5. 复用测量管道时是否保留帧数断言与双臂同次 ffmpeg？
