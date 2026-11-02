# 档案：服务端内容感知 VBR 编码（vbr_sims/vbr）

**状态：方向证伪。历史"平方根 VBR 全轨迹 QoE +1.70、131 条 100% 胜率"为假阳性——增益由平均多花 16.3%~18.2% 字节撑出，且"决策后应用提升量"把这些字节对下载/卡顿的影响完全隐藏，属物理不可实现的反事实。等码率口径的严格回答已由档案 #13（sr_aware_alloc LP oracle）给出：内容感知逐段码率分配上限 0.75~2.08 VMAF（上帝视角两趟编码）、可部署增益 ≤0.5，平方根方向甚至与最优分配反向。本线不原样重做。**
**时间：2026-08-30 ~ 08-31**
**最近复核：2026-10-09（22 个 py 全量读码或读 docstring 定性；10 个结果 json 与全部 intermediate 数据用 SR python 现算；720p 全片连续 VMAF 日志复核）**

## 1. 要回答的问题

BONES 与传统 ABR 都假设服务端提供 CBR（固定码率、逐段同质量）码流，客户端只做码率选择。本线问：**服务端按内容（运动强度/纹理）做 VBR——给复杂段多分配码率、简单段少分配——能否在不改动客户端的前提下提升播放质量？**

子问题：
1. 逐段 VMAF 实测下 VBR 相对 CBR 增益多少（五分辨率、多分配曲线）？
2. 三种与 BONES 仿真器的集成方式（改 manifest+enhance / 只改质量 / 决策后注入）哪种成立？
3. 全 131 条真实轨迹上增益是否稳定？
4. 分配曲线（二值/平方根/4 档/sigmoid/连续/阶梯）如何选优？

## 2. 方法概述

- **编码**：从真 4K60 BBB 母版按 4 秒段（GOP=120、fresh IDR、libx264 veryfast、bicubic 缩放、fps=30）独立编码五分辨率 CBR 与 VBR 阶梯；VBR 段码率 = 基础码率 × 内容因子。
- **测量**：每段失真与同段 10Mbps medium 重编码参考（同分辨率）跑 120 帧 clip libvmaf，得逐段 CBR/VBR VMAF 与文件大小，存 `data/bbb_vbr*/intermediate/{res}_data.json`。
- **策略演进**：二值（60% 高运动段 ×1.25，其余不变）→ 平方根（`base×(1+0.25√运动百分位)`）→ 4 档阶梯 [5%,10%,20%,30%] → sigmoid/线性/连续/阶梯扫描。
- **三种集成方式**：
  - 方式一（`vbr_integrate_to_bones.py`/`vbr_integrate_v2.py`）：同时替换 manifest 段大小与 enhance 质量，端到端重跑 BONES；
  - 方式二（`multitask_usm/generate_pure_vbr_enhance.py`，归本档案）：manifest 保持 CBR，只把提升量注入 enhance 的 base_quality；
  - 方式三（`post_decision_vbr.py`、`sqrt_vbr_full_traces.py`、`4stair_vbr_full_traces.py`、`full_vbr_comparison_fixed.py`）：BONES 用 CBR 数据做完全部决策，**事后只把逐段 VMAF 提升量加到质量序列上，段大小/下载/卡顿完全不变**，重算 QoE。历史正面结论全部出自方式三。

## 3. 文件清单

| 文件 | 作用 | 复核定性 |
|---|---|---|
| `vbr_concept_validation.py` | 最早的概念验证：直接用 BONES 现有码率阶梯表做等码率（±5%）逐段调换 | **E0：等码率粗测即 −6.64/−4.02，当时被忽略** |
| `vbr_small_experiment.py` | 小规模编码实验（含归一化、全片参考测量） | E6：参考未对齐且超时，有效结论≈0 |
| `vbr_encode.py` | 编码+测量函数库，`compute_vbr_bitrates` 有两次均值归一化+clip | **全目录唯一做等码率归一化的生成器，但未用于全轨迹结论**；measure_vmaf 有 E6 缺陷 |
| `vbr_batch_experiment.py`、`vbr_multi_resolution.py` | 高运动段比例扫描、多分辨率批量实验 | 同 E1 模板（独立编码、逐段测量、非等码率） |
| `vbr_integrate_to_bones.py`、`vbr_integrate_v2.py` | 方式一：生成 VBR manifest/enhance 并端到端仿真 | 端到端实测 −0.06（E2） |
| `cbr_vs_vbr_comparison.py` | 方式一/二端到端 QoE 对比 | 结果 `cbr_vs_vbr_results.json`、`cbr_vs_pure_vbr_results.json` |
| `generate_pure_vbr_enhance.py`（在 `../multitask_usm/`） | 方式二数据注入 | 端到端 −0.71（E2） |
| `generate_1080p_vbr.py`、`generate_sqrt_vbr_all_resolutions.py`、`generate_4stair_vbr_all_resolutions.py` | 二值 1080p 补测、平方根/4 档五分辨率数据生成 | **E1：无归一化，平均超支 16.3%~18.2%** |
| `continuous_bitrate_test.py`、`sigmoid_bitrate_test.py`、`staircase_bitrate_test.py` | 分配曲线扫描（均明示保持平均 +15%~16.5%） | E1，同模板 |
| `post_decision_vbr.py` | 方式三：二值策略 131 轨迹决策后注入 | **E2：物理不可实现反事实** |
| `sqrt_vbr_full_traces.py`、`4stair_vbr_full_traces.py` | 方式三：平方根/4 档 131 轨迹 | E2；与 fixed 版同策略数字不一致（E4） |
| `full_vbr_comparison_fixed.py` | 方式三 + 1080p 基准修正 + BONES/MD 双 QoE | E2、E4、E5（:16 import 已断裂） |
| `ablation_cmin_cmax.py`、`run_sim_baseline.py`、`run_sim_comparison_v4.py`、`run_stage_a_sim.py` | **CA-BONES 时期的 c_min/c_max 消融与基线仿真，错放于本目录** | 不属 VBR 线，属 8 月中内容感知 Lyapunov 线，总索引 #16 时认领 |
| 结果 json | `post_decision_vbr_full_traces.json`(36KB)、`sqrt_vbr_full_traces_results.json`(36KB)、`4stair_vbr_full_traces_results.json`(32KB)、`full_vbr_comparison_fixed.json`(174KB)、`cbr_vs_vbr_results.json`、`cbr_vs_pure_vbr_results.json`、`cmin_cmax_ablation.json`、`stage_a_sim_results.json`、`sim_comparison_v4.json` | 前 4 者为 E1/E2 污染产物，数字不可引用 |
| 全片连续测量日志 | `vbr_720p_cbr_vmaf.txt`、`vbr_720p_vbr_vmaf.txt`（各 19037 帧） | E3 复核证据：93.706 / 94.661 |
| 数据产物（未跟踪、不入库） | 原 `data/bbb_vbr_*`、`data/vbr_*`、`data/*_bitrate_test/` | 含 intermediate 与 mp4；**2026-10-27 本地已清空（约 23.5GB）**，结论以本 README 为准，需复查只能重跑编码脚本 |

## 4. 复现方式

```powershell
cd E:\dev\Code\Python\bones
$py = "E:\dev\env\miniconda3\envs\SR\python.exe"
# 等码率粗测（最快、最干净，直接读现有阶梯表）
& $py workspace/vbr_sims/vbr/vbr_concept_validation.py
# 方式三全轨迹（131 条，依赖 data/trace 与 data/bbb_vbr_*/intermediate；结论已作废，仅供复查）
& $py workspace/vbr_sims/vbr/full_vbr_comparison_fixed.py   # 注意 E5：当前 import 会失败
```
重新编码测量需先准备 `data/bbb/bbb_2160p_60fps.mp4`，单分辨率全量约 30~60 分钟；线程铁律见总索引（OMP=4、libvmaf n_threads=2）。

## 5. 结果记录（2026-10-09 全部现算）

**5.1 intermediate 实测：VBR/CBR 总字节比与逐段平均 ΔVMAF（159 段）**

| 策略 | 240p | 360p | 480p | 720p | 1080p | 实测超支 |
|---|---|---|---|---|---|---|
| 二值 | +1.374 | +1.187 | +1.151 | +0.955 | +0.830 | 1080p 现算 **+16.3%**（低分辨率 json 无 size 字段，构造值 +14.9%） |
| 平方根 | +2.356 | +2.233 | +1.277 | +0.998 | +0.848 | **+17.0%~+18.2%** |
| 4 档 | +2.317 | +2.233 | +1.240 | +0.972 | +0.802 | **+16.9%~+18.0%** |

二值超支是构造必然：60% 段 ×1.25、40% 段不变 → 平均 1+0.6×0.25 = 1.15。平方根/4 档生成器（`generate_sqrt_vbr_all_resolutions.py:70-78` 等）只做 clip、不做均值归一化，超支 17%~18%。

**5.2 方式三全轨迹（131 条，现算 json）**

| 口径 | 二值 | 平方根 | 4 档 |
|---|---|---|---|
| `full_vbr_comparison_fixed.json` BONES ΔQoE / 胜率 | +1.214 / 100% | **+1.704** / 100% | +1.603 / 100% |
| 同文件 ΔVMAF / MD-ΔQoE | +1.077 / +0.562 | +1.516 / +0.817 | +1.494 / +0.762 |
| 独立全轨迹脚本 ΔQoE / 胜率 | +1.214 / 100% | **+0.987** / 100% | **+0.836** / 100% |

**5.3 方式一/二端到端（docs/server_side_vbr_coding.md 记录，二值）**：方式一 QoE −0.06、卡顿 +71ms；方式二 QoE −0.71、卡顿 +693ms。

**5.4 等码率粗测（5.2 中 vbr_concept_validation.py 现算，480p 档）**：码率差 <5% 约束下，按运动调换档 1/档 3，VMAF **−6.64**；按纹理调换 **−4.02**。阶梯档跳变粗、仅作方向性证据；严格版本见档案 #13。

**5.5 720p 全片 concat 连续测量**：CBR 93.706、VBR 94.661（+0.955），与逐段 clip 平均（93.71/94.66）完全一致，19037 帧。

## 6. 复核发现

### E0（被忽略的早期反证，重要）
`vbr_concept_validation.py:75-128` 在开工第一天就用现有阶梯表做了等码率（±5%）内容感知调换，结果 −6.64/−4.02。后续工作线没有采用等码率约束，转而使用超支 16%+ 的生成器，方向被假阳性带跑。

### E1（致命：非等码率，"VBR 增益"主要是多花字节买来的）
- 二值/平方根/4 档三套实际用于全轨迹结论的生成器**均无均值归一化**，现算总字节超支 +16.3%~+18.2%（5.1 表）。
- 全目录唯一做了归一化（两次归一+clip）的 `vbr_encode.py:59,65 compute_vbr_bitrates` 没有被任何全轨迹结论采用。
- 增益量级与超支自洽：多花 17% 码率买 +0.8~+2.4 分段 VMAF，落在码率-质量曲线的正常斜率内，不构成"内容感知分配"的独立贡献。
- 严格等码率对照（档案 #13，TOS 183 段×6 CRF、BBB 158 段，LP oracle 总预算严格相等）：通用逐段分配上帝上限仅 0.75~2.08 VMAF、SR 感知新颖性 ≤0.41；平方根启发式在等码率下 5 个预算中 3~4 个输给等比特 CBR，复杂度权重与真实边际 VMAF/bit 排序 Spearman = −0.85~−0.93（**方向反了**：VMAF 时域掩蔽对高运动内容更宽容，复杂段每比特边际回报最低）。

### E2（致命：方式三是物理不可实现的反事实）
`post_decision_vbr.py:73-156`、`full_vbr_comparison_fixed.py:71-92` 让 BONES 用 CBR 段大小完成下载与增强决策，决策后只把逐段 VMAF 提升量加到质量序列，注释明示"卡顿和带宽不变"（post_decision_vbr.py:122-130）。VBR 实际多花的 16%~18% 字节**从不被下载**，不产生下载时间、不挤占缓冲、不产生卡顿——真实系统中这些字节必须传输，必然改变 ABR 决策与卡顿。
- 同一数据走物理可实现的方式一/二（段大小或质量真实进入仿真），端到端为 −0.06 / −0.71（5.3）。
- 即"100% 胜率、0 条下降"只在"只计收益、不计成本"的反事实里成立。

### E3（测量口径，经复核对本线结论无显著影响；并纠正一处历史归因）
- 所有生成器逐段独立编码（每段 fresh IDR、GOP=120、veryfast），逐段对同段 10Mbps medium 有损重编码参考跑 120 帧 clip libvmaf：时域 VMAF 在每个 4 秒边界重置、参考非母版。
- **但 720p 全片 concat 连续测量（19037 帧）与逐段均值完全一致（93.706/94.661 vs 93.71/94.66，5.5），证明该口径在本线没有造成显著虚高，Δ 在两臂间基本抵消。**
- **纠正归因**：`docs/sr_aware_alloc_falsification.md` 判决第 3 条把"抽样帧打包 VMAF 虚高"列为本线假阳性成因之一。经全目录 grep（`cv2/VideoWriter/select=/sample` 零命中）与日志复核，非相邻抽样帧打包成连续 clip 的事故（绝对 VMAF 虚高 20~28 分，`docs/known_issues.md:80-83`）发生在 hard_survey/H1/H2/H3/enhancer_survey 线，**不发生在 VBR 线**。本线假阳性成因只有 E1+E2。该文档措辞留待档案 #13 审计时修正。

### E4（重要：同一策略两套脚本数字不一致）
"平方根"在 `full_vbr_comparison_fixed.json` 为 +1.704，在 `sqrt_vbr_full_traces_results.json` 为 +0.987；4 档为 +1.603 vs +0.836（5.2）。差异来自 1080p 基准修正的应用范围不同（fixed 版同时改写 CBR 对照侧基准，影响 oscillation 与基线）。同一命名策略两套结果差 0.7~0.8 QoE，说明历史数字对计分口径敏感、不可复现性强。

### E5（工程）
- `full_vbr_comparison_fixed.py:16 from md_qoe import ...` 在 dca6b85 目录重组后已断裂（md_qoe 现为同级 `../md_qoe/` 包），脚本当前无法直接运行（与档案 #6 C7 同一断裂）。
- `full_vbr_comparison_fixed.py:59-60` 逐段卡顿=总卡顿/段数均摊（同档案 #6 C2）；因 CBR/VBR 两侧卡顿相同、在差值中抵消，不影响 E2 判决，但 MD-QoE 绝对值继承档案 #6 全部问题。

### E6（工程，未污染结论）
`vbr_encode.py:160-167 measure_vmaf` 失真用单段、参考用整片且无 `-ss` 对齐；1080p 下该命令 120s 超时全部失败（`vbr_test_output.txt` 可见），未产出有效数据。早期小实验（`vbr_experiment_output{,_v2,_v3}.txt`，低超支口径）VBR 为 90.89/91.17/91.65 vs CBR 91.38，增益在 −0.5~+0.3 之间，与 E0/E1 的等码率结论方向一致。

### E7（口径，2026-10-10 M0 复核补记）
本目录 `sqrt_vbr_full_traces.py:112-118`、`4stair_vbr_full_traces.py:98-101`、`post_decision_vbr.py:124-130` 自带的 QoE 汇总把毫秒 `freeze_time` 直接 /段数/10（同档案 #6 C1b，1 秒卡顿罚 100），仿真器四处 report 已于 M0 修为秒口径（见 `docs/frame_scheduling_fix_plan.md`）。本线已封板、脚本保持原样不改；其 QoE 绝对值为修正前口径，CBR/VBR 两臂同口径差值比较不受影响。

## 7. 影响面：哪些历史结论作废

| 历史结论（docs/server_side_vbr_coding.md） | 复核判决 |
|---|---|
| 131 轨迹 100% 胜率、QoE +1.21（二值）~+1.70（平方根）、VMAF +1.08~+1.52、卡顿 0 变化 | **作废**：E1 超支 16%~18% + E2 反事实，方式三不对应任何可实现系统 |
| "平方根分配最优、连续分配优于二值"的工程发现 | **作废**：等码率下平方根与最优分配反向（#13 Spearman −0.85~−0.93），3~4/5 预算输 CBR |
| 方式一 −0.06、方式二 −0.71 | **有效保留**：这是仅有的物理可实现端到端结果，本身即否定证据 |
| MD-QoE 同样验证（+0.56/+0.82/+0.76） | 作废：继承 E1/E2 污染 + 档案 #6 C1~C3 指标缺陷 |
| "VBR 对 ABR 透明、决策后应用提升"是正确集成方式 | **作废为系统结论**：该手法只能作为离线 oracle 上界探针，不能作为集成方案 |
| 增量传输+渐进增强 3G QoE +0.97（该文档"已放弃方向"处提及） | **不属本档案**：代码在 `../lookahead/run_three_way_comparison.py:5,87-96`、`run_four_way_comparison.py:5-6`，归档案 #3 |
| 720p 逐段/全片 VMAF 93.71/94.66 等绝对测量值 | 保留可引用（E3 已核口径一致），但对应的是 +16% 超支口径，勿当作等码率增益 |

## 8. 是否重做 / 重新设计

**不原样重做。** 本线的科学问题——"等总码率下，服务端按内容逐段重排码率能否提升端到端质量"——已在档案 #13 用更严格的设计（TOS+BBB 双内容、6 CRF 阶梯、严格总预算相等、贪心注水与独立 LP 双求解器复核）回答完毕：
- SR 感知重排新颖性 0.00~0.41 VMAF；
- 通用内容感知重排的上帝上界 0.75~2.08（且需两趟编码、不可部署）；
- 可部署的单遍启发式（平方根）方向性错误。

服务端 VBR 作为论文工作点**封板**。若未来工程上复活，唯一合理形态是"服务端预编码多版本 + 客户端按自身缓冲/算力透明选择"，但它既不需要新算法、上限也只有 1~2 VMAF，不建议投入。

## 9. 解决方案与后续规划

1. **结论迁移**：本线与档案 #13 共同构成"服务端/传输侧逐段码率重排"的完整封板证据；帧级粒度的比特重排价值（档案 #2：紧预算 oracle +0.8~+3.4、增益与纹理 r=0.908）仍开放，但作用点在**客户端增强预算分配**，不在服务端编码。
2. **指标教训并入帧级指标体系**：任何"重分配/再分配"类对照必须先验总字节/总 GPU 预算相等（相对误差 <0.5%），收益与成本必须在同一次仿真中同时生效；"决策后注入"只允许作为显式标注的 oracle 上界，禁止作为方案结论。
3. **文档处理**：`docs/server_side_vbr_coding.md` 为历史设计文档，随本档案提交但在文首加作废标注（指向本 README 与档案 #13），不删除（保留踩坑全过程）。
4. **数据处理**：`data/bbb_vbr*/` 六个大目录与 mp4 不入库；intermediate json 已随 20cd0ea 之后的策略按需保留，引用数字以本 README 第 5 节现算值为准。

## 10. 踩坑铁律（本线新增）

- **等码率是码率分配类工作的生死线**：生成器必须做均值归一化并现算总字节比；"平均码率提升 +16%"写在文档里却没有被当作成本，是本线最根本的失误。
- **"决策后应用"= 反事实，不是集成方式**：只改质量序列、不改段大小与下载过程的对比，衡量的是神谕，不是系统；端到端方式一/二的 −0.06/−0.71 才是真实答案。
- **同一策略多套脚本必须对表**：平方根 +1.70 vs +0.99 的分歧说明计分口径（1080p 基准、oscillation 处理）足以左右结论，归档前必须逐脚本对齐。
- **开工第一天的等码率粗测要被正视**：−6.64/−4.02 的信号与三个月后 LP oracle 的结论一致；反直觉的正面结果出现时，先回头核对最早的粗测与预算约束。
- 非相邻帧打包虚高是 hard_survey 等线的事故，勿张冠李戴到本线（E3）。

## 11. 上下游与复活清单

- **上游**：BONES 仿真器 `control/bones.py`、`simulation/simulator.py`；内容特征 `data/bbb/content_features_bbb.json`（motion_intensity/texture_complexity）；母版 `data/bbb/bbb_2160p_60fps.mp4`。
- **同批/下游**：
  - 档案 #6（md_qoe）：`full_vbr_comparison_fixed.py` 的 MD-QoE 数字继承其指标缺陷，import 断裂同因；
  - 档案 #13（sr_aware_alloc）：本线科学问题的严格等码率判决，`docs/sr_aware_alloc_falsification.md` 判决第 3 条对本线测量口径的归因需按 E3 修正；
  - 档案 #3（lookahead）：增量传输+渐进增强仿真（+0.97）在该目录，勿在本线查找；
  - CA-BONES 线：`ablation_cmin_cmax.py`、`run_sim_*.py` 为错放文件，总索引 #16 认领。
- **复活检查清单**（若日后重启服务端编码方向，逐项确认）：
  1. 生成器输出总字节与 CBR 严格相等（现算比值打印并断言 ±0.5%）；
  2. 段大小真实进入 manifest 并参与下载仿真，禁止决策后注入口径出结论；
  3. 连续时间序全片 libvmaf（本线 720p 已验证逐段口径在 Δ 上可用，但新内容沿用全片口径更稳）；
  4. 至少 TOS+BBB 双内容、多轨迹；
  5. 先跑 `vbr_concept_validation.py` 式零编码粗测与档案 #13 LP 上界，上界 <1 VMAF 即停止。
