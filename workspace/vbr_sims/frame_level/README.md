# 档案：逐帧质量张量与帧级 DP 调度（frame_level）

**状态：测量张量有效但口径有污染（保留待重测）；帧级 DP 调度结论降级（仿真器存在记账 bug 与不等预算对照），从未接入真实播放器**
**时间：2026-08-29**
**最近复核：2026-10-09（6 个脚本 + `control/bones_frame_level.py` 全量读码，结果 json 现算，DP 离散化 bug 已定量复核）**

## 1. 要回答的问题

BONES 的增强决策以**段（4 秒/120 帧）为最小粒度**：一个段内所有帧施加同一个增强档。但同一段内帧与帧的内容差异很大（纹理、运动、场景切换），SR 增益在帧间高度不均。两个问题：

1. 逐帧 VMAF 增益的差异有多大、是否与内容特征相关（帧级内容感知假设是否成立）；
2. 在固定 GPU 增强预算下，把增强算力在段内逐帧分配（简单帧跳过/给低档、复杂帧给高档）能否比"整段同一档"拿到更高平均 VMAF；
3. 把帧级分配嵌进 BONES 联合仿真器（段级 BONES 定码率与最高增强档，段内 DP 分配），端到端 QoE 是否优于纯 BONES。

## 2. 方法概述

三条链路：

- **离线测量**：BBB 1080p ladder，对 240/360/480 三档分辨率 × 5 种处理（base  bicubic、SR low/medium/high、USM-only）× 158 个完整段 × 120 连续帧，逐帧跑 libvmaf，产出张量 `data/bbb/per_frame_quality.json`，同时逐帧抽取内容特征（帧差运动、Laplacian 纹理、场景切换）。
- **离线段内调度**：在张量上对比三种段内分配策略——段级固定（基线）、按增益/ms 的贪心、帧级背包 DP（GPU 预算约束下最大化段内 VMAF），预算 2000/4000/6000 ms。
- **联合仿真**：`control/bones_frame_level.py` 继承 BONES 仿真器，段下载完成后在 BONES 给定的最高增强档内跑帧级 DP，用 DP 实际 GPU 时间推进增强缓冲；4 条网络轨迹（3g_low/4g_medium/wifi_high/volatile）对比纯 BONES。

## 3. 文件清单

脚本目录 `workspace/vbr_sims/frame_level/`：

| 文件 | 行数 | 作用 | 复核定性 |
|---|---|---|---|
| `measure_per_frame_vmaf.py` | 389 | 逐帧 VMAF 张量 + 内容特征生成器，支持断点续跑 | 数据资产有效；测量口径有污染，见 §6 B7 |
| `eval_frame_level_scheduling.py` | 346 | 离线固定/贪心/帧级 DP 对比（延迟表 `imdn_bbb_3060ti.json`） | B1 已于 2026-10-10 修复（0.1ms 向量化，见 §5.2/§11）；贪心实现有缺陷 B8 |
| `run_bones_vs_framelevel.py` | 148 | 联合仿真器 4 轨迹对比启动器 | 结论受 B1~B6 多重影响；输出路径硬编码旧 workspace 根（:141） |
| `analyze_per_frame_data.py` | 159 | 张量统计：段内波动、增益与内容相关性、边际收益 | 分析只读，数字可信（口径同 B7，方向/相关性可信） |
| `analyze_segments_detail.py` | 132 | **跨线脚本**：对比 BONES imdn_always 与 multitask 播放器输出 json（232235 vs 142540） | 结论归档案 1（multitask_usm），本目录仅登记 |
| `test_perframe_vmaf.py` | 57 | libvmaf 逐帧 json 输出可行性冒烟测试 | 一次性验证脚本；同样走 cv2 mp4v 临时文件 |
| `frame_level_scheduling_results.json` | — | 离线调度结果（3 分辨率×3 预算） | 已被 2026-10-10 0.1ms 重跑覆盖，收敛值见 §5.2；结果文件不入库 |
| `bones_vs_framelevel_results.json` | — | 联合仿真 4 轨迹结果 | 原数字，结论降级见 §7 |

范围外核心依赖：

- `control/bones_frame_level.py`（347 行，仓库 control 目录，不在 workspace）：联合仿真器主体，B2~B6 均在该文件；
- `data/bbb/per_frame_quality.json`（张量本体，2370 条×120 帧，data 目录）；
- `data/bbb/imdn_bbb_3060ti.json`：延迟/FPS 表（3060Ti 实测，现机为 5080，未更新）。

## 4. 复现方式

```powershell
cd E:\dev\Code\Python\bones
# 1) 张量测量（耗时数小时，可断点续跑，输出 data/bbb/per_frame_quality.json）
E:\dev\env\miniconda3\envs\SR\python.exe workspace\vbr_sims\frame_level\measure_per_frame_vmaf.py
# 2) 离线调度对比（输出在脚本同目录 frame_level_scheduling_results.json）
E:\dev\env\miniconda3\envs\SR\python.exe workspace\vbr_sims\frame_level\eval_frame_level_scheduling.py
# 3) 联合仿真（注意：输出路径硬编码 workspace 根，重跑前先改 run_bones_vs_framelevel.py:141）
E:\dev\env\miniconda3\envs\SR\python.exe workspace\vbr_sims\frame_level\run_bones_vs_framelevel.py
# 4) 张量分析
E:\dev\env\miniconda3\envs\SR\python.exe workspace\vbr_sims\frame_level\analyze_per_frame_data.py
```

模型权重：`player/.../enhancer/` 下 `IMDN_low_bbb_x{2,3,4}.pth`、`IMDN_medium_*`、`IMDN_high_*`；low=IMDN_RTC(num_modules=3,nf=6)，medium=IMDN_RTC 默认，high=IMDN(nf=32)（measure:49-65）。

## 5. 结论与关键数字

### 5.1 张量事实（2026-08-29 测量，2026-10-09 现算；mp4v 口径，方向可信、绝对值有偏）

全片平均 VMAF（158 段×120 帧）：

| 分辨率 | base | sr_low | sr_medium | sr_high | usm_only |
|---|---|---|---|---|---|
| 240p | 41.01 | 52.14 | 56.53 | 61.85 | 53.23 |
| 360p | 65.02 | 72.62 | 76.07 | 79.91 | 69.94 |
| 480p | 77.16 | 83.20 | 85.86 | 87.30 | 78.95 |

360p 各方法相对 base 的逐帧增益：low +7.60（负增益帧 2.3%）、medium +11.04（std 12.93，负增益 1.1%）、high +14.89（负增益 0.2%）、**usm_only +4.92 但负增益帧 29.7%**。

帧级内容感知证据（360p sr_medium，18960 帧）：

- 增益与**纹理复杂度相关 r=0.908**，与运动强度相关 r=0.516；
- 高运动帧均值 +20.56 vs 低运动帧 +6.92（差 13.64）；
- 场景切换帧（201 帧）仅 +4.70 vs 非切换 +11.11；
- 帧级 VMAF std 8.29，段均值序列 std 7.21（帧级波动为段间的 1.1 倍——段平均抹掉了约一半波动）。

### 5.2 离线段内调度（oracle 上界，158 段均值）

DP 相对"整段固定同档"的平均 VMAF 增益，三种网格口径对照：

| 分辨率/预算 | 原 step=10ms floor | step=1ms（2026-10-09 复核） | **step=0.1ms 收敛（2026-10-10 M1）** |
|---|---|---|---|
| 240p / 2000ms | +4.13 | +3.14（该锚点未复现，见下） | **+2.83** |
| 360p / 2000ms | +1.64 | +0.81 | **+0.69** |
| 360p / 4000ms | +3.85 | +3.41 | **+3.41** |
| 480p / 2000ms | +2.04 | +1.49 | **+1.40** |
| 480p / 4000ms | +0.53 | +0.42 | **+0.40** |

2026-10-10 M1 收敛性核查（240p/2000ms，同数据逐级细化）：step=10/1/0.5/0.2/0.1ms 增益 = +3.90/+2.74/+2.84/+2.83/+2.83，0.1ms 已收敛；step=1ms 本身偏离收敛值 0.09，历史 +3.14 锚点未在当前代码复现，以收敛值 +2.83 为准。统一实现见 `simulation/gpu_grid.py`（0.1ms、四舍五入、向量化 DP、最小正成本断言）。

预算宽裕（6000ms）时增益≈0（整段已可上高档，无分配空间）；贪心在所有格子都劣于固定（如 240p 54.97 vs 固定 61.85），无参考价值（B8）。

### 5.3 联合仿真（3060Ti 延迟表，4 轨迹）

| 轨迹 | ΔVMAF | ΔQoE | 增强帧比例 |
|---|---|---|---|
| 3g_low | +0.77 | **+0.83** | 0.85 |
| 4g_medium | +0.04 | −0.29 | 0.22 |
| wifi_high | +0.04 | +0.05 | 0.05 |
| volatile | −0.10 | −0.61 | 0.37 |

仅紧预算 3G 为正；高带宽几乎不触发（720p/1080p 退化为整段固定，B5）。

## 6. 复核记录（2026-10-09）

| # | 发现 | 严重度 | 位置 |
|---|---|---|---|
| B1 | **DP 时间按 10ms 向下取整，最便宜的 low 档变成零成本动作**：240p low 7.50ms、360p low 8.79ms/帧 → `int(ms/10)=0`，DP 可给每帧"免费"增强；其余档位少计 5%~27%（480 low 12.28→10、360 med 13.77→10、240 high 24→20）。`dp_enh_ratio` 高达 0.93~0.99 即源于此；`gpu_used` 同步低估，增强缓冲推进更快，卡顿也偏乐观。step=1ms 复核后离线增益缩水 0.1~1.0 分（§5.2），未归零 | 致命 | `eval_frame_level_scheduling.py:176,185`；`control/bones_frame_level.py:52,141` |
| B2 | **段间平滑惩罚方向反了且不对称**：`penalty = w*(raw−prev)/120`，只压"高于上一段"的 VMAF，低于上一段时惩罚为负、反而加分；应为 `w*abs(raw−prev)/120`。该惩罚直接进入 DP 目标函数，会系统性扭曲选档（鼓励质量下行） | 重要 | `control/bones_frame_level.py:147` |
| B3 | **段内逐帧混排的闪烁完全不进 QoE**：report 的 oscillation 只累计段均值之间的 abs 差，帧与帧之间 0/2/3 档混排（3g 轨迹 15% 帧 base、46% medium、35% high）的逐帧质量抖动被忽略——这恰是帧级调度引入的最大新代价 | 重要 | `control/bones_frame_level.py:303-304` |
| B4 | **对照不等预算**：帧级方案的 GPU 预算 = BONES 原档段时间 × 动态乘数（缓冲高 1.3 / 中 1.0 / 低 0.7），高缓冲时可比基线多花 30% 增强时间；3G 的 +0.83 部分来自多花算力而非分配更优 | 重要 | `control/bones_frame_level.py:59-63,216-222` |
| B5 | 720p/1080p 无张量数据，直接退化为"整段固定最高档"，帧级 DP 在高带宽档完全不生效（4g/wifi 增益≈0 的原因） | 重要 | `control/bones_frame_level.py:109-115` |
| B6 | **DP 是离线全知 oracle**：决策前已知该段全部 120 帧增强后的真实 VMAF；在线部署需要逐帧增益预测器，§5 全部数字是上界不是可部署性能 | 重要 | `eval_frame_level_scheduling.py:161-216`；`bones_frame_level.py:93-185` |
| B7 | 测量口径污染：① enhanced 与 reference **双臂都经 cv2 VideoWriter(mp4v) 有损临时编码**（参考帧也被压一次，绝对值整体偏移，对称性使方向/相关性可信）；② base 用 `cv2.resize(INTER_CUBIC)`（:109），播放器实际是 DecoderNvCodec 硬件缩放；③ libvmaf 按 120 帧 clip 独立跑，段边界时域 VMAF 特征重置；④ `n_threads=4`（后立铁律 2）；⑤ 只测 240/360/480，无 720/1080；⑥ 参考是 1080p ladder 成片再 mp4v，非无损母版 | 重要 | `measure_per_frame_vmaf.py:109,196-209,219,296` |
| B8 | 贪心实现有缺陷：每帧全局只被选中一次、不能先 low 后升级，按增益/ms 排序又系统性偏向廉价 low 档，结果在所有格子劣于段固定；贪心行不能作为策略对比证据 | 轻微 | `eval_frame_level_scheduling.py:123-159` |
| B9 | 输出路径硬编码旧 workspace 根（runner :141；eval :340 写脚本 cwd），dca6b85 重组后 json 在 frame_level/ 目录，重跑会在 workspace 根新写一份；延迟表为 3060Ti，现机 5080 未重测 | 轻微 | `run_bones_vs_framelevel.py:65,141` |

## 7. 影响面：历史结论逐条重新定性

| 历史结论/数字 | 复核后定性 |
|---|---|
| "帧级 DP 在 3G 下 QoE +0.83（VMAF +0.77）"（bones_vs_framelevel_results.json） | **降级，不可直接引用**。该数字同时建立在 B1（low 零成本）、B2（反向平滑）、B3（帧内闪烁不计）、B4（1.3 倍预算）之上，且 B6 离线 oracle 不可部署；只能表述为"有缺陷仿真器中的初步正向信号" |
| 离线 DP 紧预算增益 +4.13/+1.64/+2.04 等（frame_level_scheduling_results.json） | **修正为** step=1ms 口径 +3.14/+0.81/+1.49（§5.2），且仍是全知 oracle 上界、张量受 B7 污染；宽裕预算≈0 的结论不变 |
| "帧级内容感知假设成立"（增益∝纹理 r=0.908、∝运动 r=0.516，高运动帧增益差 13.6 分） | **保留（方向级）**。相关性由 18960 帧现算，mp4v 对称污染不改变相关性方向；写论文前需 rawvideo 口径复测确认数值 |
| "段平均抹掉帧间差异，帧级 std 为段间 1.1 倍" | **保留（同口径）**，同样需新口径复测 |
| "USM 逐帧 29.7% 负增益、均值 +4.92（360p）" | **保留方向**，与档案 1 的 USM 结论互证（低分辨率正、不稳定）；数值待统一口径重测 |
| 逐帧张量 `per_frame_quality.json`（2370 条） | **有效资产、挂起待重测**：结构完整、断点续跑脚本可用；B7 口径问题决定它不能直接作为论文数据，但可用于算法原型与相关性分析 |
| "高带宽下帧级方案无增益" | **保留但归因明确**：是 B5（720/1080 无数据直接退化），不是策略在高带宽天然无效 |

## 8. 是否重做：判断与方案

**判断：科学问题（段内增强算力分配）没有被证伪，被证伪的是这版仿真证据；不按原样重做，并入暂停中的"帧级指标体系"任务统一推进。** 修正记账后离线 oracle 在紧预算下仍有 +0.8~+3.4 VMAF 上界，且帧增益与纹理 r=0.908 说明增益可由廉价内容特征预测——这两点支撑继续，但在线可部署性与真实代价（帧内闪烁、真实延迟）必须先补齐。

重做步骤（每步设 GO/NO-GO 门）：

1. **张量重测（帧级指标 A 阶段）**：双 rawvideo 堆叠管道（范本 `pacnet/modern_sr/probe_modern_sr.py`），双臂对称、参考帧不二次编码；base 缩放对齐播放器 DecoderNvCodec；补 720/1080；libvmaf n_threads=2；先小样本与旧张量三方对账（旧 mp4v 口径/BONES 段级表/新 rawvideo 口径）。
2. **修仿真器四件**：DP 离散步 ≤1ms 或直接用精确 ms 浮点预算；平滑惩罚改 `abs()`；QoE 增加帧内 flicker 项（逐帧质量序列的相邻帧差），与旧 QoE 双列对照；删除 1.3/0.7 动态乘数或对基线施加同口径预算，保证等预算对照。
3. **在线可部署版**：用纹理/运动特征做轻量帧增益预测（不依赖增强后 VMAF 先知），预测误差做敏感性分析；这是从 oracle 上界走向可部署策略的关键一步。
4. **仿真门 → 播放器**：修正后在 ≥3 条轨迹、多种子下对 BONES 比胜率/中位；通过后才谈接入真实播放器逐帧决策（与暂停方案 B~E 阶段合并）。

## 9. 踩坑铁律（本工作线新增）

1. **离散化步长必须小于最小动作成本**，否则出现零成本动作，背包约束整体失效（B1）；步长选择要先列全部动作的真实成本。
2. **对照实验预算口径必须一致**：给新策略动态加预算（1.3 倍乘数）而基线不加，增益不可归因于策略（B4）。
3. **新策略引入的新抖动粒度必须进指标**：段级策略改到帧级，QoE 的 oscillation 仍只算段间，等于豁免了策略的主要代价（B3）。
4. **oracle 结果必须显式标注"全知上界、不可部署"**，并在同一篇实验里给出在线预测版的数字，否则审稿无法采信（B6）。
5. cv2 mp4v 不作为 VMAF 测量中间格式（双臂都压也不行，参考帧被污染），统一 rawvideo 管道（B7，与全仓库测量铁律一致）。
6. 仿真器继承基类后，**基类的可行性检查仍用旧段级时间、子类 push 用新 DP 时间**，两处时间口径要一起核对（`bones_frame_level.py:204` vs :238-246）。

## 10. 上下游关系

- 上游：BONES 段级质量表与延迟表 `data/bbb/imdn_bbb_3060ti.json`；ladder 视频 `data/video_mp4box/output_{240,360,480,720,1080}p.mp4`；BONES 仿真器 `simulation/simulator.py:421-433`（enhance_time 形状 (num_bitrate, num_method+1)，索引核对一致，无错位）。
- 平行/下游：
  - "增量传输 + 渐进增强"仿真（基础层先播、按帧 VMAF 增益贪心增强高优帧）与本档案是同一"帧级增益不均"观察的两个出口，归档案 7（vbr），其 3G +0.97 同样是仿真器数字、同样从未接入播放器；
  - PAC-Net halt 控制器（档案 8）是 anytime 出口思想在模型结构侧的延续；
  - `analyze_segments_detail.py` 服务于档案 1 的播放器输出对比；
  - 本档案的张量与 B7 口径结论是暂停中"帧级指标体系 A~E"的直接输入。

## 11. 复盘备注（复活检查清单）

- 复活前先确认：`data/bbb/per_frame_quality.json` 仍是旧 mp4v 口径，**禁止直接作为论文数据源**；只能用于原型验证与相关性分析。
- 张量只覆盖 240/360/480；任何 720p/1080p 帧级结论在重测补档前都不存在（B5）。
- 延迟表是 3060Ti；现机 RTX 5080 上单帧成本需重测后再跑 DP，否则预算格子整体错位。
- 联合仿真器 `control/bones_frame_level.py` 仍在仓库中、未被播放器引用（纯仿真）；复活时按 §8 步骤 2 一次性修 B1~B4，不要在旧数字上叠加新实验。
- B1/C1b 的逐行改法、顺序与验收已统一成文：`docs/frame_scheduling_fix_plan.md`（M0-M5，2026-10-10 现地核码），复活动代码以该清单为准。
- **M0/M1/M2 已完成（2026-10-10，M0 commit 3870f76）**：卡顿单位统一为秒；三处 DP 已全部切到 `simulation/gpu_grid.py` 的 0.1ms 向量化求解（B1/J1 修复，收敛增益见 §5.2）；缓冲推入与决策共用同一张网格成本表并带逐段断言（J10 修复，M2 断言曾抓出 seg158 退化路径 2.95ms 不一致）；`tests/test_gpu_grid.py` 5 项 + `tests/test_freeze_unit.py` PASSED，frame_level/three_way/four_way 四轨迹冒烟通过。B2（线性有符号平滑惩罚）仍待 M3 后复核；B3/B4/B6 未处理，M3/M4/M5 待做。
- 贪心对比行（B8）复活时直接删除或重写（允许帧内升级的两阶段贪心），旧贪心数字不进任何对照。
- 内容特征（motion/texture/scene_change）随张量一同生成、158 段完整，是步骤 3 在线预测器的现成输入，不必重抽。
