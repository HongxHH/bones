# 帧级调度复活簇统一修改清单

2026-10-10 现地核码后成文。本文只规定**改什么、怎么验**；缺陷现象与历史数字定性见三份工作线档案（`workspace/vbr_sims/frame_level/README.md` §6 B1、`workspace/vbr_sims/lookahead/README.md` §6 I1/I2/I3/J1/J2/J10、`workspace/vbr_sims/md_qoe/README.md` §6 C1b），不重复。

约定：本清单经审阅后再动代码；按 M0→M1/M2→M3/M4→M5 顺序提交，每步带验收；不在旧数字上叠加新实验，重跑结果写带时间戳的新 json，不覆盖封板档案数据。

## 缺陷与修改点总表

| 编号 | 缺陷 | 现地位置（2026-10-10 核） | 修改点 |
|---|---|---|---|
| C1b | 卡顿惩罚单位 ms 与秒混用，仿真器/播放器 QoE 尺度差 1000 倍 | `simulation/simulator.py:217,221`；`control/bones_frame_level.py:310,312`；`control/bones_incremental.py:261,263`；`control/bones_combined.py:302,304`（ms/10）vs `player/istream_player/modules/analyzer/analyzer.py:801,806`（秒/10，`total_stall_duration` 由 `_seconds_since` 累积，:550,577,581）；`simulation/buffer_model.py:35` 注释明示 idle_time 为毫秒 | M0 |
| B1 | DP 时间网格 10ms **向下**取整，最便宜档位零成本 | `workspace/vbr_sims/frame_level/eval_frame_level_scheduling.py:175,185`；`control/bones_frame_level.py:52,141` | M1 |
| J1 | DP 时间网格 10ms **向上**取整（`int(c/10)+1`，如 3ms→10ms），浅档超额计费、偏向选深 | `control/bones_combined.py:106,120` | M1 |
| J10 | DP 按网格成本决策、缓冲按连续毫秒记账，求解与记账不是同一个调度 | 决策 `control/bones_combined.py:120-156`；记账 `:236-245`（连续 `enhance_time/120` 求和） | M2 |
| I1/J2 | 保留 BONES 整段可行性闸门，"部分增强救回不可行段"从不触发 | `control/bones_incremental.py:174`；`control/bones_combined.py:213`；`control/bones_frame_level.py:204`（同一闸门） | M3 |
| I2/J3 | 可用时间白送半个段长 `buff + seg_time/2` | `control/bones_incremental.py:106`；`control/bones_combined.py:224` | M4（并入 M3 队列仿真） |
| I3/J4 | 选帧按真实逐帧增益 argsort，全知 oracle | `control/bones_incremental.py:136`；`control/bones_combined.py:161,189` | M5 |

## M0：统一卡顿单位为秒（最先做，全局前置）

**改法**

1. 缓冲模型内部保持毫秒不动（`buffer_model.py` 全模型以 ms 推进），只在 QoE 边界换算：四处 report 的 `avg_freeze = idle_time / num_seg` 统一改为 `idle_time / 1000.0 / num_seg`，使 `/10` 后与播放器同口径（1 秒卡顿惩罚 0.1）。位置：`simulation/simulator.py:217`、`control/bones_frame_level.py:310`、`control/bones_incremental.py:261`、`control/bones_combined.py:302`。
2. `simulation/buffer_model.py:35` 注释补一句"QoE 边界必须 /1000 换算为秒"；report 字典同时保留 `"freeze time"`（ms，内部口径）与新增 `"freeze time sec"`，避免下游再猜单位。
3. 播放器侧不改（已为秒）。`player/istream_player/core/analyzer.py` 无 QoE 计算，已核无需动。

**验收**

- 单元测试：构造 idle_time=1000ms、num_seg=1，仿真器 QoE 卡顿项 = −0.1，与播放器同构造一致（修前为 −100）。
- 影响面明示：修后所有仿真器历史 QoE 绝对值改变（差网下卡顿惩罚缩小 1000 倍），quality/oscillation 两项不变；凡引用仿真器 QoE 的封板数字（#2/#3/#4/#7 仿真器侧结果）标注"卡顿口径修正前，不可与播放器 QoE 同列"，播放器实测数字不受影响、仍为权威。

## M1：DP 时间网格改 0.1ms 向量化精确网格

**规则**：离散步长必须使最小正动作成本的表示误差可忽略，且不允许零成本动作。三文件中单帧成本对同（分辨率, 档位）为常数（来自整段 enhance_time/120），最小约 5.1ms（IMDN_RTC l2）；10ms 网格无论向下（B1，免费动作）还是向上（J1，浅档最贵 18% 失真）都扭曲档位相对价格。

**改法**

1. 步长 `step_ms=0.1`，成本 `cost_steps = int(round(c / step_ms))`；DP 转移用 numpy 向量化（每帧 4 次切片 max 运算，状态列约 40001，120 帧×4×40001 元素级运算，秒级），替换现有 Python 三重循环：
   - `workspace/vbr_sims/frame_level/eval_frame_level_scheduling.py:174-218`
   - `control/bones_frame_level.py:124-185`
   - `control/bones_combined.py:106-156`
2. 启动断言：最小正成本 `cost_steps ≥ 50`（≥5ms）、无零成本动作；预算列数由 `gpu_budget_ms/step_ms` 导出，删除 `bones_combined.py:127` 的硬编码 `min(...,5000)` 帽。
3. 误差上界：每帧 ≤0.05ms、每段（120 帧）≤6ms，占 4s 段预算 0.15%，在报告中输出实际网格误差。

**验收**

- 重跑 `eval_frame_level_scheduling.py`，与档案 #2 §5.2 已记录的 step=1ms 复核值对齐并更精确：240/2000 +3.14、360/2000 +0.81、360/4000 +3.41、480/2000 +1.49、480/4000 +0.42（0.1ms 口径应与 1ms 口径差 ≤0.1 VMAF）。
- 断言：任意档位 `cost_steps > 0`；DP 解的网格总成本 ≤ 预算步数。

## M2：决策与记账同一套成本（J10）

**改法**

1. 三处 DP 回溯后，推入增强缓冲的时间一律等于"决策实际占用的网格成本之和"，禁止另一套连续时间求和：
   - `control/bones_combined.py:236-245` 改为对 `decisions` 逐帧累加 `cost_steps[level]*step_ms`（与 DP 同一张成本表）；
   - `control/bones_frame_level.py:237-246` 已用 `gpu_used=best_c*step`，改为回溯决策求和（与 best_c 恒等并加断言），保持口径显式；
   - `bones_incremental.py:203-208` 同样改网格成本表。
2. 每段断言 `pushed_segment_time == Σ 决策帧成本`，违例即报错；报告附 `timing_residual_ms`（网格记账 vs 真实 enhance_time 之差）。

**验收**：全轨迹运行断言零违例；联合方案紧张轨迹结果相对修前的变化方向记入新 json（修前 −0.88/−1.12 含 J1 偏置，不可并列引用）。

## M3+M4：删整段闸门，建帧级跨段 GPU 工作队列（复活核心）

闸门（I1/J2）使部分增强只在"整段本来就来得及"时进入，机制结构性空转；`seg_time/2` 赠送（I2/J3）用无来源时间掩盖队列拥塞。两项必须一起改，没有中间态。

**改法（新增仿真模型，不在旧 buffer_enhance 上打补丁）**

1. 在 `simulation/` 新增帧级增强工作队列（如 `frame_enhance_queue.py`）：下载完成时入队 120 个帧作业 `(seg, frame, level_cap)`，每作业带**播放截止时刻**；GPU worker 按截止时刻 EDF 消费（截止时刻 = 该帧预计播放墙钟时间，由下载/播放缓冲推进逐帧给出），到点未完成的帧按基础质量播放（记 missed frame，对应播放器 fast_complete 的帧级口径）。
2. 删除三处整段闸门（`bones_incremental.py:174`、`bones_combined.py:213`、`bones_frame_level.py:204`）；BONES 的 action_enh 改为每帧 level_cap，不再是"整段做/不做"开关。
3. 删除两处 `seg_time/2`（:106、:224）；每帧可用增强时间只能由队列仿真给出，不允许解析式赠送。
4. 播放器侧已有同构实现可直接移植：`player/istream_player/modules/enhancer/pacnet_enhancer.py:153-169` 的可调度性前瞻（逐级选满足 lead ≥ l_exit×剩余帧×安全系数 的最高出口），仿真器队列的截止时间判定与它保持同一语义，避免两套调度学。
5. 质量记账逐帧化：完成帧取逐帧增强 VMAF、missed 帧取基础 VMAF，段 VMAF 为帧均值（数据 `data/bbb/per_frame_quality.json`；该张量的测量口径属 M1 质量债，见下"前置依赖"）。

**验收**

- 构造一条闸门必然触发的紧预算轨迹：旧代码不可行段 enhanced_count=0；新队列这些段 enhanced 帧数 >0 且 missed 帧数被显式统计、与卡顿账平。
- 队列守恒断言：入队帧数 = 完成 + missed + 在途；GPU 忙碌时间总和 = Σ 完成帧成本（M2 同一成本表）。
- 无任何 `seg_time/2` 类时间赠送（grep 断言）。

## M5：选帧从 oracle 改因果（M3/M4 验收后才做）

1. 现有 argsort（`bones_incremental.py:136`、`bones_combined.py:161,189`）保留为**标注 oracle 上界**（脚本加 `--oracle` 开关，输出文件名带 oracle），不得作为可部署结果。
2. 因果版只能用下载完成时可获得的信息排序（轻量运动/纹理预统计或延迟表内特征）；预测器评价用排序相关与后悔值，不用 R²。
3. 档案 #4 教训前置：不准的预测比没预测更有害（离线 R²≈0.91 在线 QoE −6.92）。因果版必须先过保守性校准（预测增益高估率有上界），否则策略退化为"永远按 l2 全增强"。

## 实施顺序与提交边界

| 步骤 | 内容 | 依赖 | 产出 |
|---|---|---|---|
| 1 | M0 单位统一 + 单元测试 | 无 | 仿真器四处 report、buffer_model 注释、测试 |
| 2 | M1+M2 网格与记账同源 + 向量化 DP | M0 | 三处 DP 重写、断言、对齐 #2 §5.2 复核值 |
| 3 | M3+M4 帧级工作队列 + 删闸门/赠送 | M1/M2 | `simulation/frame_enhance_queue.py`、三控制器接入 |
| 4 | M5 因果选帧（oracle 先单列跑通） | M3/M4、#4 教训 | `--oracle` 开关、因果排序器 |
| 5 | 六真实轨迹（0913/0914/0921 + 三条 4G）四策略对比 | 1-4 | 每策略独立卡顿向量（禁共用，md_qoe C2 教训）、仿真器/播放器同单位交叉核对 |

**前置依赖（质量数，不是调度逻辑）**：M0-M4 可用现有 `per_frame_quality.json` 验证调度行为；但任何对外引用的质量数字必须等帧级指标 A 阶段用连续 rawvideo libvmaf 管道重测质量表（范本 `workspace/pacnet/sr_aware_alloc/measure_seg_rd.py:69-145`；质量债见 #8 README §6 M1、#9 G1、#14 P3）。调度修复与质量重测是两条线，互不阻塞开工，但数字汇合前不做结论。

**明确不做**：本清单不含新指标设计（归暂停中的帧级指标体系 C 阶段，需求清单见 `docs/custom_qoe_metrics.md` 与 md_qoe 档案 §8）；不改播放器（播放器已是连续时间 + EDF halt）；M0-M2 验收前不跑新轨迹实验。
