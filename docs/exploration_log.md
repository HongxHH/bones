# 探索历史

按时间线记录所有试过的方法、参数、结果、失败原因。

---

## 阶段1：环境搭建与基线确认

**时间**：2026-08 前期

- 梳理目录结构，清理根目录冗余文件，重建 Git 仓库
- 读懂 iStreamPlayer 全链路（下载→解码→增强→渲染）
- 确认三种渲染模式：OpenGL/EGL/Headless，选择 OpenGL + 注释帧输出（无头跑实验）
- 跑通 BONES 基线实验（~12分钟/次）
- **BONES 基线结果**（stable3g 轨迹）：avg_vmaf=66.47, QOE=57.96, num_stall=16, dur_stall=22.22s

---

## 阶段2：内容感知调度初探

### 2.1 CA-BONES base_known 模式

- **方法**：段级实际 base quality + per-level 内容因子缩放平均增益
- **公式**：`effective_vmaf = base_quality_seg[i,r] + c_i(e) * avg_gain(r,e)`
- **结果**：avg_vmaf=66.79(+0.32), QOE=56.38(-1.58), stall_dur=90.05s(暴增)
- **问题**：stall 时长从 22s 暴增到 90s，QoE 反而更低

### 2.2 层面1：EWMA 实测延迟反馈

- **方法**：abort/fast_complete/正常完成三种路径都更新延迟表，未访问组合用比例推算
- **代码**：`enhancer_imdn_always.py:_update_latency_ewma()`
- **结果**：QoE 差距从 -4.15 缩小到 -1.58
- **结论**：静态延迟表低估了高级别延迟，导致过度选高级别→stall

### 2.3 层面2：per-segment KNN 延迟预测

- **方法**：基于内容特征（纹理/运动/场景切换）用 KNN 预测每段延迟缩放因子
- **代码**：`enhancer_imdn_always.py:predict_latency_factor()`
- **参数**：c_min=0.7, c_max=1.8, KNN k=5, 冷启动阈值=10段
- **结果**：avg_vmaf=66.89(+0.41), QOE=55.15(-2.81), stall_dur=73.77s
- **问题**：VMAF 最高但 stall 仍大，KNN 冷启动期预测不准
- **level分布**：0=26, 1=19, 2=98, 3=16（level2为主）

---

## 阶段3：VMAF 增益预测模型

### 3.1 模型训练

- **数据**：159段×5码率×5级别真实 VMAF + 5维内容特征
- **模型**：RandomForest（100棵树，深度10），12维特征（5内容+码率/级别+5交叉）
- **结果**：R²=0.909, RMSE=3.11, 各级别 R²>0.89
- **特征重要性**：纹理×运动(26%) > 码率(14%) > 运动×级别(13%) > 纹理×级别(12%)
- **文件**：`workspace/vmaf_gain_predictor.pkl`

### 3.2 仿真器验证

- **5种模式对比**：BONES / base_known / predicted / oracle / joint(质量+延迟)
- **结果**：所有 CA 模式平均 QoE 都低于 BONES（oracle -1.79, predicted -3.37, joint -4.07）
- **根因**：仿真器无 per-segment 延迟变化，内容感知质量预测导致决策方差增大→振荡/stall
- **结论**：仿真器不是验证内容感知自适应的正确场所

### 3.3 真实播放器 predicted 模式

- **结果**：avg_vmaf=64.06(-2.41), QOE=51.04(-6.92)
- **level分布**：0=32, 1=42, 2=55, 3=30（极端选择，level3暴增）
- **问题**：预测 VMAF 导致过度自信决策，极端级别选择→振荡/stall
- **结论**：不准确的预测比没有预测更有害

---

## 阶段4：CARES 资源效率方向

### 4.1 核心思想

不在质量奖励上做内容感知（导致振荡），而在**资源惩罚**上做内容感知：
```
reward = vmaf_enh_avg(全局平均,稳定) + gamma_p - λ_i(内容感知) * enhance_time
```
- 高内容段 → λ 小 → 资源惩罚小 → 值得增强
- 低内容段 → λ 大 → 资源惩罚大 → 不值得增强

### 4.2 λ_max=0.01（第一次）

- **结果**：avg_vmaf=61.11(-8.1%), QOE=52.10(-10.1%), GPU能耗-52.1%
- **level分布**：0=120, 1=27, 2=12, 3=0（过度保守，120段不增强）
- **问题**：λ 太大，大部分段被惩罚到不增强

### 4.3 λ_max=0.003（第二次，当前最优）

- **结果**：
  - avg_vmaf=65.79(-1.0%), QOE=57.11(-1.5%)
  - GPU平均功耗-18.9%, 总能耗-18.8%, 峰值功耗-43.9%, 峰值温度-12.1%
  - 总增强时间-13.3%, 资源效率(VMAF/kJ)+21.9%
- **level分布**：0=30, 1=97, 2=31, 3=1（level1为主，轻量增强覆盖大部分段）
- **结论**：QoE 仅降1.5%，GPU资源显著降低，资源效率提升21.9%

---

## 阶段5：根因分析

### 5.1 为什么内容感知 QoE 低于 BONES

- **QoE 公式**：`qoe = avg_vmaf - oscillation_vmaf - avg_freeze/10`
- **拆解**（BONES vs CA-BONES layer1）：
  - avg_vmaf: +0.32（CA更高）
  - oscillation_vmaf: +1.85（CA暴增，完全吃掉VMAF优势）
  - avg_freeze/10: +0.043（CA略高）
- **根因**：内容感知调度增加决策方差→相邻段质量差异大→oscillation暴增→QoE下降
- **结构性问题**：内容感知天然增加振荡，BONES全局平均更保守稳定

### 5.2 fast_complete 不是我们独有

- `config.py:111`: `use_fast_complete = True` 是默认值
- BONES 和 CA-BONES 实验都开启了 fast_complete
- fast_complete 段 VMAF 是加权平均：`base*(1-ratio) + enhance*ratio`，比完整增强低

---

## 阶段6：VMAF 平滑机制

### 6.1 级别切换惩罚（失败）

- **方法**：在奖励中加入 `α × |level_i - level_{i-1}|`，α=2.0
- **结果**：avg_vmaf=67.13, QOE=55.03, oscillation=**12.06**（反而升高）
- **失败原因**：同一级别下不同段的 VMAF 差异很大（base quality + 内容因子不同），级别不变不代表 VMAF 不变
- **教训**：必须直接惩罚 VMAF 差值，而非级别差值

### 6.2 VMAF 差值切换惩罚（成功）

- **方法**：在奖励中加入 `α × |vmaf_i - vmaf_{i-1}|`，α=0.3
- **代码**：`ca_bones.py:_bones_control()`，`self.last_vmaf` 追踪上一段预测 VMAF
- **结果（单独使用）**：avg_vmaf=67.80(+1.33), QOE=60.14(+2.18), oscillation=7.60(-0.90)
- **结论**：直接惩罚 VMAF 差值精确命中 oscillation 根因

---

## 阶段7：CARES + VMAF 平滑组合

### 7.1 组合公式

```
reward = vmaf_enh_avg + gamma_p - λ_i * enhance_time - α * |vmaf_i - vmaf_{i-1}|
```

两个机制正交：CARES 影响"选什么级别"（资源效率），VMAF 平滑影响"级别是否变化"（稳定性）。

### 7.2 第一次实验（α=0.3, λ=0.003）

- **结果**：avg_vmaf=67.99, QOE=**64.28**, oscillation=**3.67**, GPU能耗=26.23kJ
- **level分布**：0=39, 1=44, 2=75, 3=1（level2为主）
- **结论**：QoE 超越 BONES 6.32 分，oscillation 降低 57%，GPU 能耗降低 16.3%

### 7.3 第二次重复实验（α=0.3, λ=0.003）

- **结果**：avg_vmaf=67.60, QOE=**63.66**, oscillation=**3.91**, GPU能耗=24.72kJ
- **level分布**：0=35, 1=92, 2=32, 3=0（level1为主）
- **发现**：两次实验 level 分布完全不同，但 QoE 都很高 → 路径依赖/双稳态现象

### 7.4 路径依赖根因分析

- **前73段完全一致**，第74段开始分歧
- **原因**：VMAF 平滑惩罚鼓励保持与上一段接近的 VMAF，一旦选了某个级别就形成"惯性"
- **双稳态**：level1主导和 level2主导都是稳定均衡点，初始段微小差异决定路径
- **不是 BONES 本身的问题**：原始 BONES 无 VMAF 平滑，level 分布稳定（level2=89段）

---

## 阶段8：系统调参（进行中）

### 8.1 α 调参（λ=0.003 固定）

| α | avg_vmaf | QOE | oscillation | dur_stall | level分布 | GPU能耗 |
|---|---|---|---|---|---|---|
| 0.1 | 66.75 | 60.04 | 6.69 | 33.54s | 32/48/78/1 | 27.77kJ |
| 0.2 | 66.82 | 62.12 | 4.68 | 43.77s | 36/96/26/1 | 24.28kJ |
| 0.3 | 67.60-67.99 | 63.66-64.28 | 3.67-3.91 | 55.97-67.13s | 路径依赖 | 24.72-26.23kJ |
| 0.5 | - | - | - | - | - | 待跑 |

**趋势**：α 越大 → oscillation 越低 → QoE 越高；α=0.3 出现路径依赖。

### 8.2 λ 调参（待跑，固定最优 α）

计划测试 λ=0.0/0.002/0.005，找到资源效率与 QoE 的最佳平衡点。

### 8.3 环境问题

- 发现两个残留的 `run_stage_a_sim.py` 仿真进程（PID 67772, 82644），已运行 5 小时，已杀掉
- 这些进程可能影响了部分实验的 GPU 资源，调参实验需在干净环境下重跑（计划晚上进行）

---

## 阶段9：多段前瞻规划（Lookahead / MPC）

### 9.1 动机

- 单段贪心决策导致路径依赖/双稳态（阶段7.4）
- 无法利用未来内容和带宽信息
- 目标：滚动时域优化，每步联合决策未来 N 段，只执行第一段

### 9.2 实现

- **代码**：`control/bones_ca.py:_lookahead_control(N)`
- **缓冲区模型**：连续速率近似——增强以 `seg_time/en_time` 速率从下载缓冲搬到增强缓冲，播放以 1:1 消费增强缓冲
- **目标函数**：N 步 `(drift - V*reward)/seg_size` 求和 + 显式卡顿惩罚
- **带宽预测**：当前 EWMA 估计作为未来 N 段恒定预测
- **复杂度**：O((num_br × num_lvl)^N)，N=2 时 400 种组合，可接受

### 9.3 dl_time 单位 bug（关键修复）

- **现象**：前瞻在 4G/高带宽轨迹上 QoE 仅 ~80（BONES 基线 ~96），VMAF ~84（基线 ~97）
- **根因**：`dl_time = seg_size / bandwidth * 1000` 把字节（bytes）当成了千比特（Kb）。`seg_size` 单位是 bytes，`bandwidth` 单位是 kbps，正确公式应为 `dl_time = seg_size * 8 / bandwidth`
- **影响**：下载时间被高估 125 倍（3.2MB 段在 36Mbps 下实际 719ms，代码算成 90 秒），导致前瞻预测 massive 卡顿，完全失真
- **修复**：`control/bones_ca.py:378` 改为 `dl_time = self.seg_size[idx + k, d] * 8.0 / bandwidth`
- **教训**：仿真器中手动计算物理量时必须严格核对单位，特别是 bytes vs bits

### 9.4 多轨迹验证结果（N=2，修复后）

| 轨迹 | 带宽场景 | BONES QoE | CA-N1 QoE | CA-N2 QoE | 结果 |
|---|---|---|---|---|---|
| 3G_stable | 低带宽稳定 | 77.64 | 76.92 | **78.57** | ✅ 胜 |
| 3G_var1 | 低带宽波动 | -111.78 | -141.55 | **-99.18** | ✅ 大胜 |
| 4G_bus | 高带宽移动 | 96.57 | 96.94 | **97.23** | ✅ 胜 |
| 4G_car | 高带宽移动 | 96.24 | 96.37 | 94.66 | ⚠️ 略输 |
| 4G_bicycle | 高带宽移动 | 96.49 | 96.94 | **97.14** | ✅ 胜 |
| hd_fixed | 极高带宽固定 | 89.96 | 90.35 | 86.16 | ⚠️ 略输 |

**6 轨迹中 4 胜 2 负**。2 个略输轨迹（4G_car、hd_fixed）可能与恒定带宽预测不准确或卡顿惩罚偏保守有关，待进一步调优。

### 9.5 仿真器脚本

- `workspace/run_lookahead_sim.py`：单轨迹前瞻仿真
- `workspace/run_lookahead_multi_trace.py`：多轨迹批量验证
- 结果保存为 `lookahead_sim_results.json` / `lookahead_multi_trace_results.json`

---

## 失败方法汇总（更新）

| 方法 | 结果 | 失败原因 |
|---|---|---|
| CA-BONES base_known（无延迟反馈） | QoE -4.15 | 静态延迟表低估高级别延迟→stall |
| CA-BONES predicted（真实播放器） | QoE -6.92 | 预测VMAF导致过度自信→极端级别选择→振荡 |
| CA-BONES joint（仿真器，质量+延迟预测） | QoE -4.07 | 仿真器无真实per-seg延迟变化 |
| per-seg KNN延迟预测（层面2） | QoE -2.81 | KNN冷启动期预测不准→stall |
| CARES λ_max=0.01 | QoE -10.1% | λ太大→过度保守→120段不增强 |
| **级别切换惩罚（α=2.0）** | **oscillation=12.06** | **同一级别VMAF差异大，级别不变≠VMAF不变** |

## 失败方法汇总

| 方法 | 结果 | 失败原因 |
|---|---|---|
| CA-BONES base_known（无延迟反馈） | QoE -4.15 | 静态延迟表低估高级别延迟→stall |
| CA-BONES predicted（真实播放器） | QoE -6.92 | 预测VMAF导致过度自信→极端级别选择→振荡 |
| CA-BONES joint（仿真器，质量+延迟预测） | QoE -4.07 | 仿真器无真实per-seg延迟变化 |
| per-seg KNN延迟预测（层面2） | QoE -2.81 | KNN冷启动期预测不准→stall |
| CARES λ_max=0.01 | QoE -10.1% | λ太大→过度保守→120段不增强 |
