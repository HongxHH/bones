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
- **参数**：c_min=0.7, c_max=1.8, KNN k=5, 冷启动阈值=10段
- **结果**：avg_vmaf=66.89(+0.41), QOE=55.15(-2.81), stall_dur=73.77s
- **问题**：VMAF 最高但 stall 仍大，KNN 冷启动期预测不准

---

## 阶段3：VMAF 增益预测模型

### 3.1 模型训练

- **数据**：159段×5码率×5级别真实 VMAF + 5维内容特征
- **模型**：RandomForest（100棵树，深度10），12维特征
- **结果**：R²=0.909, RMSE=3.11
- **文件**：`workspace/vmaf_gain_predictor.pkl`

### 3.2 仿真器验证

- **结果**：所有 CA 模式平均 QoE 都低于 BONES
- **根因**：仿真器无 per-segment 延迟变化，内容感知质量预测导致决策方差增大→振荡/stall
- **结论**：仿真器不是验证内容感知自适应的正确场所

### 3.3 真实播放器 predicted 模式

- **结果**：avg_vmaf=64.06(-2.41), QOE=51.04(-6.92)
- **问题**：预测 VMAF 导致过度自信决策，极端级别选择→振荡/stall
- **结论**：不准确的预测比没有预测更有害

---

## 阶段4：CARES 资源效率方向

### 4.1 核心思想

不在质量奖励上做内容感知（导致振荡），而在**资源惩罚**上做内容感知：
```
reward = vmaf_enh_avg(全局平均,稳定) + gamma_p - λ_i(内容感知) * enhance_time
```

### 4.2 λ_max=0.003（当前最优）

- **结果**：avg_vmaf=65.79(-1.0%), QOE=57.11(-1.5%), GPU能耗-18.8%, 资源效率+21.9%
- **结论**：QoE 仅降1.5%，GPU资源显著降低

---

## 阶段5：根因分析

### 5.1 为什么内容感知 QoE 低于 BONES

- **QoE 公式**：`qoe = avg_vmaf - oscillation_vmaf - avg_freeze/10`
- **根因**：内容感知调度增加决策方差→相邻段质量差异大→oscillation暴增→QoE下降
- **结构性问题**：内容感知天然增加振荡，BONES全局平均更保守稳定

### 5.2 fast_complete 不是我们独有

- `config.py:111`: `use_fast_complete = True` 是默认值
- BONES 和 CA-BONES 实验都开启了 fast_complete

---

## 阶段6：VMAF 平滑机制

### 6.1 级别切换惩罚（失败）

- **方法**：在奖励中加入 `α × |level_i - level_{i-1}|`，α=2.0
- **结果**：oscillation=**12.06**（反而升高）
- **失败原因**：同一级别下不同段的 VMAF 差异很大，级别不变不代表 VMAF 不变

### 6.2 VMAF 差值切换惩罚（成功）

- **方法**：在奖励中加入 `α × |vmaf_i - vmaf_{i-1}|`，α=0.3
- **结果**：avg_vmaf=67.80(+1.33), QOE=60.14(+2.18), oscillation=7.60(-0.90)

---

## 阶段7：CARES + VMAF 平滑组合

### 7.1 组合结果（α=0.3, λ=0.003，两次均值）

| 指标 | BONES | 组合 | 变化 |
|------|-------|------|------|
| avg_vmaf | 66.47 | 67.80 | +2.0% |
| QOE | 57.96 | 63.97 | +10.4% |
| oscillation | 8.50 | 3.79 | -55.4% |
| GPU能耗 | 31.34kJ | 25.48kJ | -18.7% |

### 7.2 路径依赖/双稳态

- 两次实验 level 分布完全不同（level2主导 vs level1主导），但 QoE 都很高
- **原因**：VMAF 平滑惩罚鼓励保持与上一段接近的 VMAF，一旦选了某个级别就形成"惯性"

---

## 阶段8：多段前瞻规划（Lookahead / MPC）

### 8.1 dl_time 单位 bug（关键修复）

- **根因**：`dl_time = seg_size / bandwidth * 1000` 把字节（bytes）当成了千比特（Kb）
- **影响**：下载时间被高估 125 倍，前瞻预测 massive 卡顿
- **修复**：`dl_time = seg_size * 8 / bandwidth`

### 8.2 多轨迹验证结果（N=2，修复后）

**6轨迹中4胜2负**。2个略输轨迹（4G_car、hd_fixed）可能与恒定带宽预测不准确有关。

---

## 阶段9：多任务增强（Multi-Task Enhancement）

**时间**：2026-08-28

### 9.1 动机

- BONES 只支持 SR 超分增强，增强类型单一
- 高带宽下直接下1080p，增强完全无用（"最高码率不增强"假设）
- 目标：支持 SR + 去模糊 + 插帧等多种增强组合，突破1080p限制

### 9.2 实现

- **MultiTaskEnhancer**：`player/istream_player/modules/enhancer/enhancer_multitask.py`
  - 扩展 IMDNEnhancer，支持5种任务组合（none/sr/sr_deblur/deblur/sr_deblur_fi）
  - NAFNet-Small 去模糊模型（2.68M params, FP16, 7.7MB显存）
  - 去模糊在输入分辨率运行（240p~16ms, 1080p~31ms），先去模糊再超分
- **MTBONESController**：`player/istream_player/modules/nes/mt_bones.py`
  - 资源感知GPU状态追踪 + 内容感知任务选择
  - Lyapunov优化选(码率,任务组合)对

### 9.3 1080p非超分增强

- **问题**：BONES假设scale=1（1080p）不需要增强，直接abort
- **方案**：将1080p加入`deblur_only_set`，允许combo 3（deblur-only）通过
- SR组合对1080p标记为无效延迟(1e6)，控制器自然回避
- **意义**：突破"最高码率不增强"限制，1080p仍可通过去模糊提升感知质量

### 9.4 hd_fs轨迹带宽发现

- **现象**：控制器总是选1080p+none，增强从不触发
- **根因**：hd_fs轨迹平均带宽15.7Mbps（范围1.2-28Mbps），足够直接下载1080p
- **结论**：多任务增强的价值在中等带宽(1-5Mbps)轨迹中更明显

### 9.5 待解决

- deblur对1080p的VMAF增益数据缺失（控制器不知道deblur能提升质量）
- FI（插帧）未实现（RIFE模型未下载）
- 中等带宽轨迹对比实验待跑

---

## 失败方法汇总

| 方法 | 结果 | 失败原因 |
|---|---|---|
| CA-BONES base_known（无延迟反馈） | QoE -4.15 | 静态延迟表低估高级别延迟→stall |
| CA-BONES predicted（真实播放器） | QoE -6.92 | 预测VMAF导致过度自信→极端级别选择→振荡 |
| CA-BONES joint（仿真器） | QoE -4.07 | 仿真器无真实per-seg延迟变化 |
| per-seg KNN延迟预测 | QoE -2.81 | KNN冷启动期预测不准→stall |
| CARES λ_max=0.01 | QoE -10.1% | λ太大→过度保守→120段不增强 |
| 级别切换惩罚（α=2.0） | oscillation=12.06 | 同一级别VMAF差异大，级别不变≠VMAF不变 |

---

## 工程清理记录（2026-08-28）

- 删除 `enhance/IMDN/dataset/bbb/`：IMDN训练数据集，36.8GB，推理不需要
- 删除 `player/output/mt_smoke_test/`、`mt_smoke_test2/`：旧冒烟测试输出，301.6MB
- 删除 `player/output/istream_run_dir/`：临时文件，93.8MB
- 清理 `workspace/` 版本化脚本：保留最新v4，删除v1-v3及旧JSON结果
- 更新 `docs/current_state.md`：从前瞻规划阶段更新到多任务增强阶段
