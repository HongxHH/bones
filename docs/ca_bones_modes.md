# CA-BONES 各模式

## 概述

CA-BONES（Content-Aware BONES）在 BONES 的 Lyapunov 优化框架基础上，引入内容感知。三种模式通过配置文件的 `ca_mode` 参数切换。

- **代码位置**：`player/istream_player/modules/nes/ca_bones.py`（真实播放器）、`control/bones_ca.py`（仿真器）
- **配置**：`ca_mode` 可选值：`base_known`, `predicted`, `cares`

## Lyapunov 优化核心（`ca_bones.py:_bones_control`）

```
obj_drift = buff_down * seg_time + buff_enh * enhance_time
obj_reward = vmaf_enh_avg + gamma_p          # 各模式不同
obj = (obj_drift - V * obj_reward) / seg_size
decision = argmin(obj)
```

## 模式1：base_known

### 原理

使用段级实际 base quality + per-level 内容因子缩放平均增益。

```
effective_vmaf[i,r,e] = base_quality_seg[i,r] + c_i(e) * avg_gain(r,e)
```

- `base_quality_seg[i,r]`：第 i 段第 r 码率的实际 base VMAF（从 enhance JSON 加载）
- `c_i(e)`：第 i 段第 e 级别的内容因子（per-level）
- `avg_gain(r,e)`：全局平均增强增益

### 内容因子计算（`ca_bones.py:_load_segment_data`）

```
c_i(e) = mean_r(actual_gain_i(r,e)) / mean_i(mean_r(actual_gain_i(r,e)))
```

裁剪到 `[c_min, c_max]`（默认 [0.7, 1.8]），e=0 设为 1.0。

### 配置参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `ca_c_min` | 0.7 | 内容因子下限 |
| `ca_c_max` | 1.8 | 内容因子上限 |
| `ca_enhance_file` | `../data/bbb/imdn_bbb_3060ti.json` | 段级 VMAF 数据 |

### 实验结果（stable3g，+ EWMA延迟反馈）

- avg_vmaf=66.79(+0.32 vs BONES)
- QOE=56.38(-1.58 vs BONES)
- oscillation_vmaf=10.35(+1.85 vs BONES)
- **问题**：oscillation 暴增，完全吃掉 VMAF 优势

## 模式2：predicted

### 原理

用训练好的 RandomForest 模型预测段级 VMAF 增益，替代内容因子缩放。

```
predicted_vmaf[i,r,e] = base_quality_seg[i,r] + model.predict(features(i,r,e))
```

### 预测模型

- **模型**：RandomForest（100棵树，深度10）
- **R²**：0.909，RMSE=3.11
- **特征**（12维）：5内容特征 + 码率/级别归一化 + 5交叉特征
- **文件**：`workspace/vmaf_gain_predictor.pkl`
- **训练脚本**：`workspace/train_vmaf_predictor.py`

### 配置参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `ca_predictor_path` | `../workspace/vmaf_gain_predictor.pkl` | 模型文件 |
| `ca_feature_path` | `data/content_features/bbb_features.json` | 内容特征 |

### 实验结果（stable3g）

- avg_vmaf=64.06(-2.41 vs BONES)
- QOE=51.04(-6.92 vs BONES)
- level分布：0=32, 1=42, 2=55, **3=30**（BONES level3只有8）
- **问题**：预测 VMAF 导致过度自信决策→极端级别选择→振荡/stall

## 模式3：cares

### 原理

**不在质量奖励上做内容感知**（避免 oscillation），而在**资源惩罚**上做内容感知。

```
reward = vmaf_enh_avg(全局平均,稳定) + gamma_p - λ_i(内容感知) * enhance_time
```

- 高内容段 → λ 小 → 资源惩罚小 → 值得增强
- 低内容段 → λ 大 → 资源惩罚大 → 不值得增强

### λ 计算（`ca_bones.py:_init_cares`）

```
content_score_i = normalize(texture_i * motion_i)  # [0, 1]
λ_i = λ_min + (λ_max - λ_min) * (1 - content_score_i)
```

### 配置参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `cares_lambda_min` | 0.0 | 高内容段的 λ（惩罚最小） |
| `cares_lambda_max` | 0.003 | 低内容段的 λ（惩罚最大） |
| `ca_feature_path` | `data/content_features/bbb_features.json` | 内容特征 |

### 实验结果（stable3g，λ_max=0.003）

- avg_vmaf=65.79(-1.0% vs BONES)
- QOE=57.11(-1.5% vs BONES)
- oscillation_vmaf=8.67(+2.0% vs BONES，几乎持平)
- GPU能耗-18.8%, 峰值功耗-43.9%, 资源效率+21.9%
- level分布：0=30, **1=97**, 2=31, 3=1（level1为主）

### 为什么 cares 避免了 oscillation

- 质量奖励用全局平均 `vmaf_enh_avg`（与 BONES 相同），决策保守稳定
- 内容感知只体现在资源惩罚 `λ_i * enhance_time` 上，影响级别选择但不引入质量跳变
- 结果：oscillation 与 BONES 几乎持平（8.67 vs 8.50）

## 三种模式对比

| 模式 | 质量奖励 | 内容感知位置 | oscillation | QoE vs BONES |
|---|---|---|---|---|
| base_known | 段级base+内容因子缩放 | 质量奖励 | 高(+1.85) | -1.58 |
| predicted | RandomForest预测VMAF | 质量奖励 | 很高(+4.49) | -6.92 |
| cares | 全局平均(稳定) | 资源惩罚 | 几乎持平(+0.17) | -1.5% |

**结论**：在质量奖励上做内容感知会增加 oscillation，在资源惩罚上做内容感知更稳定。
