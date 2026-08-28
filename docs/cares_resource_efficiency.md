# CARES 资源效率方向

## 核心思想

不追求 QoE 超越 BONES，而是追求**相同 QoE 下的 GPU 资源效率提升**。

**关键洞察**：BONES 增强分配与内容完全脱节——Level 3（最高增强，性价比最低）被分配给内容最低的段，Level 0（不增强）反而是内容最高的段。内容感知方法的机会在**资源分配**而非**质量奖励**。

## 原理

### 奖励函数

```
reward = vmaf_enh_avg(全局平均,稳定) + gamma_p - λ_i(内容感知) * enhance_time
```

- **质量奖励用全局平均**：与 BONES 相同，决策保守稳定，oscillation 不增加
- **资源惩罚内容感知**：λ_i 随内容复杂度变化

### λ 计算

```
content_score_i = normalize(texture_i * motion_i)  # [0, 1]
λ_i = λ_min + (λ_max - λ_min) * (1 - content_score_i)
```

- 高内容段（content_score 高）→ λ 小 → 资源惩罚小 → 值得增强
- 低内容段（content_score 低）→ λ 大 → 资源惩罚大 → 不值得增强

### 代码位置

- 控制器：`player/istream_player/modules/nes/ca_bones.py`
  - `_init_cares()`：加载内容特征，计算 λ_i
  - `_bones_control()`：CARES 模式下 reward 减去 `λ_i * enhance_time`
- 配置：`player/ca_cares_config.json`

## 配置参数

| 参数 | 默认值 | 含义 |
|---|---|---|
| `ca_mode` | `cares` | 启用 CARES 模式 |
| `cares_lambda_min` | 0.0 | 高内容段的 λ（惩罚最小） |
| `cares_lambda_max` | 0.003 | 低内容段的 λ（惩罚最大） |
| `ca_feature_path` | `data/content_features/bbb_features.json` | 内容特征文件 |
| `use_content_latency` | false | 是否启用 per-seg 延迟预测（CARES 建议关闭） |

## 实验结果（stable3g 轨迹，RTX 3060Ti）

### λ_max 调参对比

| λ_max | avg_vmaf | QOE | QoE变化 | GPU能耗 | 能耗变化 | level分布(0/1/2/3) |
|---|---|---|---|---|---|---|
| 0.01 | 61.11 | 52.10 | -10.1% | 15.00kJ | -52.1% | 120/27/12/0 |
| **0.003** | **65.79** | **57.11** | **-1.5%** | **25.45kJ** | **-18.8%** | **30/97/31/1** |
| 0(BONES) | 66.47 | 57.96 | 0% | 31.34kJ | 0% | 21/41/89/8 |

### λ_max=0.003 详细指标

| 指标 | BONES | CARES | 变化 |
|---|---|---|---|
| avg_vmaf | 66.47 | 65.79 | -1.0% |
| QOE | 57.96 | 57.11 | -1.5% |
| oscillation_vmaf | 8.50 | 8.67 | +2.0% |
| num_stall | 16 | 13 | -18.8% |
| dur_stall | 22.22s | 22.55s | +1.5% |
| GPU平均功耗 | 47.14W | 38.22W | **-18.9%** |
| GPU峰值功耗 | 236.65W | 132.82W | **-43.9%** |
| GPU总能耗 | 31.34kJ | 25.45kJ | **-18.8%** |
| GPU峰值温度 | 58.0°C | 51.0°C | **-12.1%** |
| 总增强时间 | 90.9s | 78.7s | **-13.3%** |
| 资源效率(VMAF/kJ) | 2.12 | 2.58 | **+21.9%** |

### 增强级别分配变化

| Level | BONES | CARES | 说明 |
|---|---|---|---|
| 0(不增强) | 21 | 30 | 低内容段跳过增强 |
| 1 | 41 | **97** | 轻量增强覆盖大部分段 |
| 2 | 89 | 31 | 只有高内容段用中等增强 |
| 3 | 8 | 1 | 几乎消除最高级增强 |

CARES 将增强主力从 level2（89段）转为 level1（97段），用轻量增强覆盖大部分段，重增强只留给高内容段。

## 为什么 CARES 有效

1. **避开 oscillation 陷阱**：质量奖励用全局平均（与 BONES 相同），不引入质量跳变
2. **内容感知在资源侧**：λ_i 影响级别选择，但不直接改变质量奖励
3. **BONES 资源分配不合理**：Level 3 性价比最低却给了内容最低的段，CARES 纠正了这一点
4. **level1 性价比高**：level1 延迟短（0.57s vs level2 0.74s），VMAF 增益也不错（6.5 vs 9.1），覆盖大部分段效率高

## 待验证

- [ ] 多轨迹验证（低带宽/高波动/4G/5G）
- [ ] 多视频验证（bbb 之外的视频）
- [ ] QoE 完全持平（<0.5% 损失）的 λ 值
- [ ] 消融实验：去掉内容感知 λ（用全局 λ）
- [ ] 移动设备/边缘设备上的实际价值（电池寿命、设备温度）
- [ ] 与 per-seg 延迟预测的组合效果

## 相关文件

- 配置：`player/ca_cares_config.json`
- 对比脚本：`player/compare_cares.py`
- 资源效率分析：`player/analyze_resource_efficiency.py`
- 分析结果：`workspace/resource_efficiency_analysis.json`
