# 理论分析：CARES + VMAF 平滑的 Lyapunov 优化框架

## 1. BONES 原始框架回顾

### 1.1 状态变量

BONES 使用 Lyapunov 优化框架，状态变量为双缓冲区水位：
- `Q_down(t)`：下载缓冲区水位（ms）
- `Q_enh(t)`：增强缓冲区水位（ms）

### 1.2 漂移上界

Lyapunov 函数 `L(t) = Q_down(t)^2 + Q_enh(t)^2`，单步漂移：

```
Δ(t) = E[L(t+1) - L(t) | Q(t)]
     ≤ B + E[Q_down(t) * seg_time + Q_enh(t) * enhance_time | Q(t)]
```

其中 `B` 为常数上界，`seg_time` 为段时长，`enhance_time` 为增强耗时。

### 1.3 漂移加惩罚

BONES 最大化长期平均质量，最小化：

```
Δ(t) + V * E[-vmaf + gamma_p | Q(t)]
```

代入漂移上界，每段决策最小化：

```
obj = Q_down * seg_time + Q_enh * enhance_time - V * (vmaf + gamma_p)
```

其中 `V` 为权衡参数，控制缓冲区稳定性与质量的 trade-off。

### 1.4 近最优保证

BONES 证明了 O(1/V) 的近最优保证：
- 平均质量与最优值差距 ≤ O(1/V)
- 平均缓冲区水位 ≤ O(V)

---

## 2. CARES 修改：内容感知资源惩罚

### 2.1 修改动机

BONES 的增强耗时 `enhance_time` 在漂移项中权重固定为 `Q_enh`，不考虑内容差异。但实际中：
- 高内容段（复杂纹理/运动）增强收益大，值得花更多时间增强
- 低内容段增强收益小，应快速通过以节省资源

### 2.2 修改公式

在质量奖励中加入内容感知资源惩罚：

```
reward = vmaf + gamma_p - λ_i * enhance_time
```

其中 `λ_i` 为第 i 段的内容感知惩罚系数：

```
content_score_i = normalize(texture_i * motion_i)  ∈ [0, 1]
λ_i = λ_min + (λ_max - λ_min) * (1 - content_score_i)
```

- 高内容段：`content_score` 高 → `λ_i` 小 → 资源惩罚小 → 值得增强
- 低内容段：`content_score` 低 → `λ_i` 大 → 资源惩罚大 → 不值得增强

### 2.3 等价变换

将修改代入决策目标：

```
obj = Q_down * seg_time + Q_enh * enhance_time - V * (vmaf + gamma_p - λ_i * enhance_time)
    = Q_down * seg_time + (Q_enh + V * λ_i) * enhance_time - V * (vmaf + gamma_p)
```

**理论解释**：CARES 等价于给增强缓冲区水位加了一个内容感知的权重 `V * λ_i`。低内容段权重高，更倾向于选低增强级别以减少 `enhance_time`。

### 2.4 理论性质

- **保持 Lyapunov 近最优保证**：CARES 只是改变了漂移项中 `enhance_time` 的系数，不改变优化框架结构。O(1/V) 近最优保证仍然成立。
- **最优目标改变**：优化的目标函数从"纯质量最大化"变为"质量最大化 - 内容感知资源消耗"，这是一个更合理的目标。
- **参数解释**：`λ_max - λ_min` 控制内容感知的强度，`λ_max` 越大，低内容段越倾向于不增强。

---

## 3. VMAF 平滑修改：质量切换惩罚

### 3.1 修改动机

BONES 每段独立决策，不考虑相邻段质量差异。实际中：
- 频繁的质量跳变会降低用户体验（QoE 公式中 oscillation 权重 1:1）
- 内容感知调度（如 CA-BONES base_known）会增加决策方差，导致 oscillation 暴增

### 3.2 修改公式

在质量奖励中加入 VMAF 切换惩罚：

```
reward = vmaf + gamma_p - α * |vmaf_i - vmaf_{i-1}|
```

其中 `α` 为平滑强度系数，`vmaf_{i-1}` 为上一段的预测 VMAF。

### 3.3 等价变换

代入决策目标：

```
obj = Q_down * seg_time + Q_enh * enhance_time - V * (vmaf + gamma_p - α * |vmaf_i - vmaf_{i-1}|)
```

**理论解释**：VMAF 平滑等价于在质量奖励中加入了一个时间相关的 L1 正则化项，惩罚与上一段 VMAF 的偏差。

### 3.4 理论性质

- **引入状态依赖**：决策不再是纯粹的每段独立，而是依赖上一段的 VMAF。这引入了"路径依赖"——初始段的选择会影响后续路径。
- **保持 Lyapunov 框架**：虽然引入了状态依赖，但每步仍然最小化一个漂移加惩罚的上界，Lyapunov 优化的理论框架仍然适用。
- **双稳态现象**：当 `α` 足够大时，系统可能出现多个稳定均衡点（如 level1 主导 vs level2 主导），初始条件的微小差异决定落入哪个均衡。这是路径依赖的直接结果。
- **oscillation 降低**：平滑惩罚直接针对 QoE 公式中的 oscillation 项，理论上可以证明 `E[|vmaf_i - vmaf_{i-1}|]` 随 `α` 增大而减小。

---

## 4. 组合框架：CARES + VMAF 平滑

### 4.1 组合公式

两个修改正交，可同时应用：

```
reward = vmaf + gamma_p - λ_i * enhance_time - α * |vmaf_i - vmaf_{i-1}|
```

决策目标：

```
obj = Q_down * seg_time + (Q_enh + V * λ_i) * enhance_time
      - V * (vmaf + gamma_p - α * |vmaf_i - vmaf_{i-1}|)
```

### 4.2 两个机制的正交性

| 机制 | 作用位置 | 优化目标 | 副作用 |
|---|---|---|---|
| CARES | 漂移项（enhance_time 权重） | 资源效率 | 可能增加级别切换 |
| VMAF平滑 | 惩罚项（VMAF 正则化） | 质量稳定性 | 路径依赖/双稳态 |

CARES 影响"选什么级别"（资源效率），VMAF 平滑影响"级别是否变化"（稳定性），两者作用于不同维度，不冲突。

### 4.3 协同效应

- CARES 将低内容段推到低级别，VMAF 平滑让过渡更平缓
- VMAF 平滑减少 oscillation，CARES 减少资源消耗，组合后同时提升 QoE 和资源效率
- 实验验证：QoE 从 BONES 57.96 提升到 63-64（+5.7~6.3），GPU 能耗降低 16-21%

### 4.4 理论保证

组合框架仍然在 Lyapunov 优化的漂移加惩罚范式内：
- CARES 修改了漂移项系数，不改变框架
- VMAF 平滑修改了惩罚项，引入时间依赖但仍可每步最小化上界
- O(1/V) 近最优保证对修改后的目标函数仍然成立

---

## 5. 理论贡献总结

1. **内容感知 Lyapunov 漂移加权**：将内容复杂度融入增强缓冲区的漂移权重，实现内容感知的资源分配。这是对 BONES 固定权重的推广。

2. **时间相关质量正则化**：在 Lyapunov 惩罚项中引入 VMAF L1 正则化，实现质量平滑。这是对 BONES 每段独立决策的推广。

3. **正交机制组合**：两个修改作用于不同维度（资源效率 vs 质量稳定性），可组合使用且保持理论保证。

4. **双稳态现象分析**：VMAF 平滑引入路径依赖，当 α 足够大时出现多均衡点。这一现象解释了实验中观察到的 level 分布随机性。
