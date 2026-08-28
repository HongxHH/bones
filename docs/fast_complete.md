# fast_complete 机制

## 是什么

在神经增强过程中，根据在线学习（UCB 算法 + 滑动窗口统计）动态调整阈值，**提前终止**当前段的增强，避免错过播放 deadline。

- **代码位置**：`player/istream_player/modules/enhancer/fast_complete_policy.py`
- **集成位置**：`player/istream_player/modules/enhancer/enhancer_imdn_always.py`
- **配置开关**：`use_fast_complete`（`config.py:111`，默认 `True`）

## 为什么是我们的工作点

BONES 原论文**没有**这个机制。BONES 假设增强延迟固定可预测，要么完整增强要么 abort（完全丢弃已增强帧）。

实际系统中神经增强延迟**不确定**（受内容复杂度、GPU 温度、并发影响）：
- 固定阈值太保守→提前终止太多→质量损失大
- 固定阈值太激进→错过 deadline→卡顿

fast_complete 通过 UCB 探索-利用平衡，在线学习最优提前终止帧位置，在**保证 deadline 的前提下最大化增强质量**。

## 技术要点

### UCB 配置（`fast_complete_policy.py:10-19`）

| 参数 | 默认值 | 含义 |
|---|---|---|
| `delta` | 0.2 | 允许错过 deadline 的概率上界（越小越保守） |
| `waste_budget_s` | 0.2 | 基础 waste_budget，允许的最大额外开销时间 |
| `waste_budget_cap_s` | 0.3 | waste_budget 上限 |
| `check_interval_min` | 10 | 检查间隔最小帧数 |
| `check_interval_max` | 30 | 检查间隔最大帧数 |

### 滑动窗口统计

- `SlidingWindowStats`（`fast_complete_policy.py:32`）：在线估计每级别增强延迟的均值/方差
- 窗口大小：30 段
- 缓存机制：避免重复计算统计量

### 与 abort 的区别

| 机制 | 触发条件 | 已增强帧处理 | VMAF |
|---|---|---|---|
| abort | 缓冲区不足，强制中止 | 完全丢弃 | base VMAF（未增强） |
| fast_complete | 预测无法在 deadline 前完成 | 保留已增强帧，未增强帧插值 | 加权平均（见下） |

## 对 VMAF 的影响

fast_complete 段的 VMAF 不是完整增强 VMAF，而是加权平均（`analyzer.py:628-640`）：

```python
ratio = enhanced_cnt / (enhanced_cnt + interpolated_cnt)
real_vmaf = base_vmaf * (1 - ratio) + enhance_vmaf * ratio
quality_drop_due_to_fast = enhance_vmaf - real_vmaf
```

- `ratio` = 已增强帧数 / 总帧数
- `ratio=1` → 完整增强，`real_vmaf = enhance_vmaf`
- `ratio=0` → 完全未增强，`real_vmaf = base_vmaf`
- `0<ratio<1` → 部分增强，加权平均

## 结果文件中的相关字段

| 字段 | 含义 |
|---|---|
| `enhancer_use_fast_complete` | 是否开启 fast_complete |
| `fast_complete_segments_count` | 触发 fast_complete 的段数 |
| `fast_complete_quality_drop_total` | fast_complete 导致的质量下降总和 |
| `fast_complete_quality_drop_avg` | 平均每次 fast_complete 的质量下降 |
| `fast_complete_saved_frames_total` | fast_complete 节省的模型增强帧数 |
| `fast_complete_saved_frames_by_level` | 按增强级别统计节省的帧数 |
| `fast_complete_quality_switches_total` | fast_complete 段内质量切换总次数 |

## 与 CA-BONES 的交互

- CA-BONES 选更高级别→增强时间更长→更可能触发 fast_complete→实际 VMAF 比静态表低
- 这是 CA-BONES predicted 模式 VMAF 低于预期的隐性原因之一
- 所有实验都开启 fast_complete（默认值），所以 BONES 和 CA-BONES 在这一点上是公平对比
