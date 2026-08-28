# 增强管线接口设计与多任务扩展规划

> 技术参考文档，用于方向三（端侧资源感知多任务视频增强）的实现指导。

## 1. 当前架构

### 1.1 核心接口

**Enhancer (ABC)** — `player/istream_player/core/enhancer.py`
- `start(adaptation_sets)`: 加载模型、测量延迟
- `is_ready()`: 就绪检查
- `get_latency_table(segment_index=None)`: 返回 (5,5) 延迟表（download_action × enhance_level），单位秒
- `get_quality_table()`: 返回 VMAF 质量表 (5,5)
- `remain_task()`: 当前增强剩余时间
- `cleanup()`: 清理

**NESController (ABC)** — `player/istream_player/core/nes.py`
- `update_selection(adaptation_sets, index)`: 返回 (download_actions, enhance_actions)
- enhance_actions 是整数 0-4（增强级别）

### 1.2 数据流

```
NES决策(download_action, enhance_action)
  → 下载到 download_buffer
  → 移入 enhance_buffer（携带 enhance_action）
  → Enhancer dequeue
  → 逐帧解码(DecoderNvCodec) → 增强(IMDN model) → 存储(segment.decode_data)
  → replace 回 download_buffer
  → Renderer 播放
```

### 1.3 IMDNEnhancerAlways 关键组件

- `model_pool`: `(scale, level) → torch.nn.Module`，仅 IMDN 超分模型
- `latency_table`: (5,5) numpy，EWMA 实时更新 + 内容感知 KNN 预测
- `quality_table`: (5,5) VMAF，从 JSON 加载
- `enhance_one_frame(surf, model, tensor_converter)`: Surface→Tensor→模型推理→bicubic resize→CPU Tensor
- `fast_complete`: 段即将播放时，剩余帧用解码器直接输出（不增强）
- `segment_stats`: 每段统计信息，供 Analyzer 聚合

### 1.4 NES 与 Enhancer 交互点

`bones.py:126`: `self.enhance_time = self.enhancer.get_latency_table() * 1000 / self.enhance_safety`
`bones.py:140`: `self.vmaf_enh_avg = self.enhancer.get_quality_table()`

调度器只通过这两个方法获取增强器信息，不直接访问模型或缓冲区。

---

## 2. 多任务扩展的核心挑战

| 维度 | 当前 | 多任务需求 |
|------|------|-----------|
| enhance_action | 单整数 0-4（级别） | 任务组合（如 SR+Deblur+FI） |
| latency_table | (5,5) download×level | 需扩展任务组合维度 |
| quality_table | VMAF (5,5) | 多维度质量（VMAF/FPS/清晰度） |
| model_pool | (scale,level)→IMDN | 多模型（SR/Deblur/FI） |
| 增强管线 | decode→IMDN→resize | 顺序任务执行（SR→Deblur→FI） |
| 资源感知 | 无 | GPU利用率/显存/功耗作为状态 |

---

## 3.  proposed 接口设计（阶段1目标）

### 3.1 增强任务类型定义

```python
class EnhanceTaskType(Enum):
    SR = 0        # 超分辨率
    DEBLUR = 1    # 去模糊
    FI = 2        # 插帧
    DENOISE = 3   # 去噪
    NONE = 4      # 不增强
```

### 3.2 增强动作表示

用位掩码或元组表示任务组合：
```python
# 方案A：位掩码（简单，最多8种任务）
enhance_action = (SR | DEBLUR)  # 同时做超分和去模糊

# 方案B：元组（灵活，可带参数）
enhance_action = (("sr", level=2), ("deblur", level=1))
```

**推荐方案A**（位掩码）用于调度器决策空间，简单且可枚举；方案B用于增强器内部执行。

### 3.3 EnhancerBase 扩展接口

```python
class MultiTaskEnhancer(Enhancer):
    # 新增方法
    def get_task_latency(self, task_mask: int, resolution: tuple, segment_index=None) -> float:
        """获取指定任务组合在指定分辨率下的延迟（秒）"""
        pass

    def get_task_quality(self, task_mask: int, base_quality: dict) -> dict:
        """获取指定任务组合的质量增益（多维度）"""
        pass

    def get_gpu_state(self) -> dict:
        """获取当前 GPU 状态：utilization, mem_used, mem_total, power, temp"""
        pass

    def supported_tasks(self) -> list:
        """返回支持的增强任务列表"""
        pass

    def get_model_memory(self, task_mask: int) -> float:
        """获取指定任务组合所需的 GPU 显存（MB）"""
        pass
```

### 3.4 延迟表扩展

从 (5,5) 扩展为 (5, num_task_combinations)：
- download_action (5) × task_mask (e.g., 8 = 2^3 for SR/Deblur/FI)
- 每个组合的延迟 = 各任务延迟之和（顺序执行）或最大值（并行执行）
- 实测时用 EWMA 更新，未访问组合用各任务延迟之和推算

### 3.5 质量表扩展

从单维 VMAF 扩展为多维度：
```python
quality = {
    "vmaf": 85.0,           # 超分带来的 VMAF
    "fps": 60,              # 插帧带来的帧率
    "sharpness": 0.85,      # 去模糊带来的清晰度
    "noise_level": 0.1,     # 去噪后的噪声水平
}
```

调度器需要将多维度质量聚合为标量奖励（自定义 QoE 指标，见方向三点5）。

---

## 4. 迁移路径

### 阶段0（当前）：基础建设
- 任务1：接口抽象（本文档）
- 任务2：引入轻量视频SR模型，建立多模型延迟表
- 任务3：引入去模糊+插帧模型，建立多任务延迟表
- 任务4：GPU资源监控完善
- 任务5：文献调研

### 阶段1：多任务调度框架
- 扩展 enhance_action 为任务组合
- 扩展延迟表和质量表
- 实现 MultiTaskEnhancer 包装器（兼容现有 IMDNEnhancerAlways）
- 实现资源感知调度算法（基于 Lyapunov 改造或启发式）
- 仿真器验证

### 阶段2：自定义指标与真实播放器验证
- 设计多维度 QoE 指标
- 真实播放器集成
- 多轨迹对比实验
- 消融实验

### 阶段3：深化
- 模型层面优化（动态计算分配、多任务共享骨干）
- 理论分析
- 论文撰写

---

## 5. 关键代码位置索引

| 组件 | 文件 | 关键行/方法 |
|------|------|------------|
| Enhancer 抽象接口 | `core/enhancer.py` | 全文 |
| NESController 抽象接口 | `core/nes.py` | 全文 |
| IMDN 增强器实现 | `modules/enhancer/enhancer_imdn_always.py` | `enhance_one_frame`(612), `get_latency_table`(1016), `get_quality_table`(1048), `load_model`(814), `measure_latency`(874) |
| BONES 调度器 | `modules/nes/bones.py` | `bones_decision`(109), `_bones_control`(151) |
| CA-BONES 调度器（真实播放器） | `modules/nes/ca_bones.py` | 全文 |
| CA-BONES 调度器（仿真器） | `control/bones_ca.py` | `_lookahead_control`(338) |
| GPU 监控 | `utils/gpu_monitor.py` | 待确认 |
| 内容特征 | `utils/content_features.py` | 待确认 |
| 增强计划生成器 | `modules/enhancer/enhancement_plan.py` | 全文 |
| IMDN 模型定义 | `modules/enhancer/imdn_model.py` | IMDN, IMDN_RTC |

---

## 6. 注意事项

1. **向后兼容**：MultiTaskEnhancer 应包装现有 IMDNEnhancerAlways，不破坏现有实验流程
2. **延迟表单位**：当前 get_latency_table 返回秒，调度器乘以 1000 转毫秒。扩展时保持一致
3. **fast_complete**：多任务场景下更复杂，各任务的完成度需要分别追踪
4. **显存管理**：多模型同时驻留显存可能 OOM，需要模型加载/卸载策略
5. **任务顺序**：SR→Deblur→FI 有依赖关系（FI 需要连续帧，SR 改变分辨率），不能任意并行
6. **内容感知**：不同内容适合不同任务组合，需与现有内容特征基础设施结合
