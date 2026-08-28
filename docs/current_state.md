# 当前状态

## 多任务增强（Multi-Task Enhancement）阶段（2026-08-28）

### 已完成

| 组件 | 文件 | 状态 |
|------|------|------|
| MultiTaskEnhancer | `player/istream_player/modules/enhancer/enhancer_multitask.py` | ✅ 支持sr/sr_deblur/deblur三种组合，1080p deblur-only |
| MTBONESController | `player/istream_player/modules/nes/mt_bones.py` | ✅ 资源感知GPU状态追踪+内容感知任务选择 |
| 多任务数据 | `data/multitask_bbb_rtx5080.json` | ✅ 5种组合的延迟+VMAF表 |
| 模块注册 | `module_composer.py` | ✅ `--mod_enhancer multitask --mod_nes mt_bones` |
| 1080p非超分增强 | enhancer_multitask.py | ✅ scale=1表示允许deblur-only，SR组合被标记为无效 |

### 任务组合定义

| combo | 名称 | 管线 | 适用分辨率 |
|-------|------|------|-----------|
| 0 | none | 直接播放 | 所有 |
| 1 | sr | NanoVSR/IMDN超分 | 240p/360p/480p |
| 2 | sr_deblur | NAFNet去模糊(输入分辨率) → SR | 240p/360p/480p |
| 3 | deblur | NAFNet去模糊 → 双三次放大 | 所有(含1080p) |
| 4 | sr_deblur_fi | 去模糊 → SR → 插帧 | ⚠️ FI未实现，降级为sr_deblur |

### 关键技术决策

1. **去模糊在输入分辨率运行**：NAFNet在240p下~16ms/帧，在1080p下~31ms/帧。先去模糊再超分，避免在高分辨率上做去模糊。
2. **1080p deblur-only**：突破BONES"最高码率不增强"的限制，1080p仍可做去模糊。SR组合对1080p标记为无效延迟(1e6)，控制器自然回避。
3. **hd_fs轨迹带宽极高**：平均15.7Mbps（范围1.2-28Mbps），足够直接下载1080p，控制器选1080p+none。多任务增强的价值在中等带宽(1-5Mbps)轨迹中更明显。

---

## 历史阶段成果

### CARES + VMAF 平滑组合（stable3g轨迹，RTX 3060Ti）

| 指标 | BONES | CARES+VMAF平滑 | 变化 |
|------|-------|----------------|------|
| avg_vmaf | 66.47 | 67.80 | +2.0% |
| QOE | 57.96 | 63.97 | +10.4% |
| oscillation | 8.50 | 3.79 | -55.4% |
| GPU能耗 | 31.34kJ | 25.48kJ | -18.7% |

### 前瞻规划N=2（仿真器，6轨迹）

6轨迹中4胜2负。关键修复：dl_time单位bug（bytes误当Kb，下载时间高估125倍）。

---

## 当前在做什么

- **多任务增强系统集成**：已完成MultiTaskEnhancer + MTBONESController的播放器集成，冒烟测试通过
- **1080p非超分增强**：已实现，待验证deblur对1080p的VMAF增益
- **工程清理**：删除IMDN训练数据集(36.8GB)、旧实验输出、版本化脚本

---

## 下一步计划

### 短期
1. **验证1080p deblur效果**：测量deblur对1080p的VMAF增益，更新质量表
2. **中等带宽轨迹实验**：用1-5Mbps轨迹验证sr_deblur组合的QoE收益
3. **FI（插帧）实现**：下载RIFE模型，实现combo 4

### 中期
1. **自定义评价指标**：多维度质量分(VMAF+时序平滑度+清晰度) + 资源感知QoE
2. **多轨迹对比实验**：BONES vs CA-BONES vs MT-BONES，多轨迹+消融
3. **内容感知去模糊**：根据运动水平自适应决定是否去模糊

---

## 关键发现

1. **增强的价值在中等带宽**：高带宽(>10Mbps)下直接下1080p，增强无用；低带宽(<1Mbps)下增强导致卡顿；中等带宽(1-5Mbps)是"用缓冲换质量"的甜区
2. **去模糊在输入分辨率做更高效**：240p去模糊16ms vs 1080p去模糊31ms，先去模糊再超分的管线更优
3. **BONES的"最高码率不增强"假设已过时**：1080p在4K时代只是及格线，仍可通过去模糊等非超分增强提升感知质量
4. **VMAF质量表缺少去模糊增益数据**：当前控制器不知道deblur能提升质量，倾向于选none。需要测量deblur的VMAF增益
