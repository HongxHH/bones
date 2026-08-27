## BONES 项目学习指南

本指南面向希望快速理解并上手 BONES（Near-Optimal Neural-Enhanced Video Streaming）项目的读者，包含：项目概览、目录/文件职责、如何运行仿真与原型系统、关键模块关系、以及建议的学习路径。

### 项目概览
- **论文**: BONES: Near-Optimal Neural-Enhanced Video Streaming（SIGMETRICS 2024）
- **核心目标**: 将神经增强（如超分/复原）与自适应码率（ABR）决策联合优化，以提高端到端用户 QoE。
- **两大部分**:
  - **仿真环境**（轻依赖，CPU 可跑）：用于复现论文实验、对比不同 ABR/增强策略、自动调参。
  - **原型系统（播放器）**（GPU/驱动要求高）：端到端播放 + 增强 + 渲染链路，验证系统可行性与性能权衡。

---

## 目录结构总览

- `README.md`: 顶层说明（仿真/系统快速开始、依赖、数据链接）。
- `doc/`: 文档（如 `enhance.md`, `enhance_ch.md`），记录增强数据的采集流程与格式。
- `experiment/`: 论文复现实验脚本与批量运行入口（`run_all.py`, `run.py`）。
- `control/`: ABR 与增强决策算法集合及评测脚本（`bones.py`, `mpc.py`, `pensieve.py`, `nas.py`, `bola.py`, 等）。
- `simulation/`: 仿真器与网络/缓冲/会话模型（`simulator.py`, `network_model.py`, `buffer_model.py`）。
- `config/`: 各类配置项的轻量包装（与 `control/` 对应的简版配置门面）。
- `rl/`: 强化学习相关的结构与训练入口（如 `rl/pensieve/`, `rl/nas/`）。
- `enhance/`: 神经增强模型组织（如 `IMDN/`, `NAS/`），与 `player/` 使用的模型资源相呼应。
- `player/`: 原型系统核心（`play.py` 播放器入口，`istream_player/` 子模块、特征、脚本与数据）。
- `data/`: 用于仿真与播放器的示例数据（`bbb/` 视频元数据与增强质量、`trace/` 网络轨迹）。

---

## 快速上手

### 仿真环境（推荐先学）
依赖：`numpy`

1) 运行 BONES 算法单次仿真（带输出监视/详细日志）
```bash
cd control
python bones.py --monitor --verbose
```

2) 复现论文主要结果（批量）
```bash
cd experiment
python run_all.py
```

3) 数据说明
- 视频元数据与增强性能 JSON 位于 `data/bbb/`。
- 如需复现实验数据采集流程，参见 `doc/enhance.md`（和 `doc/enhance_ch.md`）。

### 原型系统（播放器）
依赖：`asyncio`, `PyTorch`, `VPF` (NVIDIA VideoProcessingFramework), `OpenGL`；需 Linux + CUDA GPU。

1) 下载 DASH 切片
- 从 README 中提供的链接下载并解压，覆盖 `player/data/video_mp4box/`。
- 如需自行切分，进入 `player/data/video_mp4box/` 运行：
```bash
bash segment_1080p_mp4box.sh
```

2) 运行（无头模式，推荐）
```bash
cd player
python play.py -i data/video_mp4box/bbb.mpd -t data/3Glogs/report.2010-09-13_1003CEST.json
```

3) 启用 OpenGL 渲染（图形模式，对 GPU 性能要求较高）
```bash
python play.py -i data/video_mp4box/bbb.mpd -t data/3Glogs/report.2010-09-13_1003CEST.json --mod_renderer opengl
```
可通过调整 `player/istream_player/config/config.py` 的 `display_width`/`display_height` 来降低渲染负载。

---

## 关键模块详解

### simulation/（仿真器）
- `simulator.py`: 仿真核心，驱动会话、网络、缓冲、下载等过程；对接控制策略产生决策并记录指标。
- `network_model.py`: 网络吞吐/时延的抽象模型，可基于 `data/trace/` 重放或统计生成。
- `buffer_model.py`: 播放缓冲动态与卡顿事件建模。
- `session_model.py`: 会话生命周期与媒体段调度。
- `tput_history.py`: 吞吐历史与预测工具。
- `logger.py`: 仿真日志与指标收集。

作用关系（简要）：
- `simulator.py` 周期性调用控制策略（来自 `control/`），基于 `network_model.py` 产生的带宽来决定下一段的码率/分辨率/增强等级，并用 `buffer_model.py` 更新缓冲与 QoE 指标。

### control/（控制与算法）
- 经典/对比算法：`mpc.py`, `bola.py`, `pensieve.py`（及训练脚本），`nas.py` 等。
- BONES 相关：`bones.py`（主算法），`bones_vary.py`（参数/场景变化），`autotune.py` 与若干 `autotune_bones_*.npz`（自动调参与表格）。
- 评测/对比：如 `dynamic.py`, `buffer.py`, `tput.py` 等辅助脚本；`*_test*.txt` 为批量结果。

常见用法：
- 从 `simulation/` 调用控制算法的 `select`/`step`/`decide` 等接口，返回“选取的码率/增强配置”。

### experiment/（复现实验）
- `run_all.py`: 批量运行论文实验（组合不同算法、轨迹、视频配置）；自动化收集结果。
- `run.py`: 更细粒度地指定实验参数的入口。

### player/（原型系统）
- 入口：`play.py`。
- 数据/资源：`player/data/`（含 `video_mp4box/`、网络日志 `3Glogs/`）。
- 核心实现：`istream_player/`（播放器主循环、调度、网络 IO、增强模块桥接、渲染等）。
- 其他：`features/`（可能是特征提取/统计）、`scripts/`（辅助脚本）。

运行路径（简化）：
1) `play.py` 解析参数，加载 MPD/轨迹与配置。
2) 播放器拉取分段 → 根据策略决策码率/增强 → 调用增强模型（参考 `enhance/`）→ 解码/渲染（OpenGL 可选）。

### enhance/（增强模型）
- 子目录：`IMDN/`, `NAS/` 等，组织具体模型与权重路径（与 `player/imdn_path.json` 等配置对应）。
- 质量指标 JSON：`player/imdn_bbb_quality.json`、`player/imdn_div2k_quality.json` 映射质量-开销，用于策略取舍。

### rl/
- `pensieve/`, `nas/`：强化学习相关训练/策略文件夹（与 `control/` 中算法的 RL 版本相呼应）。

### config/
- 轻量封装各类策略配置（名称同 `control/` 脚本），用于集中管理默认参数与开关。

### data/
- `bbb/`: 视频元数据与增强性能数据（JSON）。
- `trace/`: 网络轨迹（3G/4G/WiFi 等），仿真器与播放器均可使用。

### doc/
- `enhance.md`, `enhance_ch.md`: 增强数据采集、质量评估与 JSON 格式说明，便于复现数据。

---

## 典型端到端流程（概念图）
1) 轨迹/网络模型（`data/trace/` 或 `simulation/network_model.py`）给出吞吐与时延。
2) 控制策略（`control/bones.py` 等）根据视频元数据与增强开销/收益（`data/bbb/`、`player/*_quality.json`）选择下个分段的码率与增强级别。
3) 仿真器或播放器执行下载与增强；缓冲更新（`simulation/buffer_model.py` 或播放器内部状态）。
4) 记录 QoE/卡顿/清晰度，供实验脚本汇总（`experiment/run_all.py` 输出）。

---

## 学习路径建议
- **第 1 阶段：理解问题与数据**
  - 阅读 `README.md` 与 `doc/enhance_ch.md`，理解增强质量/开销 JSON 的含义。
  - 浏览 `data/bbb/` 与 `data/trace/`，对输入数据有直观认识。
- **第 2 阶段：跑通仿真**
  - 直接运行 `control/bones.py`；再用 `experiment/run_all.py` 观察不同算法的结果差异。
  - 重点读 `simulation/simulator.py` 与 `control/bones.py` 的交互（决策接口与状态输入）。
- **第 3 阶段：策略细节**
  - 对比 `mpc.py`, `bola.py`, `pensieve.py`, `nas.py` 与 `bones.py` 的差异（代价函数、约束、状态特征）。
  - 看 `autotune.py` 如何生成/使用 `autotune_bones_*.npz`。
- **第 4 阶段：原型系统**
  - 准备 GPU/依赖后运行 `player/play.py`（先无头模式），验证端到端效果。
  - 按需查看 `istream_player/` 的调度与增强调用路径，并结合 `imdn_path.json` 与 `*_quality.json`。

---

## 常见问题与提示
- 图形模式 FPS 低或卡顿：优先使用无头模式，或降低 `display_width`/`display_height`。
- 强依赖缺失（VPF/OpenGL/PyTorch）：请严格按 README 安装版本；使用 CUDA 兼容的驱动与显卡。
- 数据路径不匹配：确保下载的 `video_mp4box/` 覆盖到 `player/data/` 指定路径；检查 JSON 中的相对路径与文件是否存在。
- 可重复研究：建议固定随机种子与轨迹集合，保留 `experiment/` 输出以对比。

---

如需更深入的代码走读建议（按函数/类粒度），告诉我你最关心的模块或文件名，我可以继续为你标注调用图、数据结构与关键函数说明。 