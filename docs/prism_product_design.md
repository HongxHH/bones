Prism 产品与实验平台设计
========================

> 状态：**L0–L3 已验收**；产品树为 **`prism/{cli,control,desk}`**；入口 **`python -m prism`**（桌面窗 Lab + 画面）。




> 产品名：**Prism**（棱镜——把一路流拆成可对比的实验臂，便于观测）  
> 说明：仓库目录可暂仍叫历史名；**对外产品名、CLI、UI 文案、API 产品字段一律用 Prism**，不出现他人品牌名。

Context and motivation
----------------------

当前已有「重服务端造流 / 本机 player 增强出画 / 浏览器控制台操控」技术骨架，但使用上仍是多进程拼装。研究真正需要的不是通用播控，而是**可复现、可对比、可观测的实验环**。

纯浏览器无法拉起本机 GPU 增强；UI 焊回 OpenCV 又难扩展。因此：

**网页是实验面板，播放与增强是本机仪器；可选桌面壳只负责一键编排。**

Goals
-----

- **一体入口**：一次命令进入 Prism 工作台
- **实验一等公民**：配置 → 跑 → 观测 → 对比 → 归档
- **观测闭环**：VMAF 等质量指标 + 客户端运行时指标汇入同一 Experiment/Run
- **配置可声明**：YAML 配方；UI 与 CLI 同 schema
- **UI 分层**：Lab Web = 面板；player = 仪器；桌面窗 = Lab + 画面一体壳

Non-goals（近期不做）
---------------------

- 浏览器内解码 / WebGPU 重做增强
- 公网多租户与账号体系
- 商用 ABR 控制台、素材 CMS
- 壳内重写编排器或 ffmpeg CLI
- 第一期控制面必做 Seek / 精细锁码率

已锁定的产品决策（按推荐方向，无需再选）
----------------------------------------

| 议题 | 决定 |
|------|------|
| 默认出画 | **单臂调试默认 OpenCV**；**批跑 / CLI 默认 headless** |
| 多臂 GPU | **默认串行**（播完臂 A 再臂 B）；UI 提供「强制并行」高级选项 |
| Lab 挂载 | **挂在控制面** `http://127.0.0.1:8080/lab`，不另开端口 |
| 桌面窗 | **`prism/desk/`（PySide6）**：左 Lab、右画面；`python -m prism` 默认入口 |
| 默认素材 Preset | **BBB 已有 DASH 树**作为「导入即播」唯一默认素材 |
| 壳技术 | 桌面壳用 PySide6 WebEngine 嵌 Lab；不再维护 Electron |
| 命名 | 产品 **Prism**；模块/CLI `prism`；配置 `apiVersion: prism.lab/v1` |

Implementation considerations
-----------------------------

### 设计原则

1. **仪器与面板分离**：解码/调度/GPU 增强在 player；配置/对比/归档在 Prism UI + server  
2. **声明式实验优先**：UI 是配方编辑器与监视器；同一 YAML 可 CLI 重跑  
3. **指标有归属**：挂 `experiment_id` + `run_id` + `arm`（+ 可选 `session_id`）  
4. **插件边界不变**：新能力进 Recipe/Packager/Delivery/Measurer/Hook  
5. **壳可替换**：Lab 仍是 Web；桌面窗只换宿主（当前 PySide6）

### 约束

- Windows 为主；Python player + GPU 环境已有  
- Origin/Session 已支持动态 MPD 与 `policy_version`  
- 本地能力经 StatusExporter / Prism 启动器 / 可选壳，不经纯网页沙箱直调进程  

Product form（产品形态）
-----------------------

### 一句话

**Prism 是本地神经增强流媒体实验工作台**：控制面造流与归档，player 当测量仪器出画，Web 当实验面板，桌面窗把面板与画面放在同一窗口。

### 三角色

| 角色 | 职责 | 形态 |
|------|------|------|
| **Control Plane** | Asset/Job/Publication/Session/Experiment；插件；Origin；档案 | 常驻 HTTP（默认 `:8080`） |
| **Instrument** | 拉流、调度、增强、出画；导出状态与 telemetry | 本机 player 进程 |
| **Prism Shell** | 起停控制面、嵌 Lab、右侧画面；不实现业务编排 | `python -m prism` → `prism/desk` |

### 用户怎么感知「一个产品」

1. 运行 `python -m prism`  
2. 同一窗口左侧是工作台 `/lab`，右侧是完整分辨率画面  
3. 在工作台里：选内容 → 配实验 → 开跑 → 看曲线；开始后不会再弹出播放器窗口  
4. 暂停或停止后，右侧停在最后一帧  

**增强仍在本机播放器里完成。** 画面进入窗口右侧。命令行单独跑播放器时，仍可走原来的窗口或 GLUT 路径。

### 结构图

```
┌─────────────────────────────────────────────────────────┐
│  Prism Shell (python -m prism → desk/)                  │
│   · 确保 control plane :8080                             │
│   · 左 Lab / 右画面                                      │
│   · 按需 spawn player（--mod_status，独立 status_port）  │
└─────────────┬───────────────────────────┬───────────────┘
              │                           │
              ▼                           ▼
     ┌────────────────┐          ┌────────────────────┐
     │ server :8080   │◄────────│ Prism Lab UI        │
     │ Origin / Jobs  │          │ /lab 实验面板       │
     └────────┬───────┘          └─────────┬──────────┘
              │ DASH                       │ 控制 / telemetry
              ▼                            ▼
         ┌────────────────────────────────────┐
         │ player（出画）+ StatusExporter     │
         └────────────────────────────────────┘
```

Experiment model（实验模型）
---------------------------

| 对象 | 含义 |
|------|------|
| **Experiment** | 研究问题容器：名称、假设、共享素材、备注 |
| **Spec** | 声明式配方（YAML） |
| **Run** | 一次执行：`run_id`、时间、fingerprint、状态 |
| **Arm** | 对比臂（如 `cbr` / `vbr_nes`） |
| **Session** | 臂的分发会话（MPD policy） |
| **PlayerAttach** | 附着的 player（pid、status_url） |
| **MetricsBundle** | 该 Run/Arm 的指标与导出路径 |

### 声明式配方示例（`prism/experiments/*.yaml`）

```yaml
apiVersion: prism.lab/v1
kind: ExperimentSpec
metadata:
  name: bbb-play
  notes: "导入 BBB 并播放"
spec:
  content:
    source: publication:pub_xxx
  arms:
    - id: play
      publish:
        plugin: identity
      session:
        policy:
          allowed_representation_ids: [0, 1, 2]
          mpd_type: static
      player:
        mod_nes: always_greedy
        mod_renderer: opencv
  measure:
    - plugin: vmaf
      params:
        reference: ${content.reference}
        distorted_glob: "**/*720*.mp4"
  collect:
    client_metrics: true
    interval_ms: 500
    export: [json, csv]
  execution:
    arm_mode: serial   # 默认；parallel 为高级选项
```

### 黄金路径

1. 导入 DASH（默认 BBB Preset）→ Publication  
2. 创建 Experiment（`identity` 导入已有 DASH）  
3. Start Run → publish → 建 session → 拉起 player → 采集 telemetry  
4. 观测曲线 / 可选改 policy 看 `policy_version`  
5. Stop → 可选 measure → archive  
6. 查看摘要；CSV/JSON 导出  

CLI（与 UI 对等）：

```text
python -m prism                  # 启动工作台
python -m prism run <spec.yaml>
python -m prism status <run_id>
python -m prism export <run_id> --format csv
```

验收（控制面已起时）：

```text
python prism/scripts/accept_prism.py
```

示例配方：`prism/experiments/bbb_play.yaml`。

Observability（观测）
--------------------

| 层 | 来源 | 示例 |
|----|------|------|
| L0 运行时 | StatusExporter → telemetry API | buffer、bandwidth、码率、enhance_action、卡顿事件 |
| L1 会话 | Origin logs + policy 时间线 | 动态 MPD 是否生效 |
| L2 质量 | Measurer（VMAF 等） | vmaf_mean |
| L3 摘要 | archive | fingerprint、臂对比表、导出文件 |

**回流**：player 周期性上报 → `experiments/{id}/runs/{run_id}/telemetry/{arm}.jsonl`  
Lab Run 页近实时曲线；研究规模不引入重型时序库。

Configuration（配置）
--------------------

| 层 | 内容 |
|----|------|
| 机器环境 | `PRISM_FFMPEG`、`PRISM_REDIS_URL`、端口（实现期同步兼容旧环境变量，避免迁徙剧痛） |
| 平台默认 | ladder、segment、worker labels → ServerConfig + 可选 `prism.yaml` |
| 实验配方 | arms / plugins / policy / player → YAML + API |
| 运行覆盖 | headless、单臂、强制并行 → CLI / UI Advanced |

- 插件登记 `params_schema`，UI 按 schema 出表单  
- 表单 ↔ YAML 双轨；Preset：「BBB 导入即播」

UI 形态
-------

### 信息架构

```
Prism Lab (/lab)
├── Home         最近 Experiment / Run；继续上次
├── Library      Publications / Assets；BBB 一键导入
├── Experiments  列表 → 详情 → Run 工作台
├── Jobs         转码 / 测量队列
└── System       健康、worker、路径、日志
```

### Run 工作台（主界面）

- 顶栏：名称 · run_id · 状态 · Start/Stop  
- 左：Arms（附着仪器、policy 快捷项；默认串行进度）  
- 中：buffer / bitrate / enhance 曲线 + 卡顿与 policy 事件  
- 右：Jobs + VMAF 等卡片  
- 底：日志抽屉  

### 与现有 player 控制台

- `player/console` → **仪器本地调试页**（单进程 pause/resume）  
- 实验主路径一律走 `/lab`  
- switch_source / launch 收束为「附着仪器」

### 桌面窗（`prism/desk/`）

`python -m prism`（默认 `desk`）打开 PySide6 窗口：左嵌 `/lab/?shell=1`，右接 `prism.frame_sink` 画面。  
player 仍由 Lab attach API spawn/kill。Quit 只结束本窗自己拉起的控制面。

Error handling and UX
---------------------

| 场景 | 表现 | 恢复 |
|------|------|------|
| server 未起 | 启动器拉起；失败则 System 红条 | 重试 |
| player/GPU 失败 | 该 Arm 报错，其它臂继续 | 改模块后重附着 |
| telemetry 断 | 曲线暂停、点变灰 | 重连或重启 player |
| 无 libvmaf | measure `skipped` +原因 | 安装或改 noop |
| 配方非法 | Start 前阻断 | 按字段提示 |

分阶段落地（实现顺序）
----------------------

### L0 — 入口与面板骨架

- `python -m prism`：health、拉起 server、打开 `/lab`  
- `/lab`：Library / Jobs / System / Experiments 列表（接现有 API）  
- 文案与窗口标题使用 Prism  
- Preset：BBB 导入即播入口按钮  

### L1 — Run + 配置

- ExperimentSpec（Pydantic）+ YAML  
- `POST .../runs` 展开 Arms → Session  
- UI：编辑 arm、Start/Stop；默认 `arm_mode: serial`  
- 调试默认 opencv，CLI run 默认 headless  

### L2 — 观测闭环

- telemetry API + jsonl  
- StatusExporter 可选上报（run/arm/session）  
- Run 页曲线；Stop 后摘要进 archive  

### L3 — 记录与导出

- 单次实验的 CSV  
- `prism run|status|export`  
- 常用插件 `params_schema` 表单  

### L4 — 桌面窗（`prism/desk/`，替代原 Electron 方案）

- `prism/desk/`：Lab + 画面一体；控制面未起时拉起  
- `python -m prism` 默认打开该窗；`python -m prism lab` 仍可走系统浏览器  

Testing / Acceptance（摘要）
----------------------------

- **L0**：未起 server 时 `python -m prism` 能打开 `/lab` 并列出 publications  
- **L1**：合法 YAML Start Run → 有 run_id 与各臂 mpd_url  
- **L2**：播放 ≥30s → telemetry 有数据且曲线更新  
- **L3**：单次实验可导出 CSV  
- **L4**：`python -m prism` 打开桌面窗；热启动 `--no-server` 不杀已有控制面  

你怎么参与（本轮约定）
----------------------

- 实现前**不**再向你抛选择题；推荐项已写入上文「已锁定的产品决策」。  
- 你的输入是：**看产品形态 → 给见解**；我根据见解改设计文档，你确认后再开工。  
- 当前请重点看：三角色是否顺、Run/Arm 是否好用、UI 信息架构是否够用、命名 Prism 是否接受。  

变更记录
--------

- 2026-04-12：首版设计；产品名定为 Prism；锁定出画/串行/挂载/Electron 节奏/BBB Preset。
- 2026-04-12：L0 落地 — `prism/` 启动器、`server/lab/`、`GET /v1/system`、根路径重定向 `/lab/`。
- 2026-04-13：L1 落地 — `ExperimentSpec`、`/v1/runs*`、Lab Run 面板、`python -m prism run <spec>`；配方现位于 `prism/experiments/`。
- 2026-04-13：L2 落地 — `POST /v1/telemetry`、jsonl 落盘、Run metrics、StatusExporter 上报、Lab 曲线。
- 2026-04-13：L3 落地 — `/v1/compare`、CSV export、Lab Compare 页、`prism status|export|stop`。
- 2026-04-13：Review 修复 — attach 闸门 / preparing 异步 / Stop→measure / prism agent / telemetry 重试 / Redis claim / Compare delta / YAML 解析。
- 2026-10-13：验收闭环 — `accept_prism.py`；Run 按条发布已有 DASH。
- 2026-10-20：工具脚本迁入 `prism/scripts/`；根目录 `scripts/`、`shell/` 已删除。
- 2026-10-13：L4 — 曾落地 Electron `desktop/`；后由 desk（PySide6）替代，`desktop/` 已删除。
- 2026-10-13：Lab 收成素材 / 实验 / 对比；实验页预览播放器 JPEG（`POST /v1/preview`）。
- 2026-10-19：桌面窗口右侧出完整分辨率画面。开始播放不再另开窗口。
- 2026-10-20：产品树收拢为 `prism/{cli,control,desk,experiments,scripts,var}`；根目录 `output/`、`server/` 已删；控制面入口为 `python -m prism serve`。
- 2026-10-19：逐帧索引与 JPEG 画面存档。`GET /v1/runs/{id}/frames` 按臂和帧号读取。
- 2026-10-19：实验页时间轴。拖动某一帧时，右侧画面显示该帧和当时的信息。拖动不会启动播放器。
- 2026-10-19：右侧画面可 1:1 查看。`python -m prism` 打开这个窗口。
- 2026-10-19：一次实验只保留一条条件。左右对比、实验对比页和「下一臂」已去掉。



  
