# 统一帧级质量基线测量

按 `docs/frame_quality_measurement.md` 规范，测量 BBB 根本方法（bicubic、IMDN 三档）的逐帧 VMAF，产出权威数据集 `data/quality/`，替换各线自测口径。

- 建立：2026-10-11，分支 dev/experiments
- 状态：**全量完成（2026-10-26）** → 消费方迁移待办
- 规范依据：`docs/frame_quality_measurement.md`（若缺文件以本 README + 脚本 docstring 为准）

## 测量对象

- 参考：`data/video_mp4box/output_1080p.mp4`（1920×1080@30，19039 帧，158 个完整 4s 段×120 帧，尾段 79 帧）
- 输入各档均 19039 帧（ffprobe 核实）：

| 输入 | 尺寸 | SR scale | 测量臂 |
|---|---|---|---|
| 240p | 426×240 | x4 | bicubic, sr_low, sr_medium, sr_high |
| 360p | 640×360 | x3 | bicubic, sr_low, sr_medium, sr_high |
| 480p | 854×480 | x2 | bicubic, sr_low, sr_medium, sr_high |
| 720p | 1280×720 | — | bicubic（BONES 不提供 720 SR，旧表为 -1） |

合计 13 条臂 × 19039 帧 = 247507 个逐帧分数。

## 方法与权重（锁定）

权重族：**BBB 训练权重**（与旧 `per_frame_quality.json` 及 content_aware 运行模式一致，保证与历史结论连续）。

| 臂 | 模型类 | 权重（`player/istream_player/modules/enhancer/imdn_weight/`） |
|---|---|---|
| sr_low | `IMDN_RTC(scale, num_modules=3, nf=6)` | `IMDN_low_bbb_x{2,3,4}.pth` |
| sr_medium | `IMDN_RTC(scale)`（默认 nf/num_modules） | `IMDN_medium_bbb_x{2,3,4}.pth` |
| sr_high | `IMDN(scale, nf=32)` | `IMDN_high_bbb_x{2,3,4}.pth` |
| bicubic | ffmpeg `scale=1920:1080:flags=bicubic`（YUV 域） | 无 |

模型类定义：`player/istream_player/modules/enhancer/imdn_model.py`；运行时加载映射 `player/imdn_path.json`。

后置处理与运行时一致：模型输出后 `F.interpolate(size=(1080,1920), mode="bicubic", align_corners=False)`（运行时见 `enhancer_imdn.py:345`）。

暂不测：DIV2K 权重族（零样本泛化口径，作为后续可选扩展，同一管道加臂）、ultra 档（BONES 默认决策不使用）、PAC 三出口（另排同管道补测）。

## 管道结构

每输入档：

- LR 生产者：`ffmpeg -i output_<r>p.mp4 -vf format=rgb24 -f rawvideo -`（主线程按帧读取后跑三个模型）；
- bicubic 臂：纯 ffmpeg，`scale=1920:1080:flags=bicubic` 后直接对参考跑 libvmaf（无 stdin）；
- 三个 SR 臂：各一个 ffmpeg（rgb24 rawvideo stdin + 参考第二输入 + 单 libvmaf n_threads=1，逐帧 json），每臂一个 feeder 线程 + maxsize=4 队列，互不阻塞；
- stderr 各自落 `tmp/err_*.log`，rc≠0 即判失败。

不采用 4 臂堆叠单图拓扑：crop+split+4 map 的 filtergraph 在本机死锁（2026-10-11 实测，consumer 空转后全线程 Wait；2 臂堆叠可行，见 `workspace/pacnet/sr_aware_alloc/measure_seg_rd.py:69-145`）。

测量前帧数断言：LR 与所有 vmaf 日志帧数等于目标帧数（冒烟 120 / 全量 19039）。

**帧数锁（2026-10-26）**：双输入 libvmaf 上，输出侧 `-frames:v N` 在本机 `ffmpeg-6.1.1-full_build-shared` 会多打分（冒烟出现 137≠120）。脚本改为两端 `trim=end_frame=N,setpts=PTS-STARTPTS`。验收脚本：`_scratch/_accept_one.py <240|360|480|720>`。

## 工具链（重要）

本机实测可用：`E:\dev\tool\ffmpeg-6.1.1-full_build-shared\bin\ffmpeg.exe`（过滤器列表含 `libvmaf`；须配合上述 trim）。运行显式传 `--ffmpeg <path>` 或环境变量 `BONES_FFMPEG`。

`workspace/ffmpeg_gyan/` 在本机可不存在；若换构建，先 `--smoke --only 240` 核对均值与墙钟。

排障记录：曾出现「libvmaf 慢 20 余倍」假象，根因是只对第一输入限 `-t`、参考流被排空全片；与「`-frames:v` 多打分」是两类问题，勿混用旧修复当唯一手段。

## 输出

`data/quality/bbb_240.json`、`bbb_360.json`、`bbb_480.json`、`bbb_720.json`。脚本：`workspace/quality_baseline/measure_quality_baseline.py`（`--smoke/--only/--ffmpeg`、断点续跑）。

## 资源约束

单 worker；`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=4`；libvmaf 每实例 n_threads=1（4 实例共 4 线程）；开跑前告知。

## 结果（2026-10-26，RTX 5080，SR env）

冒烟（240p 首段 120 帧，墙钟 22.5s）：bicubic 51.479、sr_low 51.892、sr_medium 57.081、sr_high 63.809（与 2026-10-11 冒烟同向）。

全片均值（19039 帧，相对 `output_1080p.mp4`）：

| 输入 | bicubic | sr_low | sr_medium | sr_high | 墙钟 |
|---|---:|---:|---:|---:|---:|
| 240p | 39.296 | 51.736 | 56.079 | 61.493 | 54.4 min |
| 360p | 64.036 | 72.483 | 75.913 | 79.920 | ≈41 min |
| 480p | 76.823 | 83.363 | 86.088 | 87.645 | 44.8 min |
| 720p | 90.132 | — | — | — | 11.4 min |

四档均单调 `bicubic ≤ low ≤ medium ≤ high`；720 仅 bicubic。合计墙钟约 2.5 h。

阶段 2 提示：全片 240p 上 `sr_low − bicubic ≈ +12.4`（冒烟首段几乎打平），专用低档尾巴在 240p 主路径上**不是空档**。

## 阶段 2（2026-10-27）

见 `PHASE2.md` / `phase2_select_cf.json`。**GO**：推荐 `K=5`、`C=3`（high）、`F=1`（low）；5080 上 medium 过快，C=2 需 K≥8–20 不采用。

## 阶段 3（2026-10-27）

见 `PHASE3.md` / `phase3_summary_all.json`：三轨迹×四臂已跑完。**判决 NO-GO**（F 尾不宜作默认；机制可清浪费，稳健默认偏 `whole_F`）。一次性帮手在 `_scratch/`。

## 消费方迁移清单

1. 三控制器（`control/bones_frame_level.py`、`bones_incremental.py`、`bones_combined.py`）新增 `data/quality/` 读取，替换 `data/bbb/per_frame_quality.json`；
2. 对比实验中所有策略（含 BONES 基线）的画质评分统一从权威帧数据现算段值，保证一把尺子；
3. 复跑 `workspace/m5_causal_rank/run_oracle_vs_edf.py` 与 `run_six_trace_regression.py`，确认结论方向（EDF 对 BONES 5/6 胜、oracle 头间约 0.05–0.15）；
4. `data/bbb/per_frame_quality.json` 改名 `.deprecated`，`workspace/vbr_sims/frame_level/README.md` 加替换注记；
5. 更新 `docs/current_state.md` 与本文件，精确 git add 提交（脚本、`data/quality/`、文档；不提交 tmp/log/视频）。

## 不做

- 不改 BONES 基线的 ABR 决策逻辑；
- 不重测已封板工作线的历史结论，仅在新数据上复跑当前主线对比；
- 4K、旁路线不并入本基线。
