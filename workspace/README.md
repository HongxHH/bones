# workspace 索引

实验工作区。**复盘/上手顺序：先在本页定位目录 → 读该目录 README（十一节档案，含复核记录与复活清单）→ 需要叙事再去 `docs/` 对应文档。** 本页只做索引和归属，不重复档案内容。

档案约定：每条独立工作线一份目录 README（要回答的问题/方法/文件清单/复现/结论数字/复核记录/影响面/是否重做/踩坑/上下游/复活清单）；结论数字以档案 README 与结果 json 为权威，docs 叙事文档可能滞后。

## 主线：PAC-Net 与后续探针（`pacnet/`）

| 目录 | 主题 | 状态 | 目录档案 |
|---|---|---|---|
| `pacnet/`（根） | PAC-Net 共享骨干三出口 anytime 网：离线 verdict、stage1-4 训练、halt 控制器、理论验证、TOS 满血三轨迹 | 已封板：K=1.5 下浅出口被 IMDN_RTC 严格 dominate | `pacnet/README.md`（§6 含 F1-F6 复核） |
| `pacnet/scripts/` | PAC-Net 训练/测量/评估脚本（原散在根目录） | 代码留存 | 归 `pacnet/README.md` |
| `pacnet/weights/` | PAC-Net 定型权重（stage2_mix_nocond 等） | 留存（*.pth 不入库，已跟踪者除外） | 归 `pacnet/README.md` |
| `pacnet/multi_k/` `sweep_thresh/` `multi_trace_logs/` | K=1.5 gpuburn 多轨迹、安全系数 1.0-1.3 扫描 | 数据留存：halt vs force_e3 三轨迹全胜，vs IMDN 卡顿 0/3 劣 | 归 `pacnet/README.md` |
| `pacnet/hard_survey/` | 7 档退化×7 方法普查 + LPIPS/PSNR 复验 | 已封板：VMAF 绝对分口径作废、LPIPS 逐帧证据有效 | `hard_survey/README.md` |
| `pacnet/visual_compare/` | 增强前后四列对比 PNG（4 张） | 定性证据，不可引数值 | 归 hard_survey 档案 §3.2 |
| `pacnet/h1_condnet/`（含 `h1_codecmeta/`） | H1 codec 元数据（QP+MV）条件网 | 已封板 NO-GO：干净对照下真条件≈乱序 | `h1_condnet/README.md` |
| `pacnet/h2_warp_oracle/` | H2 DIS 光流 warp 邻帧融合 oracle | 已封板 NO-GO：9 个多帧方法全负 | `h2_warp_oracle/README.md` |
| `pacnet/roi_tile/` | H3 选择性原生分辨率 tile（上帝上限） | 已封板 NO-GO：216 配置全负 | `roi_tile/README.md` |
| `pacnet/sr_aware_alloc/`（含 `bbb/`） | SR 感知逐段码率分配 LP oracle | 已封板 NO-GO：TOS+BBB 新颖性 ≤0.41；BBB 全量测量曾被叫停，续跑先查 json 完整性 | `sr_aware_alloc/README.md` |
| `pacnet/modern_sr/` | 真 4K60 BBB 上 1080p→4K x2 零样本探针 | 已封板 NO-GO：中位 +1.2、LPIPS 16/16 恶化、med/high 超时 | `modern_sr/README.md` |
| `pacnet/side_channel/` | 高频残差旁路信息论上界（grainextract/merge） | 已封板 NO-GO：+2 VMAF 需 31% 主码流，1~2% 有地板 | `side_channel/README.md` |
| `pacnet/logs/` | 后台 worker 日志 | 不入库（*.log/*.err 全局 ignore） | — |

## 8 月仿真器时代（`vbr_sims/`）

脚本运行依赖头部 `sys.path.append("..")` 与 cwd，**散件只认领归属、不物理移动**（移动曾导致复现失效）。

| 目录 | 主题 | 状态 | 目录档案 |
|---|---|---|---|
| `vbr_sims/multitask_usm/` | 多任务增强管线、USM、NAFNet 去模糊 | 已封板：1080p USM 负增益；hd_fs 增强不触发根因是质量表写死 −inf | `multitask_usm/README.md` |
| `vbr_sims/frame_level/` | 帧级 DP 调度 + 逐帧 VMAF 测量管线 | 已封板：10ms 记账 bug 修正后 oracle 增益 ≤3.4 | `frame_level/README.md` |
| `vbr_sims/lookahead/` | 多段前瞻 MPC、增量传输+渐进增强、联合方案 | 已封板：前瞻为未来内容 oracle 上界；增量被整段闸门架空 | `lookahead/README.md` |
| `vbr_sims/md_qoe/` | 自定义 MD-QoE 指标 | 设计稿，未落地；帧级指标体系重启时并入 | `md_qoe/README.md` |
| `vbr_sims/vbr/` | 服务端 VBR/平方根/阶梯码率 | 已封板（含作废 banner）：历史 131 轨迹全胜是非等码率+反事实造假 | `vbr/README.md` |
| `vbr_sims/_misc/` | 各早期工作线散件，**无独立档案**，归属见下方认领表 | 留存，不删 | 本页认领表 + `docs/key_findings.md` |

## 其他目录

| 目录 | 主题 | 状态 | 档案 |
|---|---|---|---|
| `enhancer_survey/` | 现成增强算子普查（NAFNet/STDF/NanoVSR）+ leave-segments-out 对症微调 | 已封板：现成算子全输 IMDN；对症跨块定性成立，+7~14 含 checkpoint 选择泄漏 | `enhancer_survey/README.md` |
| `gain_predictor/` | KNN 延迟预测 + RF/HistGBDT VMAF 增益预测器（step0-23 全套，含 pkl/csv） | 已封板：离线 R²≈0.91 在线 QoE −6.92，不准的预测比没预测更有害 | `gain_predictor/README.md` |
| `deblur_vmaf_test/` | 最早期去模糊/USM/wiener 段 mp4 对比（大 mp4 数据） | 已封板，整目录 ignore 不入库 | 归 multitask_usm 档案 |
| `sim_logs_v2/` `sim_logs_v3/` | 0 字节 .log 空壳（批量仿真占位） | 无内容，ignore | — |

## 散件认领表（物理位置保留）

| 散件 | 实际位置 | 归属工作线 | 采信规则 |
|---|---|---|---|
| `ablation_cmin_cmax.py` `run_sim_baseline.py` `run_sim_comparison_v4.py` `run_stage_a_sim.py` | `vbr_sims/vbr/` | CA-BONES/增益预测器时期（仿真器 BONES/CA 对比，import `control.bones_ca`） | **不是 #7 VBR 证据**；vbr/README:43,160 已标注 |
| `run_mt_comparison.py` `run_mt_10traces.py` `compare_bones_mt.py` `mt_comparison_10traces.json` | `vbr_sims/lookahead/` | 多任务增强（BONES_MT）对比 | **数据污染不可引用**（消费已判编造的 multitask json），lookahead/README M1 已封 |
| `run_three_way_comparison.py` `run_four_way_comparison.py` 及结果 json | `vbr_sims/lookahead/` | 前瞻/帧级/增量/联合对比（本档案） | 合成阶梯轨迹，可字节复现，见 lookahead/README L11 |
| `train_vmaf_predictor.py` | `vbr_sims/_misc/` | 增益预测器前身（阶段A 内容感知 VMAF 预测） | 已被 `gain_predictor/step0-23` 全套取代，仅溯源用 |
| `analyze_content_aware_gains.py` `analyze_content_features.py` `extract_content_features.py` `extract_sr_levels.py` `inspect_quality_data.py` `content_aware_nes_report_v2.md` | `vbr_sims/_misc/` | CA-BONES 内容因子时期 | 结论以 `docs/key_findings.md`、`docs/ca_bones_modes.md` 为准 |
| `sweep_resource_params.py` `resource_efficiency_analysis.json` | `vbr_sims/_misc/` | CARES 资源惩罚/能耗线 | 结论以 `docs/cares_resource_efficiency.md` 为准 |
| `validate_compression_restoration.py` | `vbr_sims/_misc/` | 传统去压缩（USM/双边/NLM）早期验证 | 归 multitask_usm；与 #5 step1b 同结论（传统法无稳定增益） |
| `test_mt_quick.py` `check_abort.py` | `vbr_sims/_misc/` | 多任务管线调试（bones_mt） | 调试件，无结论价值 |
| `test_pipeline_minimal.py` `test_log_path.py` `verify_vmaf_log.py` `vmaf_log_test.json` | `vbr_sims/_misc/` | 逐帧 VMAF 管线调试 | 管线本体在 `vbr_sims/frame_level/measure_per_frame_vmaf.py`（部分 docs 写 workspace 根为陈旧路径） |
| `generate_traces.py` | `vbr_sims/_misc/` | 合成网络轨迹工具 | lookahead 合成轨迹来源（L11） |
| `*_output*.txt` `sim_log_*.txt` `measure_progress.txt` `*pid*.txt` `*.err` `progress.txt` `tmp/` | 各目录 | stdout 转储/运行时产物 | 全部 ignore 不入库，可随时清理 |
| `temp_log.txt`（workspace 根） | — | 仿真器脚本硬编码 `log_path="temp_log.txt"` 的占位输出 | ignore |

## 入库与运行约定

- **入库**：代码、结果 json、档案 README、权重中已显式跟踪者。
- **不入库**（.gitignore）：视频 mp4、*.pth/*.pkl/*.npy（已跟踪者除外）、*.log/*.err、pid/progress、`tmp/`、stdout 转储 txt、`deblur_vmaf_test/`、`sim_logs*/`、训练 trainlog（随权重目录，档案另有说明）。
- commit 只精确 add 本轮文件；player/output 数据、日志、视频禁止顺手提交。
- 新增工作线 = 新建目录 + 十一节 README + 本页加一行；两个档案主题重叠时并入已有档案，不新建重复文件。
- 每次工作线有结论/反转时，同 commit 更新该档案 README 与 `docs/current_state.md`。

## 复现入口

- SR 环境解释器：`E:\dev\env\miniconda3\envs\SR\python.exe`；播放器工作目录 `player/`。
- 线程铁律：OMP/OPENBLAS/MKL=4、libvmaf n_threads=2、**单 worker 串行**（双 worker 同 tmp 文件名曾互抢死亡，多线程曾打满 16 核卡死桌面）；重任务开跑前告知。
- VMAF 铁律：连续时间序双 rawvideo libvmaf 管道为金标准（范本 `pacnet/sr_aware_alloc/measure_seg_rd.py`）；禁止非相邻抽帧 cv2 mp4v 打包（曾虚高 20~28 分）。
- vbr_sims 下脚本：cwd=对应子目录（`sys.path.append("..")` 找 `control/`）；reorg 后复现需 cwd=workspace 且 PYTHONPATH=仓库根。
- docs 侧导航：`docs/README.md`；当前阶段状态：`docs/current_state.md`。
