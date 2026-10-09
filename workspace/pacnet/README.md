# 档案：PAC-Net 可渐进增强网（共享骨干多出口 anytime SR + halt 控制器）

**状态：已封板（系统侧判决：单帧 anytime 在轻量 CNN 小倍率超分甜区无帕累托空间，不原样重做；复活条件见 §8/§11）**
**时间：2026-09-10 ~ 09-21（阶段 0~3 + 播放器集成 + 多轨迹判决）**
**最近复核：2026-10-09（全量读码 + 全部判决脚本实跑复现 + 结果 json 逐字核对）**

## 1. 要回答的问题

BONES 的增强动作是四个**互相独立**的 IMDN 档（l0~l3），选中后整段跑完，算力不够只能整段 fast-complete 丢弃。本工作线验证一个 anytime 网络假设：

1. 用一个共享 6-block IMDN 骨干、在第 2/4/6 block 后开三个出口（e1/e2/e3），一次前向可在任意出口中断并拿到可用结果，质量-延迟是否单调、弹性税（相对同等规模专用网）有多大；
2. 算力紧张时，播放器按"可调度性前瞻"逐帧选出口（halt 控制器），能否比"强制最深出口"少卡顿、比"固定轻量档（IMDN l2）"质量高，即同时占住帕累托两端。

## 2. 方法概述

- **网络**：`PACNet(nf=32, upscale=4, exits=(2,4,6))`，三出口各自带重建头；从 IMDN high（nf=32）DIV2K 权重做 warm 迁移（`load_imdn_warm`，92 项参数）；另有 `PlainIMDN` 对症单模型对照。训练分阶段扫损失/课程/级联/多种子（阶段 1）、跨分辨率混合退化（阶段 2）、级联隔离与只保护浅出口（阶段 3/3b）、三种子复测（阶段 4）。
- **halt 控制器（播放器内最终版）**：纯 EDF 式可调度性前瞻，不用内容增益。对每帧逐级检查 `lead ≥ l_exit × 剩余帧 × 安全系数`，选满足条件的最深出口；连 e1 都不可行时退 fast-complete。安全系数默认 1.0，`PACNET_E3_THRESH`/`PACNET_E2_THRESH` 覆盖；`PACNET_FORCE_EXIT=0/1/2` 强制 e1/e2/e3 做消融对照。
- **算力注入**：`_gpuburn.py` 乘法模式 `PACNET_BURN_K=1.5`，每帧按实测前向 `fwd_ms×(K−1)` 烧 2048² matmul，端到端≈×K，三出口延迟比不变；PAC 与 IMDN 两个增强器同一调用点。
- **实验战役**：阶段 0 离线 16 段 GO 判决 → 阶段 1~4 训练弹性税定型 → 阶段 2 跨分辨率 → 离线 halt 仿真 → 播放器集成（BBB/TOS）→ 满血三轨迹 → K=1.5 加压多轨迹（halt/force_e3/IMDN 三策略）→ 安全系数扫描 → 含真实下载时间线的反事实与全 K 帕累托复核。

## 3. 文件清单

工作目录 `workspace/pacnet/`。本档案覆盖根 json、`scripts/`、`multi_k/`、`sweep_thresh/`、`weights/`（不入库）、`logs/`（不入库）、`multi_trace_logs/`（不入库）。子目录 `h1_codecmeta/`、`h1_condnet/`、`h2_warp_oracle/`、`hard_survey/`、`visual_compare/`、`roi_tile/`、`sr_aware_alloc/`、`modern_sr/`、`side_channel/` 分属档案 #9~#15，不在本档案。

### 3.1 scripts/（44 个 py，dca6b85 从 pacnet 根迁入）

| 分组 | 文件 | 作用 | 复核定性 |
|---|---|---|---|
| 模型/自检 | `pacnet_model.py` | PACNet/PlainIMDN/ExitHead/CondEncoder | 与播放器版 MD5 完全一致（F5） |
| | `selftest_model.py`、`selftest_cascade.py` | warm 迁移、出口 shape、截断自检 | 有效 |
| | `forward_equiv_check.py` | warm 后 e3 与标准 IMDN 逐元素等价（diff=0） | 有效，迁移正确性的硬证据 |
| 训练/评测 | `train_pacnet.py`、`run_stage1.py`、`run_stage2_curriculum.py`、`run_stage2_degradation.py`、`run_stage3_cascade.py`、`run_stage3b_protect.py`、`run_stage4_multiseed.py` | 六路径弹性税实验与定型 | 输出 json 全部逐字核对（§5）；训练脚本本身未逐行审计 |
| | `eval_pacnet.py`、`analyze_pacnet.py` | 16 段 PSNR/延迟判决（eval_pac16、eval_eq 两套 checkpoint） | 输出 verdict json 逐字核对 |
| halt 仿真/理论 | `run_stage3_halt_sim.py` | 账户模型离线 halt（1590 帧，低/中/高三预算 ×12 策略） | 有效，数字现算（§5） |
| | `run_theory_validation.py` | 竞争比/阈值/噪声/缓冲/稳态 α 30 trial | **F1：离线最优不是真上界，"竞争比"措辞须降级** |
| 播放器判决复核 | `sim_e0_falsify.py` | 虚拟更浅出口 e0 能否救场 | 已实跑：e0 全扫描从不被选中 |
| | `sim_pareto_k.py` | 固定码率消除下载变量后 K=1~6 帕累托 | 已实跑：K≥1.5 全面反输 |
| | `sim_joint_bitrate_enhance.py` | 码率×增强联合反事实 | 已实跑 |
| | `sim_download_counterfactual.py` | 严格复刻 trace.py 分块限速的下载时间线 | 已实跑；系统性低估卡顿 15~21s（M2） |
| | `analyze_stall_attribution.py` | 真实 json 的 slack/未增强帧/下载墙钟归因 | 已实跑 |
| | `analyze_loop_blocking.py` | 事件循环阻塞对下载的拖累（经验回归，不仿真） | 已读，方法成立 |
| | `sim_variable_compute.py` | 慢变马尔可夫/iid/burst 时变算力 | 已读，结论同恒定 K（反输 1~2.7） |
| | `sim_tos_halt.py`、`run_multi_trace.py` | 早期 TOS halt 仿真/满血多轨迹播放器批处理 | 未逐行审计；结论以真实 json 为准 |
| | `summarize_tos.py`、`summarize_multi_k.py`、`summarize_sweep.py`、`compare_runs.py`、`inspect_json.py` | 结果汇总/对照/查字段 | 汇总数字已与原始 json 交叉核对 |
| 测量 | `measure_pacnet_vmaf.py`、`measure_tos_vmaf.py`、`measure_tos_vmaf_full.py`、`measure_tos_imdn.py` | 质量表 VMAF 测量 | **M1：16 帧稀疏采样 + cv2 mp4v 打包，见 §6** |
| | `measure_locked_latency.py`、`measure_burn_calib.py` | 锁定频率延迟、gpuburn 标定 | 未逐行审计；延迟表 json 已被播放器实验消费 |
| 一次性迁移 | `patch_base_latency.py`、`patch_framerate.py`、`patch_gpuburn.py`、`patch_halt_experiment.py`、`patch_imdn_burn.py`、`patch_known_issues.py`、`patch_known_issues_counterfactual.py`、`patch_pacnet_doc.py`、`patch_state_counterfactual.py`、`patch_tos_support.py`、`patch_vmaf_accounting.py` 等 13 个 | 对播放器/文档的幂等文本补丁，含 "already patched/skip" 标记 | **F5：效果经读码确认全部在播放器/文档中，禁止重跑** |

### 3.2 根 json（18 个，均入库）

- 判决：`eval_pac16.json`/`eval_pac16_verdict.json`（定型 checkpoint）、`eval_eq.json`/`eval_eq_verdict.json`（另一 checkpoint）；
- 训练：`stage1_sweep.json`、`stage2_sweep.json`、`stage2_cross_res.json`、`stage3_sweep.json`、`stage3b_sweep.json`、`stage4_multiseed.json`；
- halt/理论：`stage3_halt_sim.json`、`theory_validation.json`（2026-10-09 重跑复现）；
- 质量测量：`pacnet_vmaf_measurements.json`、`tos_vmaf_measurements.json`、`tos_vmaf_full.json`、`tos_imdn_vmaf.json`；
- 段增益：`per_seg_psnr.json`（159 段 j=75 帧 e1/e2/e3 PSNR，gain 均值 1.166/std 1.004/min 0.150/max 4.702，本次现算一致）；
- 汇总：`multi_trace_summary.json`（BBB 满血 3G×3+4G×3，注意不是 TOS）。

### 3.3 实验战役目录

- `multi_k/`：`run_multi_k.ps1` + `summarize_multi_k.py` + `summary_multi_k.json` + progress + 9 组日志（.log/.err 不入库）。
- `sweep_thresh/`：`run_sweep_thresh.ps1` + `summarize_sweep.py` + `summary_sweep.json` + 6 组日志（不入库）。
- `weights/`：30 个 pth/trainlog，**不入库**；播放器实际加载 `weights/stage2_mix_nocond_x4_best.pth`（995947 字节）。
- `logs/`（105 个日志）、`multi_trace_logs/`（24 个）：不入库。

### 3.4 播放器落点（仓库 tracked 代码）

- `player/istream_player/modules/enhancer/pacnet_enhancer.py`（364 行，注册名 `pacnet`，继承 IMDNEnhancerAlways）；
- `player/istream_player/modules/enhancer/pacnet_model.py`（与 scripts 版 MD5 一致）；
- `player/istream_player/modules/enhancer/_gpuburn.py`；IMDN 侧同函数调用在 `enhancer_imdn_always.py:639`；
- 帧级记账 `player/istream_player/modules/analyzer/analyzer.py:622-672`；
- 质量/延迟表（播放器运行时加载，**当前未跟踪，建议随本档案提交**）：`player/pacnet_quality.json`、`player/pacnet_tos_quality.json`、`player/pacnet_latency.json`、`player/pacnet_latency_tos.json`；
- 主叙事文档 `docs/pacnet_progressive_enhancer.md`、理论文档 `docs/pacnet_halt_theory.md`。

## 4. 复现方式

```powershell
# 离线判决（纯 CPU，秒级~分钟级）
cd E:\dev\Code\Python\bones\workspace\pacnet\scripts
E:\dev\env\miniconda3\envs\SR\python.exe sim_pareto_k.py
E:\dev\env\miniconda3\envs\SR\python.exe sim_e0_falsify.py
E:\dev\env\miniconda3\envs\SR\python.exe run_theory_validation.py
# 真实播放器（单组约 13 分钟，headless；必须串行，线程铁律见 §9）
cd E:\dev\Code\Python\bones\workspace\pacnet\multi_k
powershell -ExecutionPolicy Bypass -File run_multi_k.ps1
```

播放器单组命令：`python play.py --config tos.yaml --input data/tears_of_steel/tos.mpd --mod_enhancer pacnet --trace data/3Glogs/<trace>.json`，工作目录 `player/`；`PACNET_BURN_K`、`PACNET_FORCE_EXIT`、`PACNET_E3_THRESH`、`PACNET_E2_THRESH` 经环境变量控制。素材：TOS `E:\dev\Data\tears_of_steel_1080p.mp4`（1920×800@24fps，184 段×96 帧，DASH 产物 `player/data/tears_of_steel/`）；BBB `data/video_mp4box/output_*.mp4`（30fps，158 段×120 帧）。

## 5. 结论与关键数字（均为本次复核现算/逐字核对值）

### 5.1 阶段 0：离线 GO（BBB 16 段，j=75 单帧/段，PSNR；`eval_pac16_verdict.json`）

- e3=29.63 dB；对症单模型 imdn_lo=29.917（**−0.287**），通用 imdn_div=28.365（**+1.265**）；e1 相对 plain2 **−0.765**（弹性税）；
- 越深越单调链占比 0.9844；困难段 e3−e1=**+2.506**，简单段仅 +0.756（内容异质性成立）；
- 延迟 e1/e2/e3=14/26/39 ms（BBB 30fps 口径）。另一 checkpoint（eval_eq）：−0.404/+1.148、e2=26.44 ms。
- 迁移事故（已修复）：warm 迁移曾臆测键名，入口 fea_conv 与最深出口重建头漏迁保持随机，e3 一度仅 24.2 dB 误判 NO-GO；修正后 92 项全迁、`forward_equiv_check.py` 与标准 IMDN 逐元素 diff=0。

### 5.2 阶段 1/3/3b/4：弹性税消除失败（六路径全负）

等权损失、像素自蒸馏 α 扫（α 越大越差）、深度课程（cur10 e3 29.505）、课程+冻结（mono 链仅 0.641）、残差级联三段隔离（崩到 28.223）、只保护 e1 级联（28.268）均未挽回。三种子定论（`stage4_multiseed.json`）：**弹性税 −0.781±0.056 dB**，e3=29.551±0.038，单调链 0.901±0.019，困难增益 2.461±0.046。结论：弹性税是共享多出口 anytime 结构的固有梯度冲突代价。

### 5.3 阶段 2：跨分辨率混合退化（`stage2_cross_res.json`，720p 为留出未见退化）

B_mix_nocond（240/360/480 混合、无在线条件）在全部分辨率优于 A_single240，720p e3 36.925 vs 36.326（**+0.599**）；C_mix_cond（CondEncoder 在线统计 FiLM）720p e3=36.709，低于 B（−0.216），定型不挂 CondEncoder。工程坑：720p 大输入 cuDNN 慢速路径，解法逐模型加载 + `forward_tiled`（tile=640、重叠 16）；CondEncoder 不得用 batch 维标准化（batch=1 统计恒 0）。

### 5.4 离线 halt 仿真（`stage3_halt_sim.json`，1590 帧）

低预算（20ms/帧）：固定 e3 有效质量 21.72（掉帧 28.5%），贪心 28.827（掉帧 3.5%），**CAv2_t2.0 29.321（掉帧 1.9%），比贪心 +0.494**。中/高预算下 CAv2 不优于贪心（29.958 vs 29.998；30.055 vs 30.518）——内容感知 halt 只在紧预算区有价值。

### 5.5 TOS 离线质量/延迟（240p，16 帧稀疏采样，口径见 M1）

VMAF：bicubic 80.43，e1 86.18，e2 87.12，**e3 90.22**，l1 87.02，**l2 88.70**，l3 89.64；360p：94.26/96.51/97.62/99.02/96.43/97.26/98.11；480p：97.92/99.70/99.80/99.84/99.53/99.84/99.84。延迟（ms/帧）：e1 11.64、e2 24.19、e3 33.79；l1 3.6、l2 5.1、l3 9.8。**关键结构事实：e1/e2 在质量与延迟两个维度同时被 l2 严格 dominate，只有 e3 质量最高。**

### 5.6 真实播放器：满血（无 burn）

- TOS 三轨迹（`summarize_tos.py`，headless/BONES/fast 开）：ΔQoE 0913 **+1.30**、0914 **+1.98**、0921 **+1.61**，平均 **+1.63**（ΔVMAF +1.13，卡顿 +0.6s 打平）；3/3 胜，质量切换 48 vs 69，PAC fc 全 0。但 halt 172/177/183 段全程 e3，IMDN 主力 l2（108/135/156 段）——**halt 不换档，PAC 退化为"更深的固定模型"，anytime 卖点未被检验**。
- BBB 三轨迹（`multi_trace_summary.json`，另一战役）：ΔQoE +0.42/+3.32/+5.42（卡顿大幅少于 IMDN），三条 4G 轨迹均 −0.10（打平）。

### 5.7 真实播放器：K=1.5 加压多轨迹（`multi_k/summary_multi_k.json`，TOS）

| trace | 策略 | VMAF | QoE | 卡顿 | 次数 | fc 段 | 出口/档位分布 |
|---|---|---|---|---|---|---|---|
| 0913 | halt | 93.87 | 91.11 | 9.04s | 2 | 1 | e1/e2/e3=1.0/11.1/87.9% |
| 0913 | force_e3 | 93.62 | 90.77 | 14.99s | 3 | 41 | 100% e3 |
| 0913 | IMDN | 93.26 | 90.05 | 0.09s | 1 | 1 | l0/l1/l2/l3=55/11/69/49 |
| 0914 | halt | 92.61 | 90.50 | 2.46s | 2 | 1 | 2.1/13.2/84.6% |
| 0914 | force_e3 | 91.05 | 88.32 | 8.67s | 6 | 77 | 100% e3 |
| 0914 | IMDN | 92.12 | 89.12 | 0.06s | 1 | 4 | l2 为主 114 段 |
| 0921 | halt | 89.13 | 86.90 | 84.51s | 23 | 15 | 12.5/18.4/69.2% |
| 0921 | force_e3 | 88.06 | 85.80 | 102.05s | 31 | 89 | 100% e3 |
| 0921 | IMDN | 89.27 | 87.21 | 68.96s | 16 | 5 | l2 150 段 |

- **halt vs force_e3：三轨迹全胜（VMAF/QoE/卡顿全部占优）——anytime 降档保命机制本身成立。**
- **halt vs IMDN：2/3 胜，最差网 0921 反输（QoE −0.31），卡顿三条轨迹全部劣于 IMDN。**

### 5.8 安全系数扫描（`sweep_thresh/summary_sweep.json`，系数越大越保守）

0921：s1.0=89.13/86.90/84.51s/23 次/fc15；s1.1=88.93/86.75/81.17s/23/fc20；**s1.2=89.65/87.46/81.93s/17/fc4（VMAF/QoE 反超 IMDN，卡顿仍 81.9 vs 69.0）**；s1.3=88.88/86.61/83.90s/22/fc20。0914：s1.0=92.61/90.50/2.46s；s1.2=93.09/90.64/8.47s/3 次（卡顿反升）。系数效果非单调，**卡顿总时长追不平 IMDN，调参救不了**。

### 5.9 含下载时间线的反事实与全 K 帕累托（本次全部实跑复现）

- **全 K 帕累托（`sim_pareto_k.py`，固定 repr0 消除下载变量，帧级 oracle 自适应上界）**：K=1.0 +0.58、1.2 +0.57、**1.5 −0.80、1.8 −2.24、2.0 −2.88、2.5 −4.19、3.0 −5.07、4.0 −5.69、5.0 −6.39**；越紧张越穿。
- **e0 证伪（`sim_e0_falsify.py`）**：K=1.5 下 halt_e1 已 0 掉帧（88.302）；虚拟更浅出口（l0 扫 3.5~11.64、V0 扫全域、safe 扫 1.0/1.2/1.5）**逐字节相同、e0 从不被选中**；force_e3 掉帧 98.75%（K=1.5）；IMDN l2 全 K 0 掉帧稳在 88.70。
- **联合反事实（`sim_joint_bitrate_enhance.py`）**：固定 repr0+halt 卡顿 0、VMAF 88.79，仅比固定 repr0+IMDN l2（88.70）高 **0.09**；固定 repr1 两边都卡 512.2s，halt 83.57 反输 IMDN 88.70；满血 e3 角点上界 90.22。
- **下载归因（`analyze_stall_attribution.py`）**：三策略 slack 全 ≥0（无"增强做完已过播放点"）；halt 中位 slack 1.53s（force 0.71、IMDN 12.68）；halt 未增强帧 540（force 1914、IMDN 240）；halt 总墙钟 802.4s > 视频 736s，卡顿主因是 BONES 在烂网选了高码率档（下载变量），不是增强抢 GPU。
- **时变算力（`sim_variable_compute.py`）**：慢变马尔可夫/iid/20% 突发 2.2× 同样反输 1~2.7。

### 5.10 根因判决

共享 nf=32 重骨干的浅出口是"深网半成品"：e1（11.64ms,86.18）、e2（24.19ms,87.12）被专用轻模型 IMDN_RTC l2（5.1ms,88.70）在质量和延迟上**严格 dominate**；唯一质量最高的 e3（33.79ms,90.22）在算力紧张时不可持续。anytime 网络想覆盖的"紧算力灰区"恰好是共享骨干浅架最弱的区间，而该区间正是轻量专用 CNN 的甜区。**单帧、小倍率、轻量 CNN 超分场景下，多出口 anytime 没有帕累托空间。**

## 6. 复核记录（2026-10-09）

| # | 发现 | 严重度 | 位置/证据 |
|---|---|---|---|
| F1 | 理论验证的 offline_opt 按增益降序贪心、遇第一个不可行即 break，**不是真正的离线最优上界**（均匀代价下高增益段时间聚簇，按增益排序的集合可调度性差；真上界需对时间打包做 DP/背包）。B=0 时它只选 14.5% e3，因果 CA 选 55.3%，**ca_ratio=1.0051>1**（比值大于 1 直接证伪分母是上界）。0.9932/γ=0.9956 只能表述为"对增益排序贪心基线的经验性能比"，不能称竞争比保证；threshold_strategy 的可行性修复还用了全知缓冲检查（非因果），与因果 CA 存在口径差 | 重要（理论措辞） | `run_theory_validation.py`；`theory_validation.json` buffer_analysis B=0；`docs/pacnet_halt_theory.md` 相关措辞须降级。**不影响系统侧判决**（判决建立在真实播放器 json + 同仿真内对比上） |
| F2 | 播放器加载的 `per_seg_gain` 是**死数据**：最终 `_halt_decide` 纯 EDF 可调度性，完全不读增益；且 TOS 的 `per_seg_psnr_tos.json` 在 workspace/pacnet 与 player 均不存在（Test-Path=False），TOS 运行实际走 gain=1.0；文件头 docstring 仍写"段内容难度决定"，属陈旧描述 | 轻微 | `pacnet_enhancer.py:73-88`（加载）、`:153-169`（不使用）、`:1-2`（陈旧 docstring） |
| F3 | fast-complete 有**两道闸门**，summary 的 fc 段数含两类，引用须区分：①段级预中止（用 BONES 下发 enhance_action 的 `_min_threshold` 整段 is_enhance=False，analyzer 走 `:672` 全 base）；②帧级 fc（循环内剩余帧置 None、cnt 补满，analyzer 走 `:649-668` 按 enhanced ratio 混合） | 重要（口径） | `pacnet_enhancer.py:229` 与 `:261-270`；`analyzer.py:631-668`、`:672` |
| F4 | 两套 fps/延迟口径并存：理论/BBB 30fps（DT=33.3、d=14/26/39）与帕累托/TOS 24fps（DT=41.67、d=11.64/24.19/33.79）；`run_theory_validation.py` 默认 safe=1.5/gain_thresh=1.5，播放器默认 safe=1.0。跨战役数字禁止直接混用 | 重要（口径） | 各脚本常量；`pacnet_enhancer.py:40` 默认延迟是 BBB 口径，TOS 靠 `pacnet_latency_tos.json` 覆盖 |
| F5 | 播放器版与 scripts 版 `pacnet_model.py` MD5 均为 `9A921B7646BCD15999FAC765DCA9FB71`；13 个 `patch_*.py` 均为幂等一次性迁移脚本，效果经读码确认已在播放器（双增强器 gpuburn、analyzer 帧记账、TOS 延迟缓存、halt 环境变量、DASH 分数帧率等），**不需要也不应重跑** | 登记 | 见 §3.1 |
| M1 | **质量表测量口径稀疏**：`measure_tos_vmaf_full.py` 在全片 linspace 抽 16 帧（184 段只覆盖 16 个位置）、`measure_pacnet_vmaf.py` 抽 16 段各 1 帧（j=75），再把非相邻帧用 cv2 mp4v 打包成 16 帧连续 clip 喂 libvmaf；阶段 0/PSNR 与 per_seg_psnr 同为单帧/段。非相邻帧打包制造人工切镜（与 hard_survey 虚高事故同族，见档案 #9），双臂都走同一 mp4v 使偏差部分对称，但 SR 帧与 bicubic 帧高频内容不同、差分压缩失真不保证抵消。**播放器所有 PAC/IMDN 的 VMAF/QoE 都查这张表，从不是对真实增强输出逐帧算的**（与 `docs/known_issues.md:76-77` 口径声明一致）。延迟侧为 cuda 同步实测，不受影响 | 重要（证据强度） | `measure_tos_vmaf_full.py:71-90`、`measure_pacnet_vmaf.py`；#9 的连续帧复核只覆盖困难退化区间，240p→1080 的 PAC-vs-IMDN 质量差**从未用 rawvideo 连续口径重测** |
| M2 | 下载反事实仿真假设增强与下载完全并行，三策略真实码率序列重放卡顿 65.7/81.3/60.9s，对真实 84.5/102.0/69.0 **系统性低估 15~21s**（未建模增强对解码/起播/下载协程的资源反压）。同仿真内横向对比与真实排序一致，**相对结论成立、绝对卡顿数不可引用** | 重要（口径） | `sim_download_counterfactual.py` |
| M3 | gpuburn 乘法把"选深档"与"多烧 GPU 拖慢下载"人为耦合；真实弱设备是全档等比变慢，**K=1.5 可能高估深档惩罚**。但固定 repr0 消除下载变量后帕累托仍反输（§5.9），判决不依赖该耦合 | 限定条件 | `_gpuburn.py`、`sim_pareto_k.py` |
| M4 | 加压下两策略信息机制不同：PAC halt 直接用 ×K 静态延迟表（先知 K），IMDN 的 BONES 决策靠在线 EWMA 更新延迟表反应（`enhancer_imdn_always.py:970-1049`，未访问格按 avg_scale 缩放）。偏差方向对 PAC 有利，而 PAC 卡顿仍输，结论保守 | 限定条件 | 两处延迟逻辑 |
| M5 | `measure_latency` 只测 model forward，不含解码、resize、H2D/D2H 与调度开销；在 e3 延迟逼近帧间隔的临界区偏乐观 | 轻微 | 测量脚本；`docs/pacnet_progressive_enhancer.md:187` 已自陈 |
| M6 | 未逐行审计（输出 json 已逐字核对、结论可追溯）：`train_pacnet.py`、`run_stage1~4`、`eval_pacnet.py`、`analyze_pacnet.py`、`selftest_cascade.py`、`measure_locked_latency.py`、`measure_burn_calib.py`、`sim_tos_halt.py`、`run_multi_trace.py`、各 summarize/compare/inspect 脚本 | 登记 | — |

## 7. 影响面：历史结论逐条重新定性

| 历史结论/数字 | 复核后定性 |
|---|---|
| 阶段 0 GO：e3 仅比对症单模型低 0.287 dB、反超通用模型 1.265 dB，单调链 98.4%，困难段增益 2.506 | **保留**（PSNR 单帧/段口径，仅用于离线训练侧判决；M1 不影响 PSNR 对照的相对排序） |
| 弹性税 −0.78±0.06 dB、六路径消税失败 | **保留**（三种子 json 现算一致） |
| 混合退化跨分辨率泛化（含留出 720p +0.599），CondEncoder 不挂 | **保留** |
| 离线 halt 仿真紧预算 +0.494、掉帧 1.9% | **保留但限定**：只在紧预算区成立（中/高预算不优于贪心，§5.4）；账户模型仿真，非播放器 |
| TOS 满血三轨迹 3/3 胜、平均 ΔQoE +1.63 | **数字保留，定性降级**：halt 几乎不换档（172/177/183 段 e3），检验的是"更深固定模型 vs l2"，**不是 anytime 机制**；且 VMAF 查稀疏表（M1） |
| BBB 满血 PAC 占优 / 4G 打平 | **保留**（同上口径限定） |
| K=1.5 下 halt vs force_e3 三轨迹全胜 | **保留，核心机制结论**：anytime 可中断降档保命真实有效 |
| K=1.5 下 halt vs IMDN：2/3 胜、0921 反输、卡顿 0/3 劣于 | **保留，核心判决证据** |
| 安全系数扫描：s1.2 画质反超但卡顿仍 81.9 vs 69.0s | **保留**：调系数救不了卡顿 |
| 全 K 帕累托 K≥1.5 反输 0.8~6.4、e0 从不被选中 | **保留，判决性证据**（同仿真内对比，M2/M3 不改变排序） |
| 固定 repr0 下 halt 仅比 IMDN l2 高 0.09 VMAF | **保留**：下载变量消除后质量基本打平，anytime 无质量红利 |
| 理论"竞争比 0.9932/0.9956、稳态 α≈56%、B=500 e3 占比 79.9%、σ=3 仅降 0.03%" | **部分降级（F1）**：改为"对增益排序贪心基线的经验性能比"；最坏情况紧上界与弹性税下界本就是猜想（理论文档已自陈），F1 进一步说明上界构造不成立 |
| "halt 内容感知按段增益选出口"（早期文档/印象） | **作废**：播放器最终版不含增益项（F2），是纯可调度性前瞻 |
| 仿真给出的卡顿绝对值（65.7/81.3/60.9s 等） | **不可单独引用**（M2，低估 15~21s）；只能与同仿真内其他策略比 |
| 浅出口 e1/e2 的 VMAF（86.18/87.12）与 l2（88.70）的质量差 | **方向保留、数值挂起**：M1 稀疏打包口径；结构性 dominate 的延迟侧（11.64/24.19 vs 5.1ms）证据硬，质量侧需 rawvideo 连续口径复核 |

## 8. 是否重做：判断与方案

**判断：不按原样重做，单帧 anytime 共享骨干方向封板。** 六路径训练消不掉弹性税、安全系数调不平卡顿、加更浅出口 e0 不被选中、全 K 帕累托 K≥1.5 全面反输、时变算力同样反输——五条独立证据链指向同一结构根因（§5.10），不是调参或实现问题。

**但质量侧存在一个必须先补的测量债（M1）**：PAC-vs-IMDN 的质量差来自 16 帧稀疏 + cv2 mp4v 打包表，而这正是"e1/e2 被 l2 严格 dominate"判决的一半支柱（另一半延迟是硬实测）。补法不在本档案内单独做，并入暂停中的**帧级指标体系 A 阶段**（统一 rawvideo 连续全片 libvmaf、双臂对称、帧源与播放器解码路径对齐，范本 `modern_sr/probe_modern_sr.py` 的双 rawvideo 堆叠管道）：

1. TOS 240p（优先，跨视频公平口径）四码率 ×{bicubic, e1, e2, e3, l1, l2, l3} 连续帧 VMAF + 抽帧 LPIPS；
2. 若连续口径下 e1/e2 仍被 l2 双维度 dominate → 本档案永久封板；若质量排序反转（打包口径误导）→ 只重开"质量表 + 帧级记账"，halt 机制代码不动；
3. BBB 同口径新测做跨内容复核。

**未被本档案排除的相关方向**（不在此重做，登记给后续档案）：视频时序超分（多帧亚像素信息，是单帧 anytime 盖棺后唯一未排除的"换信息源"）；可伸缩/渐进引导传输（见档案 #15 与档案 #3 的增量传输仿真）。

## 9. 踩坑铁律（本工作线新增）

1. **warm 迁移必须逐层核对键名并跑逐元素等价检查**：漏迁入口层和最深重建头时网络半随机，能产出"像样但 NO-GO"的假结果，曾直接导致一次误判。
2. **anytime 延迟注入必须用乘法系数不能用固定加法**：固定加法压平出口延迟比，会让中间出口变成永远选不到的死档。
3. **halt 判据只用可证明的量（lead time）**：段内平均预算过度保守（87% 误降）、库存水位会被 fast_complete 阈值抢先；内容增益在播放器最终版不参与决策。
4. **降档实验必须有帧级 VMAF 埋点**：否则降档帧按下发的深档查表打分，虚高约 4 分/段，"0 卡顿+画质反超"是假象（`analyzer.py:631-668`）。
5. **质量表禁止稀疏非相邻帧打包成连续 clip 喂 libvmaf**（M1，与 #9 虚高事故同族）；新测量一律 rawvideo 连续帧。
6. **线程铁律**：OMP/OPENBLAS/MKL=4、libvmaf n_threads=2、单 worker 串行；双 worker 曾打满 16 核卡死桌面。重跑前先查残留 python/ffmpeg 进程。
7. **仿真里"增强与下载并行"是乐观假设**：给得出策略排序、给不出可信卡顿绝对值（M2）。
8. **大分辨率 cuDNN 慢速路径**用逐模型加载 + tile 前向（640/重叠 16）；CondEncoder 不能用 batch 维统计。

## 10. 上下游关系

- 上游：BONES 选档框架与 IMDN 三档模型/权重（`player/istream_player/modules/enhancer/imdn_weight/`）；BONES 段级质量表方法学（静态查表 QoE）；TOS/BBB 多码率 DASH 产物与 3G 轨迹（`player/data/3Glogs/`）。
- 同层：halt 理论文档 `docs/pacnet_halt_theory.md`（F1 后措辞待修）、主叙事 `docs/pacnet_progressive_enhancer.md`。
- 下游：困难退化普查（#9 hard_survey，在单帧质量差上补了连续区间/LPIPS 证据）、H1/H2/H3 三路证伪（#10~#12）、SR 感知码率分配（#13）、1080p→4K 探针（#14）、高频引导旁路（#15）——均在本档案"单帧端侧后处理信息不足"的判决后依次展开。
- 交叉：帧级 DP（档案 #2）、增益预测器（档案 #4）与本档案共享"帧级调度"命题；帧级指标体系（暂停中）是 M1 的统一还债入口。

## 11. 复盘备注（复活检查清单）

- **机制结论可直接引用**：anytime 可中断出口在算力骤紧时相对固定最深档"保命"成立（halt vs force_e3 三战三胜）；被证伪的是"共享骨干浅出口在轻量 CNN 甜区打得过专用轻模型"。
- **复活前必须先还 M1 测量债**：连续 rawvideo 口径重测 TOS 240p 质量表（§8 步骤 1~2），这是任何 PAC 相关新结论的前置。
- **不要重跑** 13 个 patch 脚本（F5）；不要直接引用仿真卡顿绝对值（M2）；不要把理论验证的 0.99xx 叫竞争比（F1）。
- **fc 段数引用先分清两道闸门**（F3）；跨 fps 口径数字禁止混用（F4）。
- 播放器 `pacnet_enhancer.py` 仍注册可选、默认不启用；4 个 pacnet*.json 配置表建议随本档案入库（缺失则播放器实验不可复现）。
- 若未来走时序 VSR：多出口思想可在"帧间传播深度可变"的新结构里重新评估，但那是新网络设计，不是本档案网络的调参延续。
