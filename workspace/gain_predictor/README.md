# 档案：VMAF 增益预测器与选择性增强（gain_predictor）

**状态：方向证伪（离线随机划分下高 R² 不代表流式可用；时序外推 R² 全负；仿真器 131 轨迹端到端净负；服务端真值选方法净增量≈0）。帧级粒度价值结论保留，并入帧级指标体系**
**时间：2026-09-05 ~ 09-06（step0~10 为 09-05 晚，step11~23 为 09-06 凌晨）**
**最近复核：2026-10-09（22 个 step 脚本 + `control/bones_selective.py` 全量读码；13 个结果 json 全部 SR python 现算）**

## 1. 要回答的问题

BONES 用全局平均增益表（每档码率×增强方法一个常数 VMAF）做联合决策，把高算力档分给"平均上"值得的段，与具体段内容无关。本工作线训练内容感知的逐段 SR 增益预测器（LR 侧特征：运动/纹理/场景切换/基础质量），试图让仿真器按段选增强方法，回答三个问题：

1. 增益能否从解码前可观测特征预测（离线 R²/排序）？
2. 预测器接入 BONES 联合决策后，端到端 QoE 能否超过静态平均表？收缩、门限、解耦、预算、算力受限等保护机制能否救回？
3. 若客户端联合决策不行，把增强层彻底解耦为非阻塞 best-effort 轨道、由"服务端真值"引导选方法，理论上限是否存在（step22）？

附带 step20：段平均是否掩盖了帧级可利用差异（为帧级方向取证）。

## 2. 方法概述

- step0~7：从 `data/bbb/per_frame_quality.json`（帧级张量，见档案 #2）构建段级/帧级数据集（12 个严格 LR 侧特征），训练 Ridge/HistGBDT，random 与 temporal 两种 holdout、5 折 CV、LORO 留一分辨率、预算蒙特卡洛，最终存三档 GBDT。
- step8~19：**另一套**数据与模型——标签改用 BONES 静态段表 `data/bbb/imdn_bbb_2080ti.json`（9 特征、无 res_idx、三分辨率混训、shuffle 5 折 OOF），产出与仿真器同形的 (159,5,5) 质量矩阵（oracle/pred/static 三份 npy），经 `control/bones_selective.py`（SelectiveBONES 继承 BONES）注入决策，在 131 条真实轨迹上对比；随后逐一尝试 alpha 收缩、下载/增强解耦、gating margin、预算阈值帕累托、收缩×阈值网格、算力倍率扫描。
- step20~21：纯数据分析，段内帧增益离散度、固定预算下段级/帧级/救最差帧三种选帧在 mean/p10/p5/min 口径下的价值。
- step22~23：独立非阻塞串行 GPU（EDF 保序、只在播放 deadline 前可行才启动，永不阻塞播放），对比 bones/FIFO/最高可行级/级别平滑/真值增益择优五策略，跨 4 个 GPU 倍率证伪"服务端引导"。

## 3. 文件清单

| 文件 | 作用 | 复核定性 |
|---|---|---|
| `step0_build_dataset.py` | 从 per_frame_quality.json 构建 dataset_seg.csv（474 行）/dataset_frame.csv（6.4MB）；12 特征严格 LR 侧；VMAF 有效域过滤 | 标签源有档案 #2 B7 口径隐患（D4） |
| `step1_eda.py`、`step2_baseline_analysis.py` | 分布/基线/相关性，出 s1~s3 图 | 仅 EDA |
| `step3_4_train_eval.py` | B0/B1/Ridge/HistGBDT；random 与 temporal holdout | **D1 核心证据：temporal R² 全负** |
| `step4b_generalization.py` | shuffle 5 折 CV + LORO 留一分辨率 | CV 乐观（D1）；LORO 跨分辨率不稳 |
| `step5_budget_simulation.py` | 5 折 OOF 下固定预算选段蒙特卡洛 | 离线排序价值，仅在 shuffle 口径成立 |
| `step6_save_final_model.py` | 全量训三档 GBDT → `gain_predictor_gbdt.pkl`（1.29MB，帧张量标签） | **从未进仿真器**（D4） |
| `step7_*.py`（帧级模型相关，编号跳号） | 帧级标签建模尝试 | 产物未被下游使用 |
| `step8_build_pred_matrix.py` | 用 **2080ti 段表标签**重训（9 特征、shuffle KFold OOF）→ `gain_predictor_static.pkl`（1.22MB）+ oracle/pred/static 三 npy | **D1+D4：仿真器实际使用的预测器，OOF 口径乐观、标签源切换** |
| `step9_selftest.py` | static 矩阵与原始 BONES 逐元素一致性自检（\|Δ\|<1e-9） | 有效，保证对照基线无偏 |
| `step10_full_sim_comparison.py` | 131 轨迹 static/oracle/pred 全量对比，报 qoe/vmaf/osc/freeze/enh_ratio/gpu_ms | **D2：pred −0.253，oracle 上限仅 +0.635** |
| `step11_alpha_sweep.py` / `step12_diagnose.py` | 收缩系数 0.25~1.0 扫描 / 暴跌轨迹诊断（无 json） | D2：非单调，最佳 +0.13 |
| `step13_decouple.py` | 下载/增强解耦 + 下载序列一致性自检 | **D3：自检 388 处不匹配，解耦结论作废** |
| `step14_gating.py` | margin 0~6 扫描 + gap 分位 | D2：最佳 +0.041 |
| `step15_plot.py` | 端到端增益汇总图 s6（数字硬编码，已与 json 核对一致） | 重跑不自动更新（D7） |
| `step16_budget_pareto.py` / `step18_pareto_plot.py` | 预算阈值帕累托曲线/图 s7 | D2：pred 非单调崩溃 |
| `step17_shrink_budget.py` | alpha×阈值网格 | D2/D7：非单调 −5.x 离群污染 |
| `step19_compute_constrained.py` | GPU 倍率 1.0/0.6/0.4/0.25 | D2/D7：均值被灾难轨迹支配，假设不成立 |
| `step20_frame_value_test.py` | 帧级价值三分析（**须在仓库根运行**，相对路径） | **D6：B 分析数字未落 json；结论保留** |
| `step21_frame_value_plot.py` | s8 图（数字硬编码，含 240/min −7.94 未解释异常） | D6/D7 |
| `step22_server_guided.py` | 非阻塞 GPU + 四策略，130 轨迹×4 倍率 | **D5：真值选方法净增量≈0，方向判决** |
| `step23_server_guided_plot.py` | s9 图（读 json，可复现） | 有效 |
| `control/bones_selective.py`（177 行，仓库 control/） | SelectiveBONES：static/oracle/pred 三质量源 + alpha 收缩/解耦/gating/budget 四种保护 | 报告统一用真值 `vmaf_enh_seg` 计分（`simulation/simulator.py:203`），对照受控；解耦结构缺陷见 D3 |
| 数据产物 | dataset_seg.csv、dataset_frame.csv、**dataset_sim_seg.csv（142KB，生成脚本全仓 grep 0 命中，孤儿）**、3 个 pkl、3 个 npy、13 个 json、figures/ 9 图 | pkl/npy 为大二进制，不入库 |

**第三个 pkl 不在本目录**：`workspace/vmaf_gain_predictor.pkl`（4.9MB）是 8 月 CA-BONES 的 RandomForest（训练脚本 `workspace/vbr_sims/_misc/train_vmaf_predictor.py`，标签为 3060ti 段表、5 特征、同样 shuffle 80/20），由 `player/istream_player/modules/nes/ca_bones.py:108`、`control/bones_ca.py:159`、`player/ca_predicted_config.json` 加载——历史"真实播放器 QoE −6.92"出自它，**不是**本档案两个 pkl，引用时勿张冠李戴。

## 4. 复现方式

```powershell
cd E:\dev\Code\Python\bones
$py = "E:\dev\env\miniconda3\envs\SR\python.exe"
# 离线线（step0~7 内部 ROOT 硬编码，任意目录可跑；step20/21 例外，必须在仓库根跑）
& $py workspace/gain_predictor/step0_build_dataset.py
& $py workspace/gain_predictor/step3_4_train_eval.py
# 仿真线（脚本内 os.chdir 到 control/，依赖 data/trace 131 条轨迹与 imdn_bbb_2080ti.json）
& $py workspace/gain_predictor/step8_build_pred_matrix.py
& $py workspace/gain_predictor/step10_full_sim_comparison.py
& $py workspace/gain_predictor/step22_server_guided.py
# step20 必须在仓库根：
cd E:\dev\Code\Python\bones; & $py workspace/gain_predictor/step20_frame_value_test.py
```

## 5. 结果记录（2026-10-09 全部现算，标注可引用性）

**离线（step4，GBDT，low/med/high 顺序）**：

| 划分 | R² | MAE | Spearman | TopK50 |
|---|---|---|---|---|
| random holdout（段级 shuffle 70/15/15） | 0.848 / 0.821 / 0.769 | 1.69 / 2.45 / 3.41 | 0.869 / 0.865 / 0.826 | 0.889 / 0.917 / 0.917 |
| **temporal holdout（前 70% 训后 15% 测）** | **−1.492 / −2.188 / −2.401** | 11.90 / 19.60 / 23.90 | 0.075 / 0.545 / 0.551 | 0.500 / 0.722 / 0.833 |
| shuffle 5 折 CV | 0.800±0.195 / 0.835±0.120 / 0.851±0.047 | — | 0.86 / 0.86 / 0.88 | 0.90 / 0.89 / 0.90 |
| LORO 留一分辨率（最差 held-out） | hold_240：−5.877（low）/0.086（med）/−1.098（high） | 4.9 / 6.1 / 14.0 | 0.64~0.69 | 0.71~0.73 |

temporal 下连全局均值 B0 的 R² 也为 −1.70/−2.65/−2.85（分布整体漂移，测试段均值与训练段不同）。

**仿真器端到端（step10，131 轨迹=86 3G+40 4G+5 根目录，2080ti 表）**：

| 方案 | ΔQoE 均值 | 胜率 | ΔVMAF | Δ振荡 | GPU 时间 |
|---|---|---|---|---|---|
| oracle joint（真值段表，上限） | **+0.635** | 79.2% | +0.383 | −1.091 | −5.9% |
| pred joint（alpha=1） | **−0.253** | 42.3% | −0.142 | −0.095 | +1.5% |

分网络 oracle ΔQoE：3G +0.72（−18.08→−17.36）、4G +0.39、root +1.22。

补救扫描：alpha 收缩 a0.25=−5.37 / a0.5=−0.06 / **a0.75=+0.13** / a1.0=−0.25（非单调）；gating 最佳 margin=1.5 **+0.041**/胜率 43.8%；预算帕累托 oracle 在 th=10（省 23.5% GPU）已 −0.049，pred 任意省 GPU 点均崩溃（th=8 −5.87）；算力倍率扫描均值非单调（sm=0.25 static=14.06 反高于 sm=0.4 的 8.88；oracle sm=0.6 均值 −24.1 但胜率 72%——少数灾难轨迹支配均值，与档案 #6 C1 同源）。

**解耦（step13，已因 D3 作废）**：oracle_dc −5.118、pred_dc −5.964；下载序列自检 **dl_mismatch=388**（131 轨迹×4 配置）。

**服务端引导（step22，干净重放，edf_oracle 减 edf_max 即真值选方法的净增量）**：sm=1.0/0.6/0.4/0.25 分别 **+0.004 / +0.009 / 0.000 / 0.000**，胜率 18%/9%/3%/2%。非阻塞执行本身（edf_max vs bones_orig）：sm=1.0 时 +1.545/胜率 98%，但增强占比 0.33→0.50、GPU +40%；sm≤0.6 时振荡 5.5→7.2~8.0，净收益转负（−0.004 / −1.71 / −1.32）。

**帧级价值（step20/21，BBB 158 段×120 帧）**：增益方差主要在段间（360/480 med/high 段内方差占比仅 2.9%~5.6%，段内 std 1.3~1.8 vs 段间 7.4~14.8；240_low 例外 35.8%）；最差 5% 帧由 13.3%/15.2%/20.9% 的段承载，这些段内平均 45/40/29 帧（共 120）落在最差 5%。预算 50% 时帧级相对段级的额外质量：按增益选帧 mean 口径仅 +0.19~0.30、p10 +0.65~1.35、p5 +0.31~0.74（240 min −7.94 异常）；按"救最差帧"准则 p10 +2.1~3.7、p5 +3.1~6.9、min +3.4~6.9。

## 6. 复核记录（2026-10-09）

| # | 发现 | 严重度 | 位置 |
|---|---|---|---|
| D1 | **离线高 R² 是随机段划分假象，流式时序外推全面崩溃**：random holdout GBDT R²=0.77~0.85，temporal holdout（前 70% 段训练、后 15% 测试，真实流式口径）R² 全部为负（−1.49/−2.19/−2.40），MAE 放大 7~10 倍；全局均值、Ridge 同样全负。shuffle 5 折 CV（0.80~0.85）与 step8 仿真器所用 OOF（`KFold(shuffle=True)`，step8:71）都让时间相邻段同时进训练/测试集，内容自相关把精度抬高。temporal 下仅 med/high 保留排序力（Spearman 0.55、TopK50 0.72/0.83）。8 月 RF（R²=0.909）同样是 shuffle 80/20（`_misc/train_vmaf_predictor.py:169-173`）。**"离线很准"不能作为在线可行性证据** | 致命 | `step3_4_train_eval.py`（temporal_split）；`step4b_generalization.py:71`；`step8_build_pred_matrix.py:71` |
| D2 | **端到端预测器净负，且 oracle 上限本身只有 +0.635**：131 轨迹 pred joint −0.253/胜率 42.3%；即使决策时知道真实段级增益，oracle joint 也仅 +0.635（3G 上 +0.72），预测误差把这点空间全部吃光。alpha 收缩（最佳 +0.13 且非单调）、gating（+0.041）、预算阈值（pred 任意省 GPU 点 QoE 崩溃）、算力受限（均值被离群卡顿支配，翻盘信号胜率仅 40%）全部未救回 | 致命 | step10/11/14/16/17/19 json（§5 现算） |
| D3 | **解耦模式自带自检失败（dl_mismatch=388），−5.1/−5.96 不是干净判决**：`_enhance_only` 只冻结本段下载码率，但 `_bones_control` 的目标含 `buff_enhance` 漂移项与 `mask_late`（`control/bones.py:80,95`），增强缓冲随被替换的 action_enh 演化，后续段下载决策必然分叉——双缓冲耦合下"段内解耦"在结构上不成立。step13 的 −5.x 不能引用为"解耦方向不行"的证据；干净版本是 step22（D5） | 致命 | `control/bones_selective.py:104-119`；`control/bones.py:72-108`；step13 json |
| D4 | **两套标签源/两个模型/两套特征，结论不能互证；无跨内容验证**：step0~7 标签来自 per_frame_quality.json（档案 #2 B7：双臂 cv2 mp4v 重编码、base 用 INTER_CUBIC 与播放器硬件缩放不一致、段边界 libvmaf 时域重置），其 pkl 从未进仿真器；step8+ 实际使用的模型标签换成 imdn_bbb_2080ti.json 静态段表（2080Ti 口径、9 特征、无 res_idx）。全部实验仅 BBB 单视频，LORO 只跨分辨率不跨内容。仿真在 5080 机器上用 2080Ti 延迟/质量表。dataset_sim_seg.csv 生成脚本缺失（孤儿产物） | 重要 | step0 vs step8；`step8_build_pred_matrix.py:14-16,25-37` |
| D5 | **服务端"真值选方法"净增量≈0（方向判决，可信）**：step22 冻结码率层与 freeze（三线相同），独立非阻塞 GPU 上 oracle 选方法相对简单规则"选 deadline 内可行最高级"ΔQoE≤0.009、胜率≤18%。非阻塞执行本身在快 GPU（sm=1.0）上 +1.5 但多耗 40% GPU、增强占比 0.33→0.50，GPU 稀缺时振荡加剧转负——即有收益的是"换执行架构烧算力"，不是"服务端引导"；且该 best-effort 调度器从未在真实播放器实现 | 重要 | `step22_server_guided.py:76-156`；step22 json |
| D6 | **帧级价值只在最差尾指标下存在，mean 口径下段级粒度已足够；但关键数字没落盘**：step20 main() 只存 within/scale/cluster，analysis_b 的选帧价值数字仅在控制台与 step21 硬编码数组中，无法从 json 复现；240/min −7.94 异常未解释。结论方向（差质量按整段聚集、段内方差占比 3~11%）与帧级指标体系（worst-1%/p5/p10）互相支撑 | 重要 | `step20_frame_value_test.py:171-192,237-243`；`step21_*.py:29-30` |
| D7 | 工程：①step20/21 相对路径须在仓库根运行、analysis_c `/120` 硬编码每段帧数（:223）；②step15/21 图内数字硬编码，重跑不更新（本次已手工核对一致）；③step17/19 大量非单调 −5.x 是少数轨迹卡顿 ms/10 爆炸支配均值（同档案 #6 C1），未做中位数/截尾稳健化；④step12 无 json、step18/23 纯出图；⑤tmp_log*.txt 为 stdout 散件（不入库） | 轻微 | 见左 |

## 7. 影响面：历史结论逐条重新定性

| 历史结论/产物 | 复核后定性 |
|---|---|
| "GBDT 增益预测器离线 R²≈0.91/0.85，预测很准" | **限定引用**：仅在 shuffle 随机段划分下成立；流式 temporal 口径 R² 为负。对外只能表述为"同内容独立同分布划分下可拟合，时序外推失败"（D1） |
| "真实播放器集成 QoE −6.92，预测器过度自信"（8 月 CA-BONES，vmaf_gain_predictor.pkl） | **成立且机制现可解释**：shuffle 训练的模型在时序漂移下过度自信 → 极端选档；与本档案 step10 pred −0.253 是同一病根的两次复现 |
| "仿真器 pred joint −0.25，alpha=0.75 可回正到 +0.13" | **数字可引用，结论是"无实际收益"**：+0.13 非单调、胜率 52%，不构成工作点（D2） |
| "oracle 上限 +0.635" | **可引用，且是最重要的负面证据**：即便预测完美，段级增益改写联合决策的天花板也不到 1 分 QoE |
| "解耦下载后增强净亏 5~6 分"（step13） | **作废**：下载序列自检 388 处不匹配（D3），实验未真正解耦；解耦执行的干净结论以 step22 为准 |
| "服务端引导/真值选方法无收益"（step22/23） | **成立、可引用**：净增量 ≤0.009；但需同时报告"非阻塞执行 sm=1.0 +1.5/−40%GPU 成本"这一伴随事实，不能只讲半句 |
| "帧级粒度没有价值"（step20 mean 口径 +0.2） | **改写**：mean-QoE 下无价值；worst-tail（p5/p10/min）下"救最差帧"有 +2~7 分，且差质量按整段聚集。是否有价值取决于指标是否考核尾部（D6） |
| 预算帕累托"oracle 可省 23% GPU 持平 QoE" | 数字可引用（th=10 为 −0.049），但 pred 永远达不到该曲线，工程意义仅在"若有完美预测器" |
| gain_predictor_gbdt.pkl / gain_predictor_static.pkl / 三个 npy | 保留不入库（大二进制）；前者未被任何运行时加载，后者仅 step9~19 离线仿真使用，播放器均不加载 |

## 8. 是否重做：判断与方案

**判断：不原样重做。"段级增益预测器改写 BONES 联合决策"在现有 mean-QoE + 段级表框架内已双重证伪（预测时序失败 D1 + 上限太小 D2），调参/加保护没有空间。** 可复活的只有方法论与帧级尾部结论，并入暂停中的帧级指标体系 B 阶段：

1. **评估口径改为 walk-forward**：只允许用 t 之前的段训练/校准，滚动预测未来段；主指标看 Spearman/TopK 命中率与误差敏感性，不看 R²（分布漂移下 R² 无意义）；跨内容留一视频（LOVO）为必测项，单视频 BBB 不作为证据。
2. **目标从"改写联合决策"改为"尾部保护"**：D6 表明预测器唯一稳定的价值是 med/high 的排序力（temporal TopK50 0.72/0.83）与"差质量整段聚集"——用于识别将出现 worst-5%/worst-1% 的段并提前保底，指标对应帧级体系的 p5/p10/worst-1%，而非 avg VMAF。
3. **标签与测量先统一**：所有预测标签必须来自连续时间序 rawvideo 管道重测的帧级张量（修档案 #2 B7），仿真器延迟表换现机 5080 重测；禁止再用 2080ti 段表与帧张量两套标签互证。
4. **解耦执行若要判决**：直接沿用 step22 的冻结重放口径（不引用 step13），且只有在真实播放器实现非阻塞增强队列后才作数；预期结论已明确——真值选方法无增量，不值得做服务端引导。
5. step20 analysis_b 数字补落盘后再被引用（重跑 step20 并把 B 分析写入 json，顺带解释 240/min −7.94）。

## 9. 踩坑铁律（本工作线新增）

1. **时序数据的随机段划分是数据泄漏**：相邻段内容近乎重复，shuffle CV/随机 holdout 会把"背诵邻居"误判为泛化；流式预测必须用 temporal/walk-forward 划分，R² 之外必看排序指标。
2. **先测 oracle 上限再做预测器**：上限（+0.635）小于预测误差量级时，预测方向直接止损，不要花时间做收缩/门限补救。
3. **"解耦"必须验证被冻结的对象在运行时真的不变**：自检断言（下载序列一致性）失败时实验作废，不能带着 388 处分叉继续解释 −5 分。
4. **仿真均值在重卡顿轨迹下会被 ms/10 口径离群值支配**（与档案 #6 C1 同源）：报告必须并列中位数、胜率与逐轨迹分布，非单调结果先怀疑口径而非寻找解释。
5. **同一篇工作里切换标签源（帧张量→静态段表）必须显式声明并重做基线**，否则两个模型的结论会被互相引用。
6. 分析脚本的关键结论数字必须写入机器可读 json，不能只留在控制台和画图脚本的硬编码数组里。

## 10. 上下游关系

- 上游：`data/bbb/per_frame_quality.json`（档案 #2，帧级张量，含 B7 口径隐患）；`data/bbb/imdn_bbb_2080ti.json`（BONES 段级测量表，2080Ti 口径）；8 月 CA-BONES（`control/bones_ca.py`、`player/.../nes/ca_bones.py`、`docs/ca_bones_modes.md`，vmaf_gain_predictor.pkl 与 −6.92 失败）；`control/bones.py` Lyapunov 联合决策。
- 同层：档案 #6（卡顿 ms/秒口径，解释 step17/19 均值爆炸）；档案 #7（VBR 与增量传输，step22 的非阻塞调度思想与档案 #7 增量传输仿真相关，审计时交叉引用）。
- 下游：**暂停中的帧级指标体系 B 阶段**（在线预测器：walk-forward 评估、排序指标、尾部保护目标）直接吸收 D1/D6；帧级指标体系 A 阶段负责修 D4 的标签口径；step22 是"服务端引导选方法"的封板证据，后续不再提该方向。

## 11. 复盘备注（复活检查清单）

- 复活时不要重训 step8 同款段表模型去改写联合决策；先读 §5 的 oracle 上限行。
- 若要做在线预测，从 step3_4 的 temporal 划分与 step20 的尾部分析出发，不沿用 step8 的 shuffle OOF。
- `control/bones_selective.py` 机制代码（质量源注入、报告真值计分）无 bug，可作为后续"替换决策质量矩阵"类实验的受控基座；但 `_enhance_only` 解耦路径（D3）禁止再用于结论。
- 引用本档案数字时：离线 R² 必须同时给 random 与 temporal 两行；端到端只引用 step10/step22 json 现算值；step13/step17/step19 的均值数字禁用。
- dataset_sim_seg.csv 为孤儿产物，无法复现，引用任何基于它的结论前先重建生成脚本。
- 三个 pkl 的归属：gain_predictor_gbdt.pkl=step6 帧张量标签（未使用）；gain_predictor_static.pkl=step8 段表标签（仿真器）；vmaf_gain_predictor.pkl=8 月 CA-BONES 真实播放器（`_misc/train_vmaf_predictor.py`）。
