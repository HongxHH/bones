# 档案：帧级选帧因果化与六轨迹回归（m5_causal_rank）

**状态：M5 完成。帧级增益在下载前可得特征下基本不可预测（Spearman 0.05–0.34），学习型帧排序不部署；队列默认改为 EDF 因果排序，oracle 保留为标注上界；六真实轨迹上 COMB(EDF) 对 BONES 5/6 胜、中位 +1.30 QoE。**
**时间：2026-10-11**

## 1. 要回答的问题

M3/M4 队列的密度排序直接使用增强后真实 VMAF 计算 dv/dc，属全知 oracle（B6），不可部署。M5 回答：

1. 帧级增强增益能否只用下载完成时可获得的信息预测（排序口径，非 R²）？
2. 若不能，可部署的因果排序应采取什么结构，相对全知上界损失多少？
3. 帧级调度控制器在真实 3G/4G 轨迹上相对 BONES 基线是否稳定占优？

约束沿用 #4 教训：不准的预测比没预测更有害；时序数据禁止 shuffle 划分。

## 2. 文件清单

| 文件 | 作用 |
|---|---|
| `validate_causal_rank.py` | 56880 行样本（158 段×3 分辨率×120 帧），5 连续块交叉验证；对比 texture 单特征、HistGBDT 中位数、HistGBDT 分位数 q15 三种排序器；输出组内 Spearman、top25/50/75 预算后悔、高估率 |
| `causal_rank_offline.json` | 上述验证结果 |
| `validate_base_feature.py` | 补测离线预测量每帧 base VMAF（BONES 静态质量表同性质，非运行时先知）作为特征 |
| `causal_rank_base_feature.json` | base 特征验证结果 |
| `run_oracle_vs_edf.py` | 四合成轨迹 × 三控制器 × edf/oracle 端到端对比，每策略独立仿真器 |
| `oracle_vs_edf.json` | 端到端对比结果 |
| `run_six_trace_regression.py` | 六真实轨迹（3G 0913/0914/0921 + 4G bus/car/bicycle）× 四策略回归 |
| `six_trace_regression.json` | 回归结果 |

队列改动在 `simulation/frame_enhance_queue.py`：构造器 `rank_mode`（默认 `"edf"`），`schedule_segment` 双分支——oracle 保留密度降序多趟链逻辑；edf 按帧序扫描、每帧逐档 `try_admit` 到最深可行档；`summary` 增 "rank mode"。

控制器改动（`control/bones_incremental.py`、`bones_frame_level.py`、`bones_combined.py`）：按 `getattr(args,'oracle_rank',False)` 构造队列；report 增 "rank mode"；增强帧比例统一取队列 summary（修复 FL 分母含 720/1080 基础段导致的 85.8% vs 99.0% 差异）。

## 3. 复现方式

```powershell
cd E:\dev\Code\Python\bones
$py = "E:\dev\env\miniconda3\envs\SR\python.exe"
$env:OMP_NUM_THREADS=4
& $py workspace/m5_causal_rank/validate_causal_rank.py
& $py workspace/m5_causal_rank/validate_base_feature.py
& $py workspace/m5_causal_rank/run_oracle_vs_edf.py
& $py workspace/m5_causal_rank/run_six_trace_regression.py
& $py tests/test_frame_queue.py
```

## 4. 结果记录（2026-10-11，全部 SR python 现算）

**离线排序（组内 Spearman，L1/L2/L3 = base→low→med→high 三档增量）**：

| 排序器 | L1 | L2 | L3 | 高估率 |
|---|---|---|---|---|
| texture 单特征 | 0.335 | 0.137 | 0.119 | —（尺度不同，只看排序） |
| HistGBDT 中位数 | 0.241 | 0.102 | 0.061 | 0.45/0.46/0.49 |
| HistGBDT q15（保守下界） | 0.261 | 0.107 | 0.055 | 0.18/0.18/0.19 |
| GBDT + 每帧 base VMAF | 0.294 | 0.088 | 0.101 | 0.18 |

top25 预算后悔（每选中帧 VMAF 损失）L1：texture 0.69、q15 0.88；L2/L3 同量级 0.4–0.9。

**端到端（四合成轨迹，oracle 相对 EDF）**：平均 +0.076 QoE、+0.048 VMAF；最大单轨迹 volatile +0.60 QoE，其余 ≤0.02。全知帧排序头间可忽略。

**六真实轨迹回归（COMB EDF 对 BONES QoE）**：

| 轨迹 | BONES | COMB(EDF) | ΔQoE | oracle−EDF | COMB 卡顿 s | 增强率 |
|---|---|---|---|---|---|---|
| 3g 0913 | 77.64 | 78.94 | +1.30 | +0.26 | 0.00 | 100.0% |
| 3g 0914 | 64.17 | 68.51 | +4.34 | −0.08 | 292.12 | 99.9% |
| 3g 0921 | 65.73 | 67.08 | +1.34 | +0.70 | 5.65 | 100.0% |
| 4g bus | 96.57 | 96.64 | +0.07 | 0.00 | 0.00 | 100.0% |
| 4g car | 96.24 | 96.10 | −0.15 | 0.00 | 0.00 | 100.0% |
| 4g bicycle | 96.49 | 96.56 | +0.07 | 0.00 | 0.00 | 100.0% |

汇总：5/6 胜、中位 +1.30、均值 +1.16；真实轨迹 oracle 头间均值 +0.145。

## 5. 结论与口径限制

1. 帧级增益在段内对 motion/texture/scene_change/预测量 base VMAF 近似随机排序——段级纹理与段级增益高相关（#4，r≈0.91），但段内帧间增益差异不被这些特征承载；与 #13"增益空间均匀、SR 新颖性 ≤0.41"方向一致。
2. 学习型帧排序不接入；可部署版采用 EDF（等增益假设下最优可行性保持顺序），全知上界证明其损失 <0.15 QoE。
3. 帧级调度的真实收益集中在低带宽轨迹（3G +1.3~+4.3），4G 上与 BONES 无实质差别；4g_car −0.15 为唯一反例。
4. 回归使用 `imdn_bbb_3060ti.json` 延迟表（旧硬件），绝对卡顿（0914 292s）不代表 RTX 5080，仅横向比较有效；质量侧为旧 mp4v 口径，对外质量数字须等帧级指标 A 阶段连续 rawvideo 重测。

## 6. 踩坑铁律（本工作线新增）

1. 段级可预测不等于帧级可预测：聚合特征的高相关不能外推到段内帧排序，必须在目标粒度上直接验证。
2. 排序器的部署门槛是后悔值与高估率，不是拟合 R²；近随机排序 + 高后悔时直接退回无预测的经典 EDF。
3. report 中跨控制器的比例指标必须共用同一分母口径（队列 summary），否则会出现假差异。

## 7. 上下游关系

- 上游：#4 gain_predictor（段级预测器证伪与"不准的预测更有害"教训）；M3/M4（队列与三控制器，07d9514）；`data/bbb/per_frame_quality.json`。
- 同层：#13 sr_aware_alloc（增益空间均匀的独立证据）。
- 下游：帧级指标体系 A 阶段（连续 rawvideo 重测质量口径）；M5 之后帧调度复活簇 M0–M5 全部闭合。
