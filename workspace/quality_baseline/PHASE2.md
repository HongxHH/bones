# 阶段 2：选定 (K, C, F)

**状态：GO（2026-10-27）**  
产物：`phase2_select_cf.json`；测量脚本：`measure_latency_phase2.py`

## 要回答的问题

在 RTX 5080 上，能否找到 burn 系数 K、承诺档 C、回退档 F，使得：

1. 加压后 C 整段耗时 > 4s（120 帧），F 整段耗时 < 4s；
2. 在 3G 会下到的分辨率上，F 相对 bicubic 的全片 VMAF 增益 ≥ 2。

## 测量口径

- 路径：`forward + interpolate(1080) + uint8 + cpu`（对齐 `enhance_one_frame`）
- burn：只乘在 forward 上，`T(K)=K·fwd_ms+post_ms`
- 每格 warmup 5 + measure 30；随机 LR 张量

## 主分辨率（历史 always_greedy × 0913）

同一轨迹两次完整结果不一致：一次 480p 主导（154/159），一次 360p 主导（86/159）。阶段 3 不绑定单一 download_action，**(K,C,F) 须覆盖 240/360/480**。

## 质量间隙（阶段 1 全片均值）

| 输入 | sr_low − bicubic |
|---|---:|
| 240p | +12.4 |
| 360p | +8.4 |
| 480p | +6.5 |

三档均 ≥ 2 → F=1（low）画质门槛通过。

## 耗时要点（K=1）

| | L1 low | L2 medium | L3 high |
|---|---:|---:|---:|
| 240p seg | 0.26s | 0.34s | **6.12s** |
| 360p seg | 0.24s | 0.40s | 1.02s |
| 480p seg | 0.28s | 0.62s | 2.38s |

5080 上 **medium 远低于实时**；要把 C=2 压过 4s，K 需约 8–20，不合理。high 在 240p 上 K=1 已超预算。

## 推荐工作点（阶段 3 采用）

| 参数 | 值 |
|---|---|
| **K** | **5.0**（`PACNET_BURN_K=5`） |
| **C** | **3**（IMDN high） |
| **F** | **1**（IMDN_RTC low） |

- 三分辨率上 C=3 均超 4s、F=1 均远低于 4s（见 json `feasible_picks`）。
- `always_greedy` 默认 `enhance_action=2`，阶段 3 **必须改为 3**，否则承诺档与测量不一致。
- burn 探针（修好单次 init 后）K=1.5 实测/预测 ≈ 0.93，乘法模型可用。

## 明确不采用

- C=2 + 极大 K：能在纸上造出短缺，但 burn 过重、与 PAC 时期 K=1.5 叙事差太远。
- 只认 240p + C=3 + K=1：自然短缺成立，但 0913 并不总停在 240p。

## 阶段 3 入口条件

1. 播放器可配置承诺增强级别（C=3）。
2. 恢复执行器支持：蛮干 / bicubic 尾 / 专用 F 尾 / 整段 F。
3. 环境变量 `PACNET_BURN_K=5`；`latency_adapt` 关闭。
