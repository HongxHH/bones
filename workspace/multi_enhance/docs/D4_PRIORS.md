# D4 前置：TOS 全量先验（对齐口径）

**状态：先验完成（2026-10-28）**  
**原则：** 禁手填；与播放器 `get_resolution_set` / `enhance_one_frame` 对齐；**不**复用旧 `imdn_tos_*`。

## TOS 几何

| 行 / 名义 | 实际 WxH | display 1920×800 下 scale | 合法 tool |
|---|---|---:|---|
| 0 / 240p | 426×176 | **4** | none, SR L/M/H |
| 1 / 360p | 640×268 | **2**（非 BBB 的 3） | none, SR L/M/H |
| 2 / 480p | 854×356 | **2** | none, SR L/M/H |
| 3 / 720p | 1280×532 | 1 | none, sr_x2_med, restore |
| 4 / 1080p | 1920×800 | 1 | none, restore（Q 仍 −inf，未测 native） |

## 产出（已落地）

| 文件 | 内容 |
|---|---|
| `data/quality/tos_{240,360,480,720}.json` | 全片 17620 帧 VMAF |
| `workspace/hd_enhance_probe/h0_tos_segs_20_…_140.json` | 10 段 H0；restore Δ≈**+4.16**，x2_med Δ≈**+1.76** → GO |
| `player/multitask_tos_quality.json` | 5×6 Q |
| `player/multitask_latency_tos.json` | 5×6 T（800p 显示） |
| `workspace/multi_enhance/priors/meta_tos.json` | 元数据 |

### 全片均值（对照旧表）

| 档 | bicubic | sr_high | 旧 imdn_tos bicubic |
|---|---:|---:|---:|
| 240 | 48.3 | 69.0 | ~80.4 |
| 360 | 72.4 | 85.4 | ~94.3 |
| 480 | 83.6 | 91.9 | ~97.9 |
| 720 | 93.0 | — | ~99.8 |

## 汇表

```text
python scripts/build_priors.py --dataset tos
```

D4 冒烟结果见 [`D4.md`](D4.md)（已 GO）。
