# 高清增强探针（H0）

**状态：H0 扩样本完成（2026-10-27）→ 见文末结论**  
**问题：** 在 1080p 显示尺子下，720p 输入是否存在「超分以外 / 轻度超分」相对 bicubic 的稳定正增益？

BBB 片头白场 **禁止 seg0**；默认 `--smoke` → seg 40。

## 扩样本（10 内容段，墙钟 312s）

段号：`20,30,40,50,60,80,90,100,120,140`  
权威：`h0_720_segs_20_30_40_50_60_80_90_100_120_140.json`

| 臂 | 跨段平均 Δ | 最小 Δ | Δ>0 | Δ≥+1 |
|---|---:|---:|---:|---:|
| restore_unsharp | **+5.13** | +1.47 | 10/10 | **10/10** |
| sr_x2_medium | **+2.38** | +0.67 | 10/10 | **9/10** |
| sr_x2_low | +1.33 | −0.15 | 9/10 | 5/10 |

（此前 3 段冒烟 `h0_720_segs_40_70_100.json` 方向一致，可作对照。）

## 结论（H0）

1. **高清「第二工具」成立（有条件 GO）**：720→1080 上，相对 bicubic，存在稳定正增益选项。  
2. **菜单候选**：`restore_unsharp`（增益最大但 VMAF 偏爱锐化，须目视/LPIPS 复核后再当正式工具）；`sr_x2_medium`（更可信的轻 SR，9/10 段 ≥+1、无负段）。  
3. **`sr_x2_low` 不进默认菜单**：均值刚过线、半数段 <+1、出现微负。  
4. **seg0 作废教训保留**：片头白场曾把 SR×2 判成有害，内容段上相反。

## 下游

系统设计已转到 [`../multi_enhance/DESIGN.md`](../multi_enhance/DESIGN.md)（多类型工具注册表 + 段前可行性调度，松耦合 ABR）。本目录 H0 结果作为 `sr_x2_med` / `restore_sharp` 入表证据。

借鉴 `vbr_sims/multitask_usm` 的问题意识（高清段曾被 −inf 堵死、需要第二算子），**不继承**其质量表与调度结论；测量口径对齐 `quality_baseline`（rawvideo / trim 锁帧 / libvmaf）。

## 臂（每段只比这些，不级联）

| 臂 | 做法 |
|---|---|
| `bicubic` | ffmpeg YUV `scale=1920:1080:flags=bicubic`（与 Phase1 同） |
| `restore_unsharp` | 同上放大后 `unsharp`（轻量压缩向「锐化/去糊」代理，非 DL） |
| `sr_x2_low` | IMDN_RTC low ×2（720→1440）再 bicubic 对齐 1080 |
| `sr_x2_medium` | IMDN_RTC medium ×2，同上 |

## 门禁

- 冒烟：首段 120 帧。  
- **GO（值得扩样本）**：`restore_*` 或任一 `sr_x2_*` 相对 `bicubic` 均值 **≥ +1.0 VMAF**。  
- **NO-GO**：皆 < +1 或明显变差 → 高清多增强先停，主线转低清段前选档。  
- 扩样本（仅 GO 后）：5～10 段均匀抽样；再谈全片与策略系统。

## 用法

```text
cd workspace/hd_enhance_probe
set BONES_FFMPEG=E:\dev\tool\ffmpeg-6.1.1-full_build-shared\bin\ffmpeg.exe
# 默认 seg=40（跳过片头）；或显式多段：
E:\dev\env\miniconda3\envs\SR\python.exe measure_h0_720.py --smoke
E:\dev\env\miniconda3\envs\SR\python.exe measure_h0_720.py --segs 40,70,100
```

产物：`h0_720_segs_*.json`；`tmp/` 可删。

## 中期方向（本探针之后）

若 GO：工具注册表 + **段前可行性调度**（不用 BONES），动作 `none | restore | light_sr`。  
若 NO-GO：本目录封板，不开放播放器高清增强格。
