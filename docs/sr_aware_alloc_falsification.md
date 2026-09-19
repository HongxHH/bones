# SR 感知率失真码率分配证伪（2026-09-20，NO-GO）

## 假设

把超分网络按内容的恢复能力纳入码率分配的率失真目标：难恢复段多给码率、SR 能补回的段少给，最小化端到端"超分后失真"。若相对等码率 CBR 与平方根 VBR 有稳定 ≥1 VMAF 增益则 GO。脚本与数据：`workspace/pacnet/sr_aware_alloc/`（`gen_crf_ladder.ps1`、`measure_seg_rd.py`、`alloc_compare.py`、`seg_rd.json`、`alloc_results.json`、`alloc_pareto.png`）。

## 方法

- TOS 1920×800@24fps，426 宽（426×178，bicubic 下采样），libx264 CRF 22/26/30/34/38/42，强制 96 帧 GOP、`scenecut=0`，段间独立（`gen_crf_ladder.ps1`）；4 秒段 96 帧，共 183 整段。
- 逐段 R-D：ffprobe 逐帧 `pkt_size` 按段汇总比特；逐帧 l2（IMDN_RTC medium，x4 后 bicubic 对齐 1920×800）与 bicubic 直拉，经 rawvideo 管道送入单次 ffmpeg 双 libvmaf（`measure_seg_rd.py:96-118`），逐帧分按段聚合。**连续时间序打分**，不用抽样帧相邻打包（后者虚高，见 known_issues"抽样帧相邻打包 VMAF 虚高"条）。
- 五种策略严格等总码率（预算 = 恒定 CRF 总比特，5 个预算点；触界段固定后再分配，恰花完预算，`alloc_compare.py:64-86`）：恒定 CRF、等比特 CBR、等预算平方根（w_i=CRF30 逐段比特）、原始 R-D oracle（最大化 bicubic VMAF，凹包络+全局边际斜率注水 `alloc_compare.py:89-114`）、SR 感知 oracle（最大化 post-SR VMAF）。所有策略最终都用 post-SR VMAF 评估。

## 结果（post-SR VMAF，等总码率）

| 预算 | 恒定 CRF | 等比特 CBR | 平方根 | 原始 RD oracle | SR 感知 oracle | SR oracle−CBR | SR−原始 oracle |
|---|---|---|---|---|---|---|---|
| 34 kbps (CRF42) | 7.40 | 7.85 | 7.52 | 7.40 | 7.40 | −0.45 | +0.00 |
| 52 kbps (CRF38) | 15.43 | 18.44 | 17.16 | 19.43 | 19.58 | +1.14 | +0.15 |
| 82 kbps (CRF34) | 26.21 | 28.95 | 27.93 | 29.78 | 30.06 | +1.11 | +0.28 |
| 131 kbps (CRF30) | 38.48 | 40.17 | 39.70 | 40.92 | 41.10 | +0.93 | +0.18 |
| 214 kbps (CRF26) | 50.71 | 51.39 | 51.26 | 52.10 | 52.14 | +0.75 | +0.04 |

实际花费/预算均为 1.000（端点 CRF42 1.004，插值舍入）。SR oracle 与原始 oracle 的逐段分配平均差 8.5%，质量结果只差 0.04~0.28（最优解平坦）。

## 判决：NO-GO

1. **"SR 感知"新颖性为零**：以超分后失真为目标相对以原始失真为目标，post-SR VMAF 只多 0.00~0.28。l2 对各段的恢复近于均匀缩放，不重排"哪段缺码率"的边际排序，把 SR 写进率失真目标不改变最优分配。
2. **内容自适应分配的总天花板也不够**：上帝视角 oracle 对等比特 CBR 仅 +0.75~+1.14（最低预算反负），且这是需要逐段多 CRF 编码的两趟上限；可部署预测器只能吃到一部分，稳定 ≥1 不成立。
3. **平方根启发式在等码率下不成立**：5 个预算中 4 个输给等比特 CBR（−0.13~−1.28）。历史"平方根 +1.70 QoE / 100% 胜率"（`server_side_vbr_coding.md`）由两个混淆因素撑出：平均多给 16.6% 码率（`generate_sqrt_vbr_all_resolutions.py:70-78`，非等预算）与抽样帧打包 VMAF 虚高（见 known_issues）。
4. 恒定 CRF（恒定质量）在低码率明显劣于等比特 CBR（CRF34 −2.74），CRF 的 PSNR 型质量恒定不等于 VMAF-码率最优；但这是经典 per-segment VBR 范畴，不构成 SR 方向的论文点。

## 边界

- 单视频（TOS）、单分辨率（426 宽）、仅 VMAF、6 个 CRF 离散点、oracle 为连续插值上界。SR−原始 oracle 差在全部 5 个预算都 ≤0.28，机制性强，补 320/640 或 LPIPS 反转可能性低，但未实测。
- 未否证"服务端两趟逐段 VBR"这一经典系统改进本身（上限约 +1 VMAF），只是否证其 SR 感知变体与 ≥1 的稳定增益。
