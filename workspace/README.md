# workspace 索引

本目录按研究方向组织。每个子目录对应 docs/ 下一份文档，复盘时先看这里定位，再去 docs 看结论。

## 当前主线

| 目录 | 做的什么 | 状态 | 结论 | 对应 docs |
|---|---|---|---|---|
| `pacnet/` | PAC-Net 可渐进增强网 + 后续所有探针 | 主线 | 见下方子表 | `pacnet_progressive_enhancer.md` |
| `pacnet/modern_sr/` | 1080p→4K x2 零样本后处理 SR 探针 | 已封板 | NO-GO：VMAF 中位 +1.2、LPIPS 全段恶化 | `modern_sr_probe.md` |
| `pacnet/side_channel/` | 高频先验旁路信息论上界 oracle | 已封板 | NO-GO：+2 VMAF 需旁路 30% 主码流，非 1~2% | `side_channel_probe.md` |
| `pacnet/h1_condnet/` | H1 codec 元数据（QP+MV）条件网 | 已封板 | NO-GO：真条件≈乱序条件，ΔVMAF≤0.05 | `h1_condnet_falsification.md` |
| `pacnet/h2_warp_oracle/` | H2 DIS 光流 warp 邻帧融合 | 已封板 | NO-GO：7 档全不优于单帧 l2 | `h2_warp_oracle.md` |
| `pacnet/roi_tile/` | H3 选择性原生分辨率 tile | 已封板 | NO-GO：216 点全被均匀提分辨率支配 | `h3_roi_tile_oracle.md` |
| `pacnet/sr_aware_alloc/` | SR 感知逐段码率分配 LP oracle | 已封板 | NO-GO：TOS+BBB 跨内容，新颖性≤0.4 | `sr_aware_alloc_falsification.md` |
| `pacnet/multi_k/` `pacnet/sweep_thresh/` `pacnet/multi_trace_logs/` | PAC-Net K=1.5 多轨迹、安全系数扫描 | 数据留存 | halt 对 IMDN 卡顿优势不稳定（e1 17.5ms vs l2 7.7ms 结构性） | `pacnet_progressive_enhancer.md` |
| `pacnet/hard_survey/` | 7 档退化阶梯逐帧 VMAF/LPIPS 测量 | 数据留存 | 所有离线探针的统一退化基准 | — |
| `pacnet/visual_compare/` | 增强前后帧对比 PNG | 参考图 | 字幕/纹理被超分成假条纹等证据 | — |
| `pacnet/weights/` | IMDN/PAC-Net 训练权重 | 留存 | — | — |
| `pacnet/scripts/` | PAC-Net 训练/测量/评估脚本（原散在根目录） | 代码 | train_pacnet.py、run_stage*.py、measure_*.py | — |
| `pacnet/logs/` | 所有后台 worker 的 .err/.log/.pid | 存档 | 复现/排错用 | — |

## 8 月仿真器时代（vbr_sims/）

| 目录 | 做的什么 | 状态 | 结论 |
|---|---|---|---|
| `vbr_sims/vbr/` | 服务端 VBR、CBR、平方根、四阶梯、连续/阶梯/sigmoid 码率 | 已封板 | 历史"+1.7 QoE"是抽样帧打包虚高+多 16.6% 码率撑出 |
| `vbr_sims/lookahead/` | 多段前瞻 MPC、四方案/三方案/10 轨迹对比 | 数据留存 | 修 bytes/Kb bug 后 6 轨迹 4 胜 2 负 |
| `vbr_sims/frame_level/` | 帧级 DP 调度、逐帧 VMAF 测量 | 已封板 | 3G +0.83，预算充足时与固定档位等价 |
| `vbr_sims/md_qoe/` | 自定义 MD-QoE 指标 | 设计稿 | 未落地 |
| `vbr_sims/multitask_usm/` | 多任务增强管线、USM 锐化、NAFNet 去模糊 | 已封板 | 1080p 下 USM 无增益；hd_fs 带宽下增强不触发 |
| `vbr_sims/_misc/` | 内容特征分析、VMAF 预测器训练、特征提取等一次性脚本 | 留存 | — |

## 其他

| 目录 | 做的什么 | 状态 |
|---|---|---|
| `deblur_vmaf_test/` | 早期去模糊模型 VMAF 增益测量 | 已封板 |
| `gain_predictor/` | GBDT/RF VMAF 增益预测器（含 .pkl） | 已封板：不准的预测比没预测更有害 |
| `enhancer_survey/` | 增强算子普查（NAFNet/STDF/NanoVSR 现成算子对比） | 已封板：同口径下无现成算子占优 |

## 复现入口

- SR 环境：`E:\dev\env\miniconda3\envs\SR\python.exe`
- 连续时间序 libvmaf 铁律：禁止非相邻抽样帧打包成 clip（虚高 20~28 分）
- 线程：OMP=4、libvmaf n_threads=2、单 worker 串行（双 worker 曾打满 16 核卡死桌面）
- 大 mp4/tmp/日志均未入 git，可再生成；代码与结果 json 已入库
