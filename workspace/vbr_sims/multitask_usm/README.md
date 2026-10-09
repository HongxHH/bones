# 档案：多任务增强（SR + 去模糊/锐化 + 插帧）

**状态：已封板（证伪；离线质量表已作废）**
**时间：2026-08-28 ~ 08-30**
**最近复核：2026-10-09（全量读码 + 结果 json 现算）**

## 1. 要回答的问题

BONES 的增强维度只有"超分档位 l0~l3"。本工作线把该维度重新解释为**任务组合 ID**（仅 SR / SR+去模糊 / 仅去模糊 / SR+去模糊+插帧），验证两件事：

1. 解码后在端侧串联去模糊（NAFNet）、锐化（Unsharp Mask）、插帧（RIFE，未实现）能否带来 VMAF 增益，尤其在 720p/1080p 这种 BONES 不做 SR 的档位；
2. 多任务组合作为 BONES 联合优化的新动作空间，仿真器和真实播放器能否选出有效组合。

## 2. 方法概述

两条链路并行：

- **离线链路**：从 `data/bbb/imdn_bbb_3060ti.json`（段级 SR 质量表）出发，生成多任务质量表 `data/bbb/multitask_bbb_rtx5080.json`，喂给 BONES_MT 仿真器（`control/bones_mt.py`，本档案范围外）做组合调度。
- **实测链路**：抽 BBB 5 个 4 秒段（seg 10/40/70/100/130），对 480p/1080p（USM 扩到 360/480/720/1080）跑 NAFNet、传统方法、USM，ffmpeg/libvmaf 测增益；真实播放器侧实现 `MultiTaskEnhancer`（SR 三档 + USM-only 共 5 个任务）。

## 3. 文件清单

脚本目录 `workspace/vbr_sims/multitask_usm/`：

| 文件 | 作用 | 复核定性 |
|---|---|---|
| `generate_multitask_enhance.py` | 生成多任务质量表（延迟实测、质量估算） | **作废**：增益随机估算 + 数组索引转置，见 §6 |
| `validate_deblur_vmaf.py` | NAFNet width16 五段 VMAF 验证 | **作废**：模型未加载权重，见 §6 |
| `validate_deblur_traditional.py` | unsharp / wiener 五段实测 | unsharp 数字可参考（口径有偏）；wiener 实现错误 |
| `check_nafnet_weights.py` | 排查 NAFNet width32 权重键名 | 一次性调试脚本 |
| `validate_nafnet32_vmaf.py` | NAFNet-GoPro-width32（正确加载）五段实测 | 有效：证明运动去模糊模型不适用压缩损伤 |
| `measure_usm_full_resolution.py` | USM 四分辨率增益 + CPU 延迟实测 | 本工作线最完整的实测；口径不对称，见 §6 |
| `measure_usm_vmaf.py` | torch 版 USM（对齐播放器实现）20 段抽样 | **作废**：非相邻帧打包；输出文件已不存在 |
| `inspect_multitask_data.py` | 查看 multitask json 结构 | 一次性调试脚本 |
| `update_multitask_usm.py` | 用 USM 数字回填质量表（v1） | **作废**：分辨率行错位 |
| `update_multitask_usm_v2.py` | v2（修 FI 增益、加裁剪，先备份 .bak） | **作废**：在 v1 污染数据上再计算 |
| `generate_pure_vbr_enhance.py` | 生成"只改 VMAF 不改码率"的 VBR 质量表 | 属 VBR 主线交叉件，见档案 `vbr_sims/vbr/` |

跨目录认领：`vbr_sims/_misc/validate_compression_restoration.py`（传统压缩损伤恢复 5 方法，产物在本工作线输出目录）。

产物目录 `workspace/deblur_vmaf_test/`（105 mp4 + 5 json，mp4 未入 git）：

| json | 内容 |
|---|---|
| `vmaf_gain_results.json` | NAFNet width16（随机权重，作废） |
| `vmaf_gain_nafnet32_results.json` | NAFNet-GoPro-width32 实测 |
| `vmaf_gain_traditional_results.json` | unsharp / wiener 实测 |
| `vmaf_gain_compression_restoration_results.json` | unsharp / bilateral / nlmeans / deblock 组合实测 |
| `usm_full_resolution_results.json` | USM 四分辨率增益 + 延迟 |

播放器/仿真器侧（范围外，仅指针）：

- `player/istream_player/modules/enhancer/enhancer_multitask.py`（677 行，仍注册于 `module_composer.py:41,463`，可选未默认启用）；
- `player/scripts/run_multitask_comparison.py`；11 次运行输出 `player/output/data/mt_bones_multitask_*.json`（未入 git）；
- `control/bones_mt.py`（BONES_MT 仿真器）；
- 质量表 `data/bbb/multitask_bbb_rtx5080.json`（195.8 KB，**内容作废但仍在磁盘**）。

## 4. 复现方式

```powershell
cd E:\dev\Code\Python\bones\workspace\vbr_sims\multitask_usm
E:\dev\env\miniconda3\envs\SR\python.exe measure_usm_full_resolution.py   # 约几分钟
E:\dev\env\miniconda3\envs\SR\python.exe validate_nafnet32_vmaf.py        # 需 GoPro width32 权重
```

NAFNet 代码与权重：`enhance/NAFNet`（basicsr，`NAFNet-GoPro-width32.pth`）。输入视频 `data/video_mp4box/output_{360,480,720,1080}p.mp4`。脚本跳过已存在 mp4，重跑需先清 `deblur_vmaf_test/`。

## 5. 结论与关键数字（2026-08-28 实测，5 段，BBB 30fps，2026-10-09 从 json 现算）

**USM（OpenCV 实现，amount=1.5/radius=3；处理臂经 cv2 mp4v 重编码）**：

| 输入档 | VMAF 增益均值 | 区间 | 正增益段 | CPU 单帧延迟 |
|---|---|---|---|---|
| 360p | +14.68 | [+5.36, +21.44] | 5/5 | 0.325 ms |
| 480p | +6.80 | [−6.36, +16.89] | 4/5 | 0.830 ms |
| 720p | −1.55 | [−22.40, +9.15] | 3/5 | 1.503 ms |
| 1080p | −5.06 | [−27.21, +1.91] | 3/5 | 3.264 ms |

规律：USM 只在低分辨率（bicubic 放大后偏糊）有正增益，原生 1080p 上过锐化有害，且段间方差极大。

**NAFNet（5 段×2 档，10 组全部负增益）**：

| 模型 | 480p | 1080p |
|---|---|---|
| NAFNet width16（未加载权重，作废） | −29.12 | −27.48 |
| NAFNet-GoPro-width32（正确加载） | −32.44 | −27.70 |

width32 是有效结论：**运动去模糊预训练模型作用于压缩损伤属于域不匹配，平滑掉真实细节，VMAF 巨幅下降**。

**传统压缩恢复（480p，5 段均值）**：unsharp +7.17（4/5）≈ deblock+unsharp +6.10（3/5）> nlmeans+unsharp +0.33（2/5）> bilateral+unsharp −3.94 > bilateral −37.73；wiener（箱型 PSF 的 FFT 近似）−51.7，实现本身不成立。

**真实播放器（trace0000，10/11 次运行有效，2026-08-28~29）**：QoE（avg_vmaf−osc−stall/10 现算）87.9~90.1；绝大多数段选任务 0（不增强），USM-only（档位 4）仅最后 3 次运行出现、3~7 段；本批文件内无同条件 imdn_always 对照。

**"高带宽下增强不触发"根因**：播放器质量表写死，不是运行时保守——`enhancer_multitask.py:588-590` 720p 的 SR 三格为 −inf，`:595-598` 1080p 全格为 −inf（1080p USM 也禁用）；`imdn_always` 表同构。

## 6. 复核记录（2026-10-09）

| # | 发现 | 严重度 | 位置 |
|---|---|---|---|
| A1 | 去模糊增益 0.5~4.0、插帧 +1.0 均为 `np.random.seed(seg)` 随机估算，SSIM/PSNR 按 VMAF 比例编造，非实测 | 致命 | `generate_multitask_enhance.py:66-87,169-172` |
| A2 | `enhanced_quality_vmaf` 实际形状 (level=3, bitrate=4, seg)，生成器按 `[bitrate, level]` 读取，sr_only 行整体错位（480p 行得到 53.98，低于其 base 76.80） | 致命 | `generate_multitask_enhance.py:129-141` |
| A3 | USM 回填脚本的分辨率键为 360/480/720/1080，数据行实际是 240/360/480/720，增益整体错位一行；v2 又在 v1 污染结果上计算 FI 增益。最终 deblur_only−base 差值为 [+2.52, −23.75, −1.55, −0.29]，与任何实测都对不上 | 致命 | `update_multitask_usm.py:16-22,78-100`；`update_multitask_usm_v2.py:17-22,63-90` |
| A4 | width16 NAFNet 构造后直接 eval，**没有 load_state_dict**，−29 分是随机网络的结果，不构成 NAFNet 证据 | 致命 | `validate_deblur_vmaf.py:45-55` |
| A5 | 测量口径不对称：原始臂是 ffmpeg crf0 抽片，处理臂额外经 cv2 mp4v 有损重编码；处理臂被多压一次，正增益偏保守、负增益不受影响 | 重要 | 各 validate 脚本的 `extract_clip` 与 `process_video*` |
| A6 | `measure_usm_vmaf.py` 每段用 linspace 抽 5 帧（0/30/60/90/119）再按连续帧写 mp4，非相邻帧制造虚假运动→时域掩蔽虚高（即后来定性的"虚高 20~28 分"铁律的同款 bug）；输出 json 已不存在 | 重要 | `measure_usm_vmaf.py:86-103,133-148` |
| A7 | USM 两实现不一致：实测脚本用 OpenCV `GaussianBlur(ksize=0, sigma=3)`，播放器用 torch 卷积 kernel=7/sigma=1.5，增益数字不能直接互用 | 重要 | `measure_usm_full_resolution.py:39-41`；`enhancer_multitask.py:63-64`（torch 版另见 `measure_usm_vmaf.py:50-70`） |
| A8 | 播放器 USM 增益硬编码（240/360/480/720 = +4.70/+5.20/+1.68/+0.70，两个标"估计"），既不加载 multitask json，也与 §5 实测（+14.68/+6.80/−1.55/−5.06）矛盾 | 重要 | `enhancer_multitask.py:562-598` |
| A9 | 样本量仅 5 段、单次测量、无多种子/多轨迹；NAFNet 推理无 tile 切分（1080p 整帧 FP16） | 轻微 | 各 validate 脚本 |

**处置**：

- `data/bbb/multitask_bbb_rtx5080.json` 整体作废，所有消费它的 BONES_MT 仿真结论（`lookahead/run_mt_comparison.py`、`run_mt_10traces.py`、`_misc/sweep_resource_params.py`、`test_mt_quick.py`）不可引用；建议文件改名加 `.deprecated` 后缀或在 `_metadata` 置废弃标记（待确认后执行，不删数据）。
- 真实播放器从未使用该 json（读硬编码表），故 11 次 mt_ 播放器运行不受 A1~A3 污染，但其质量表 USM 数字（A8）本身未被干净实测支撑。
- 可信结论仅三条：① GoPro 运动去模糊权重对压缩编码内容有害（width32 实测）；② USM 类锐化在低分辨率放大场景有正增益、720p 以上为负，但具体数值受 A5/A7 污染需重测；③ 多任务组合在真实播放器里几乎不被选中，且高档位增强格被 −inf 写死。

## 7. 影响面：历史结论逐条重新定性

| 历史结论/数字 | 出处 | 复核后定性 |
|---|---|---|
| BONES_MT 仿真器多任务对比（MT vs BONES/CA-BONES、10 轨迹、资源参数扫描） | `lookahead/run_mt_comparison.py`、`run_mt_10traces.py`、`_misc/sweep_resource_params.py`、`test_mt_quick.py` | **推翻**。唯一输入 `multitask_bbb_rtx5080.json` 由 A1~A3 污染，结论无意义；`control/bones_mt.py` 机制代码本身未发现问题，坏的是数据 |
| 播放器 11 次 mt 运行 QoE 87.9~90.1 | `player/output/data/mt_bones_multitask_*.json` | 数字真实但**不能用于评价多任务优劣**：USM 增益硬编码且与实测矛盾（A8）、高档位格 −inf、本批无同条件 imdn_always 对照、USM 档仅 3~7 段被选中 |
| "NAFNet 去模糊 −27~−32，有害" | width32 实测 | **保留**。width32 权重正确加载，且 A5 的编码不对称只压低正增益，负增益方向不受影响；width16 的 −29 证据删除（A4） |
| USM 增益 +14.68/+6.80/−1.55/−5.06 | `usm_full_resolution_results.json` | **只保留方向**（低分辨率正、高分辨率负），数值挂起：A5 双臂不对称、A7 实现不一致、5 段小样本 |
| USM 延迟 0.325/0.83/1.50/3.26 ms（CPU） | 同上 | **保留**，`time.perf_counter` 实测，不受口径问题影响 |
| NAFNet 延迟 22~28 ms/帧 | `generate_multitask_enhance.py` 延迟表 | **保留**（实测部分）；30 fps 下 720p/1080p 已接近或超过 33 ms 预算 |
| "高带宽（hd_fs 15.7Mbps）下增强从不触发" | 8 月底播放器观察 | **保留，根因升级为代码事实**：质量表 720p SR 格、1080p 全格写死 −inf（`enhancer_multitask.py:588-598`），与运行时决策无关 |
| 插帧（FI）组合 +1.0 VMAF 及任何相关说法 | 质量表 combo3 | **标记"未做"**：FI 从未实现，数字为估算 |
| SSIM/PSNR 多任务数字 | multitask json | **作废**（A1 比例缩放编造） |

## 8. 是否重做：判断与方案

**判断：不按原样重做，当前挂起。** 多任务"任务组合作为调度维度"的机制没有被证伪，但它成立的前提是存在**第二个真实有效的轻量增强算子**。现状是：SR 已占住低分辨率档位；USM 只在 SR 本来就强的低分辨率有效、高分辨率为负；NAFNet 类运动去模糊有害；FI 未做。第二算子缺位，组合动作空间是空的，重建质量表只会得到"算法永远不选新组合"的同一个结果。

重做需要两个前置条件，均不在本工作线内：

1. **统一的帧级测量管道**（暂停中的帧级指标体系 A 阶段）：rawvideo 连续帧、双臂编码对称、torch 算子与播放器实现对齐。本工作线 A5/A6/A7 三个口径问题都由它一次性解决，不应再为多任务单独搭测量脚本。
2. **档案 5（enhancer_survey）算子普查复核**：STDF 等为压缩损伤设计的轻量模型当时是否已给出 NO-GO。若已证明无现成轻量算子在压缩视频上稳定有效，本线永久封板；若有候选（去块/去振铃类，非运动去模糊），按下述步骤重做。

前置满足后的重做步骤（按顺序，每步设 GO/NO-GO 门）：

1. **算子小样本判决**：5~10 段、四分辨率，统一 rawvideo 口径，VMAF + LPIPS 双指标，延迟必须 <33 ms（720p/30fps）；门槛定为至少两个分辨率稳定 ≥+1 VMAF 且 LPIPS 不恶化。
2. **逐段实测质量表**：全部数字来自步骤 1 同口径实测，禁止估算/随机数；表轴标签（240/360/480/720/1080）写断言单测；json `_metadata` 必须记录模型权重、口径、生成脚本与时间。
3. **BONES_MT 仿真重跑**：换入新表后重跑多轨迹对比，再决定是否进播放器。
4. **播放器开放高档位格**：仅当步骤 1 证明某算子在 720p/1080p 有效，才把 `enhancer_multitask.py:588-598` 的 −inf 格替换为实测值；−inf 不得凭估计开放。

质量表防呆（供步骤 2 与未来所有质量表共用）：生成脚本禁止 import random/np.random 造指标；多维数组读写处打印形状并断言；回填类脚本必须按分辨率名（而非行号）索引；json 入库前与源表逐行对账。

## 9. 踩坑铁律（本工作线新增）

1. **模型验证脚本必须确认权重真的加载了**——A4 的随机网络跑出"合理范围"的灾难性负数，差点被当成 NAFNet 结论。
2. **质量表生成禁止用随机数/比例缩放编造指标**；只要写进 json 就会被下游仿真当真。
3. **跨分辨率回填逐行核对轴标签**：BONES 表行是 240/360/480/720，不是 360/480/720/1080（A3）。
4. **多维数组索引先打印形状再写下标**，注意 `[enhance_level][bitrate][seg]` 与 `[bitrate][enhance_level]` 的区别（A2）。
5. 测量双臂编码路径必须对称：处理臂和对照臂要么都走同一重编码，要么都走 rawvideo（A5）。

## 10. 上下游关系

- 上游：BONES 段级质量表 `imdn_bbb_3060ti.json` / `imdn_bbb_quality.json`；内容特征 `data/bbb/content_features_bbb.json`（A6 脚本依赖，文件在）。
- 下游：`MultiTaskEnhancer` 仍在播放器可选模块中；本工作线"高分辨率档增强格 −inf"的发现是后续"突破 1080p 增强上限"讨论的起点；STDF 等专用压缩恢复模型的尝试见 `enhancer_survey/`（档案 5）与未入 git 的 `enhance/stdf-pytorch/`。
- 与 VBR 主线交叉：`generate_pure_vbr_enhance.py` 服务于 VBR 集成实验，归档案 7。

## 11. 复盘备注（复活检查清单）

重做决策与步骤见 §8，此处只留复查时要逐条核对的事实：

- **多任务调度思想本身未被证伪**，被证伪的是具体算子（NAFNet-GoPro、wiener）和编造的数据；若未来引入为压缩损伤训练的轻量恢复模型（去块/去振铃，非运动去模糊），任务组合维度可以重建，但质量表必须用统一 rawvideo 口径实测。
- USM 低分辨率正增益若要重新采信，纳入帧级指标体系（暂停中的张量重测任务）一并做：rawvideo 连续帧、双臂对称、torch 实现对齐播放器。
- 算力约束：NAFNet width32 在 1080p 约 6~17 FPS（历史记录），30 fps 实时预算下本就不可行；任何复活方案先过延迟关。
- FI（插帧）组合从未实现，`sr_deblur_fi` 列纯为估算，引用时按"未做"处理。
- `MultiTaskEnhancer` 仍在播放器注册可选、默认不启用，代码保留不删；`data/bbb/multitask_bbb_rtx5080.json` 建议改名 `.deprecated`（待确认，不删数据）。
