# H1：codec 元数据条件化增强——条件网离线证伪

## 定位

数据通路确认 QP/运动矢量可在解码侧零带宽取得（见 `h1_codec_metadata.md`）后，本文件记录第二步：把这些码流信息作为条件喂给单个 l2 规模轻量网（IMDN_RTC，nf=12/modules=5，24114 参数），检验"喂码流信息能否提升低码率增强"。脚本与产物在 `workspace/pacnet/h1_condnet/`。

假设 H1：在不增大网络、不增加带宽的前提下，把 QP（哪里压得狠）+ 前/后向运动矢量（往哪运动）作为条件，轻量网能在低码率段恢复得更好。

## 实验设计：三臂对照

同一批数据、同一退化、同一优化预算、同一初始化，唯一差别是条件输入：

- **base**：IMDN_RTC，不喂条件；
- **cond**：条件网，喂与当前帧严格对齐的真 QP/MV（7 通道，`codec_lib.py:33` NCOND=7：QP 1 + 前向 mvx/mvy/cover + 后向 mvx/mvy/cover）；
- **shuf**：与 cond 同结构、同初始化，但喂**别的帧**的乱序条件（只在同分辨率组内置换）。这是容量对照——若真条件不优于乱序条件，任何差异都只来自多出来的参数，而非元数据信息。

两种注入结构，用来排除"注入太弱"的假阴性：

- **add**（`cond_model.py:26` CondIMDN_RTC）：条件经两层 3×3 卷积（7→12→12）编码后，在首层特征上逐元素相加；参数 26190（+2076，+8.6%）。
- **cat**（`cond_model.py:54` CondIMDNCat）：条件作为额外 7 个输入通道拼到首层（首层 3→10 通道，前 3 通道拷贝热权重、多出 7 通道零初始化）；参数 24870（+756）。cat 三臂同构，base 恒喂零。零条件输出与热启动骨干**逐位相等（maxdiff 0.0，probe_cat.py）**。

数据：BBB（动画）训练，TOS（真人电影）全程留出做跨内容泛化。训练退化 6 个（426/320 宽 × CRF26/34/42，bicubic 下采样 + libx264），每退化独立随机取 180 帧共约 1080 帧（`build_cache.py:46`）；评测沿用 hard_survey 7 档退化 × 32 帧，统一 x4 后 bicubic 到 1920×800 对源算逐帧 VMAF（libvmaf）+ LPIPS（squeeze），对 cond−base 做逐帧配对双侧符号检验（`eval_cond.py:94`）。

## 三套协议结果（2026-09-19，TOS 留出）

| 协议 | 训练 L1（base/cond/shuf） | base 是否健康 | cond vs base | 判决 |
|---|---|---|---|---|
| warm + add，lr2e-4，30ep | 0.0343/0.0352/0.0352 | 否，微调破坏 DIV2K 泛化（anchor VMAF stock 88.92→base 79.76） | VMAF 7/7 档为负、多档 p<0.01；LPIPS 仅最重 crf42 档 cond 显著更好 | 域过拟合，不能下结论 |
| scratch + add，lr2e-4，100ep | 0.0380/0.0385/0.0386 | 否，单视频从零欠拟合（anchor base 59.72 < bicubic 76.85） | cond≈shuf≈base，条件无增益 | 欠拟合，不能下结论 |
| **warm + cat，lr2e-5，20ep（最健康对照）** | **0.03601/0.03600/0.03602（三者逐轮相同到 4 位小数）** | 基本健康，base 仅比 stock 低 2~3 分 | 见下表，**ΔVMAF 全在 ±0.05 内、ΔLPIPS≈0.001，cond≈shuf** | **条件无信息增益** |

cat 协议逐档（VMAF，stock/base/cond/shuf；末两列为 cond−base）：

| 退化 | stock | base | cond | shuf | ΔVMAF | ΔLPIPS(base−cond) |
|---|---|---|---|---|---|---|
| anchor 426/400k | 88.92 | 85.63 | 85.65 | 85.72 | +0.02 | +0.0008 |
| 426 crf26 | 78.25 | 75.82 | 75.85 | 75.88 | +0.03 | +0.0007 |
| 426 crf34 | 56.33 | 54.98 | 55.00 | 55.03 | +0.02 | +0.0010 |
| 426 crf42 | 38.36 | 37.66 | 37.64 | 37.66 | −0.02 | +0.0015 |
| 320 crf26 | 63.25 | 60.70 | 60.74 | 60.75 | +0.04 | +0.0007 |
| 320 crf34 | 44.29 | 43.04 | 43.06 | 43.07 | +0.01 | +0.0008 |
| 320 crf42 | 31.62 | 30.95 | 30.95 | 30.95 | +0.01 | +0.0013 |

ΔVMAF 最大 0.04（相对 stock−base 的 2~3 分差距是两个数量级以下的噪声）；ΔLPIPS≈0.001（LPIPS 量程 0.2~0.47，约 0.3%），且 shuf 与 cond 同向同幅（如 426crf42 LPIPS：base 0.3799、shuf 0.3798、cond 0.3814，乱序反而不差于真条件）。符号检验在若干档 p<0.01 只说明这 0.0x 量级的偏移方向稳定，不构成有意义的质量增益。

## 判决

**H1 的端侧便宜版证伪**：在"单个动画视频训练、不大规模多内容重训"的端侧约束下，无论条件在特征层相加（add）还是直接拼到输入层（cat，最强注入），无论微调轻重，真 QP/MV 条件相对无条件 / 乱序条件都**没有可测量的 VMAF 或 LPIPS 增益**；三臂训练 L1 逐轮相同，网络没有利用条件信息。

这与 PAC-Net 的教训一致：低码率增强的瓶颈不是"通用单帧轻量网缺一个条件输入"，把码流统计当作额外通道喂进去，网络不会自发学会使用。

**假阴性边界（失败是阶段性的，不是对整个 codec 元数据方向判死刑）**：

1. 文献中 codec 条件网（PnP-VCVE, CVPR'25 等）的增益依赖**大规模、多内容训练**；本实验只有单一动画视频训练、真人视频留出，warm 微调会域过拟合、scratch 会欠拟合，这本身就是端侧"不大规模重训"前提的约束体现。大规模多内容训练能否救回未在本实验检验。
2. 本实验把 MV 仅当作**额外输入通道**。MV 的真正价值可能在**显式时序配准/warping**（告诉网络从相邻帧哪个位置取像素），这需要一个对齐算子而非多几个输入通道——属于多帧/VSR 与 H2（anchor 稀疏帧 + 残差回加，NEMO 式）范畴，未被本实验证伪。
3. 残差标准解码器不导出（`h1_codec_metadata.md`），第一版未用；像素域高频/块效应图替代未试。

## 复现入口

- `codec_lib.py`：MV/QP 提取与 7 通道条件场（scan_mv `:57`、parse_qp `:86`、build_frame_cache `:112`）。
- `cond_model.py`：CondIMDN_RTC（add `:26`）、CondIMDNCat（cat `:54`），`python cond_model.py` / `probe_cat.py` 自检。
- `build_cache.py`：6 退化 × 180 帧训练缓存（cache/*.npz，gitignore，可再生成）。
- `train_cond.py --model {add,cat} --init {scratch,warm}`：三臂训练（train_arm `:112`）。
- `eval_cond.py --model {add,cat} --wdir <权重目录>`：TOS 7 档 VMAF+LPIPS+符号检验，落 `<wdir>_eval.json`。
- 结果：`h1_eval.json`（warm/add）、`weights_scratch_eval.json`（scratch/add）、`weights_cat_warm_eval.json`（warm/cat）。SR 解释器 `E:\dev\env\miniconda3\envs\SR\python.exe`。

## 下一步

条件"喂给单帧网"不成立，转向让码流信息**参与结构性运算**：H2 anchor 稀疏帧 + MV 引导残差回加 / 时序配准（MV 用来 warp 而非当通道），与视频超分主线汇合；大规模多内容训练作为资源允许时的复核项，不在端侧小样本主线上继续加码。
