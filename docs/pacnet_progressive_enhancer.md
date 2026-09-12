# PAC-Net 可渐进神经增强网络

面向流媒体"随机算力预算 + 未知压缩退化 + 内容难度不均"的端侧增强网络，与 BONES 选择层解耦的新主线。核心是**单套权重、算力可伸缩**：同一网络在第 2/4/6 个 block 后各设一个递进出口，时间够就深算、到点必出当前最优结果（anytime/可中断）；退化条件化模块让一套权重覆盖任意压缩程度。代码在 `workspace/pacnet/`。

## 与既有结论的关系

- 阶段17（见 `enhancer_operator_survey.md`）证明：对症增强增益严格集中在高运动/强压缩塌陷段，普通段持平。由此需要"困难段深算、简单段浅算、时间不够先出保命档"的单网络多档能力，而非多个固定模型切换。
- 文献空白：anytime/early-exit 动态深度几乎只用于分类/视觉推理，SR/增强侧未被开发；流媒体神经增强系统的创新都在网络外部选帧/选块/选模型，无人改网络内部使其算力可伸缩。

## 网络结构（`pacnet_model.py`）

- 共享骨干：`fea_conv` + 6 个串联 `IMDModule`（IMDB 本身残差），在第 2/4/6 个 block 后各接一个 `ExitHead`（1×1 融合 + LeakyReLU + 3×3 + 浅层残差 + pixelshuffle×4），记 e1/e2/e3。
- `forward(exit_idx=k)` 只跑到第 k 个出口即返回，真正省算力；`exit_idx=None` 返回三出口用于深监督训练。
- `CondEncoder`：从 LR 提 5 维在线统计（RGB 均值 3 + 高频能量 + 下采样残差能量），经 MLP 生成 FiLM gamma/beta 调制浅层特征，末层零初始化保证开局恒等、不破坏 warm。**阶段0先关闭，阶段2再启用验证跨退化。**
- `PlainIMDN(num_modules)`：参数化 block 数的标准 IMDN，用于造与 e1 同算力的固定浅模型对照。

## 阶段0最小证伪（纯离线，不上播放器）

用最小实验验证承重墙假设，数据管线复用 `workspace/enhancer_survey/train_leaveout.py`：真实 4.5× 裁 GT 对齐、uniform 16 段留出、DIV2K 权重 warm、Adam 2e-4、StepLR(15,0.5)、30ep、patch192、batch16、L1。测试为 16 留出段（含困难段 130/140/150）每段 4 帧共 64 帧整帧 PSNR（×4 后 bicubic 到 1080 对 GT）。脚本：`selftest_model.py` 结构自检、`train_pacnet.py` 训练、`eval_pacnet.py` 评测、`analyze_pacnet.py` 判决、`forward_equiv_check.py` 前向等价核验。

### 关键坑：warm 迁移键名（复用价值高，排查耗时最长）

播放器 IMDN 里 `conv_layer` 返回**裸 `nn.Conv2d`**（state_dict 键为 `fea_conv.weight`，无 `.conv`），`conv_block` 返回 **`Sequential`**（融合层键为 `c.0.weight`），`pixelshuffle_block` 内卷积键为 `upsampler.0.weight`。首版按"都带 `.conv`"臆测键名，导致**入口 `fea_conv` 与最深出口重建头全部漏迁、保持随机初始化**，只有 6 个 block 共 84 项迁移成功（恰好等于 84，造成"warm 成功"假象）。随机入口上 block 白算：e3 仅 24.2dB、三出口挤在一起、整体≈bicubic，一度误判 NO-GO。修正后 92 项权重全迁，`forward_equiv_check.py` 证实 PAC e3 与标准 IMDN 同权重下**逐元素 diff=0**。另修两处：`ExitHead.fuse` 必须用带 LeakyReLU 的 `conv_block` 对齐 IMDN 的 `c`（用无激活 `conv_layer` 会让 warm 失配）；`CondEncoder` 不得用 batch 维标准化（单帧推理 batch=1 时统计恒 0、条件机制失效），改固定尺度。

### 结果（16 段 64 帧整帧 PSNR，dB；主版出口损失权重 0.3/0.6/1.0、单调 margin 0.5）

| 方法 | 均值 | 说明 |
|---|---|---|
| base bicubic | 28.407 | 无增强 |
| e1（2-block 出口） | 28.546 | 保命档 |
| e2（4-block 出口） | 28.664 | 中档 |
| **e3（6-block 出口）** | **29.630** | 最深档 |
| plain2（独立 2-block 固定模型） | 29.311 | 与 e1 同算力对照 |
| imdn_div（通用 6-block） | 28.365 | DIV2K 零样本 |
| imdn_lo（对症 6-block 单模型） | 29.917 | 与 e3 同算力上界 |

判决（`eval_pac16_verdict.json`）：

- **H1b 最深档不掉队 PASS**：e3 仅比对症单模型 imdn_lo 低 0.287dB，反超通用 imdn_div 1.265dB——一套权重兼顾三档，深档几乎不付代价。
- **H1c 越深越单调 PASS**：逐帧 e3≥e2≥e1 链比例 98.4%，e3≥e1 为 100%。
- **H2 困难段值得深算 PASS**：困难段 e3−e1 均值 +2.506dB（seg140 +3.17、seg150 +3.47、seg130 +0.88），简单段仅 +0.756。算力花在困难段回报高、简单段浅算即可，anytime 按内容分配算力的核心动机成立。
- **H4 时延单调**：240p 整帧中位 e1=14/e2=26/e3=39ms；e1/e2 在 30fps 的 33ms 帧预算内，e3=39ms 与标准 6-block IMDN 相同（非多出口引入的开销）。
- **H1a 浅出口弹性税（观察项，不否决方向）**：e1 比同算力独立 plain2 低 0.765dB。改等权损失（1/1/1）后 e1 仅回升 0.05、e3 反降 0.12，证明弹性税来自共享骨干要同时服务深出口的结构本质、非损失权重可解；e1 仍优于 bicubic 与通用 6-block。阶段1用自蒸馏/深度课程收敛。

### 阶段0判决：GO

核心承重墙（越深越单调、困难段深算有效、深档基本不掉队）全部成立。**遗留与边界**：浅出口 0.7dB 弹性税待自蒸馏；仅 240p、BBB 单视频（跨视频零样本需第二素材）；CondEncoder 尚未验证跨 QP/跨分辨率；e3 距对症单模型仍差 0.29dB。

## 后续路线

1. 结构定型：自蒸馏/深度课程补浅出口弹性税，评估残差精修级联（深出口在浅出口上学增量）。
2. 退化条件化：启用 CondEncoder，验证一套权重跨 QP/码率/分辨率不掉点（H3）。
3. halt 控制器 + 接播放器：随机时间预算下在线决定停在哪个出口，是与纯 CV 超分拉开差距的关键（anytime 调度）。
4. 时域一致性（插帧/抗闪烁，第二曲线）。
5. 系统对照与多轨迹、跨视频验证。
