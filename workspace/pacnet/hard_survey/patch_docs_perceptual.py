# -*- coding: utf-8 -*-
"""把 2026-09-19 感知指标复验（H3）结论与方向转向 H1 写入三个 docs 文件。
自动保留各文件原有换行符（CRLF/LF）。每个替换断言唯一命中。"""
import io, os

DOCS = r"E:\dev\Code\Python\bones\docs"


def patch(path, repls):
    p = os.path.join(DOCS, path)
    raw = io.open(p, "r", newline="", encoding="utf-8").read()
    nl = "\r\n" if "\r\n" in raw else "\n"
    for old, new in repls:
        o, n = old.replace("\n", nl), new.replace("\n", nl)
        c = raw.count(o)
        assert c == 1, f"{path}: anchor count={c} for: {old[:50]!r}"
        raw = raw.replace(o, n)
    io.open(p, "w", newline="", encoding="utf-8").write(raw)
    print("patched", path)


# ---------- known_issues.md：难度扫描节后加感知复验节 ----------
ki_anchor = "- 含义：换更重压缩或更高倍率救不出 anytime 浅档价值区间，单帧 IMDN 系、压缩+下采样任务族的共享骨干 anytime 不成立。视频时序超分（VSR）这一“换信息源”维度尚未排除，但本地缺匹配退化的 VSR 代码/权重（anovsr、CARN-video 仅权重无推理代码，STDF 仅模型定义无权重且不放大），公平对比需训练，见 current_state 下一步。"
ki_new = ki_anchor + """

### 感知指标复验：VMAF 的小幅优势在 LPIPS 下反转为劣势（2026-09-19，H3 评测安全网）
- 起因：MSU 约 12 万人主观标注显示 VMAF 在“压缩再超分”画面上与主观分相关性仅约 0.41（纯压缩约 0.9），需排除“轻≈重”是 VMAF 失灵的假象。
- 脚本 `workspace/pacnet/hard_survey/measure_perceptual.py`（全参考 LPIPS-squeeze / CW-SSIM / VIF，7 档退化×7 方法×32 帧、统一 1920×800 对源）与 `measure_perceptual_pair.py`（l2/e3/l3 逐帧配对 + 双侧符号检验）；结果 `hard_survey_perceptual.json`、`perceptual_pair.json`。pyiqa 0.1.16（无 ERQA，DISTS 需 VGG16 528MB 默认关，`ENABLE_DISTS=1` 开启）。
- LPIPS（越低越好）配对差 d=lpips(l2)−lpips(e3)：7 档中 6 档为负且显著（anchor、426crf26 p=0.0021，426crf34 p=0.0005，426crf42 与 320crf42 p<0.0001，320crf34 p=0.007），e3 更差帧占 75%~91%；唯一均值为正的 320crf26 不显著（p=0.050）且 68.8% 的帧 e3 仍更差（均值被少数帧拉动）。独立 l3 的 LPIPS 始终最低。CW-SSIM 差 ±0.002 属噪声级；VIF 上 e3 仅 +0.001~0.01（锐化带来的高频能量）。
- 判决：不是 VMAF 掩盖了重模型收益，而是 VMAF/VIF 把过锐化的高频计为质量，LPIPS 识别出共享骨干重出口产出不自然纹理、感知上反而偏离 GT。单帧“压缩+下采样”任务的 anytime 深度伸缩彻底盖棺，且不是尺子问题；瓶颈在输入信息而非模型容量（独立重网 l3 相对 l2 的 LPIPS 优势也很小）。CW-SSIM 对 bicubic 平坦帧有除零爆炸（3e4 量级），该列只在模型间对比。"""
patch("known_issues.md", [(ki_anchor, ki_new)])


# ---------- current_state.md：新最新阶段 + 当前在做 + 下一步 + 已放弃 ----------
cs_old_head = "# 当前状态\n\n## 最新阶段：PAC-Net 阶段二 算力紧张下 halt 验证（2026-09-18，TOS + gpuburn K=1.5，修复两个埋点/口径 bug，halt 改可调度性前瞻）"
cs_new_head = """# 当前状态

## 最新阶段：单帧 anytime 盖棺，方向转向 codec 元数据条件化（2026-09-19）

两条独立证据在本日收敛，决定放弃“同信息加深/多出口”，转向“给增强网络喂解码器免费产出的信息”。

1. 文献调研（2025-2026，26 篇核验，飞书文档 MCN5dOGzco7r5axtI8oc67UnnzJ）：低码率增强的胜负手在输入端保信息而非增强网络容量——Orbis（SenSys'25）消融去掉保像素模块 QoE 跌 168%、去掉增强模型仅跌 34%；单帧深度伸缩/多出口在低码率被多家判死；codec 元数据（QP/运动矢量/残差，解码器免费产出、零带宽）条件化轻量网（PnP-VCVE, CVPR'25 等）列为首选 H1；anchor 稀疏帧 + 残差回加（NEMO，计算降近百倍、质量差约 0.41dB、能耗降约九成）列为 H2 储备；VMAF 在压缩再超分上主观 PLCC 仅约 0.41（MSU 约 12 万人）列为 H3 评测安全网。学习式双向光流在 270p 主观输给轻量 CNN（AIM2025），diffusion 主干端侧不可行。
2. H3 感知指标复验（详见 known_issues“感知指标复验”条）：hard_survey 同口径 7 档退化补 LPIPS/CW-SSIM/VIF 后，重出口 e3 相对轻量 l2 在 VMAF 的 +0~2.5 微弱优势，于 LPIPS 反转为显著劣势（6/7 档符号检验 p<0.01、e3 更差帧 75%~91%，唯一正档不显著），排除“VMAF 失灵掩盖重模型收益”。单帧 anytime 加深/早退就此盖棺。

下一步主攻 H1：把解码器侧 QP/运动矢量/残差统计作为条件输入单个轻量网（不再做多出口），零额外带宽，正对“输入端保信息”；H2 留作系统组合。

## 上一阶段：PAC-Net 阶段二 算力紧张下 halt 验证（2026-09-18，TOS + gpuburn K=1.5，修复两个埋点/口径 bug，halt 改可调度性前瞻）"""
patch("current_state.md", [(cs_old_head, cs_new_head)])

cs_doing_anchor = "## 当前在做什么\n"
cs_doing_new = "## 当前在做什么\n\n- **H3 感知指标复验完成、单帧 anytime 盖棺（2026-09-19）**：LPIPS 复验见上，下一步转 H1 codec 元数据条件化\n"
patch("current_state.md", [(cs_doing_anchor, cs_doing_new)])

cs_short_old = """### 短期（优先）
1. **PAC-Net 三出口VMAF精确测量（进行中）**：为e1/e2/e3在240/360/480p分别测VMAF，生成pacnet_quality.json替换IMDN近似表
2. **PAC-Net 多轨迹验证**：3G/4G各3-5条轨迹，验证QoE增益稳定性，看胜率和中位
3. **PAC-Net halt控制器调优**：safe_factor、gain_thresh阈值网格搜索
4. **第二曲线储备：时域增强**：插帧/抗闪烁等非超分增强维度，仅做小样本可行性摸底"""
cs_short_new = """### 短期（优先）
1. **H1 codec 元数据条件化增强（主攻）**：从 TOS 码流解析每帧/宏块 QP、运动矢量、残差统计，作为条件输入单个轻量网（不做多出口），先离线小样本验证相对无条件轻网的 VMAF/LPIPS 增益
2. **H1 条件构造与对齐**：确认 ffmpeg/解码器导出 QP/MV/残差的方式，与超分输入帧严格对齐，保证零额外带宽、仅用解码侧已有信息
3. **评测口径固定**：VMAF 与 LPIPS 并列（VMAF 在压缩再超分上高估锐化），卡顿单列；QoE 卡顿权重过弱（84s 卡顿仅扣约 0.046 分，analyzer.py:806）待补主流 λ 口径对照，原公式保留以维持 BONES 可比性
4. 已完成历史项：PAC-Net 三出口 VMAF 精确测量、多轨迹验证、halt 安全系数扫描、hard_survey 难度扫描、LPIPS 复验（均见上文与 known_issues）"""
patch("current_state.md", [(cs_short_old, cs_short_new)])

cs_drop_anchor = "- ❌ PAC-Net 梯度隔离级联（阶段1证伪，切断端到端协同）"
cs_drop_new = cs_drop_anchor + "\n- ❌ 单帧共享骨干 anytime 深度伸缩/多出口：全 K 帕累托 + hard_survey + LPIPS 复验三重证伪（2026-09-19 盖棺），瓶颈在输入信息不在容量"
patch("current_state.md", [(cs_drop_anchor, cs_drop_new)])


# ---------- pacnet_progressive_enhancer.md：3.4 判决追加感知复验 ----------
pn_anchor = "唯一未排除的是视频时序超分（多帧提供单帧拿不到的亚像素信息，属“换信息源”而非同信息加深）；但本地缺匹配退化的 VSR 代码/权重，公平对比需训练。下一步先用光流多帧融合 oracle（不训练、不 OOD）估时序信息上限，再决定是否训练轻/重 VSR 或转系统成本角度，详见 current_state。"
pn_new = pn_anchor + """ 2026-09-19 感知指标复验（`measure_perceptual.py`、`measure_perceptual_pair.py`）排除了“VMAF 失灵掩盖重模型收益”：同一 7 档退化补 LPIPS 后，l2→e3 在 VMAF 的 +0~2.5 于 LPIPS 反转为显著劣势（6/7 档符号检验 p<0.01、e3 更差帧 75%~91%，唯一正档 320crf26 不显著），CW-SSIM 无差异、VIF 仅记到锐化高频，独立 l3 LPIPS 始终最好但相对 l2 优势也很小。单帧 anytime 就此盖棺且非尺子问题；结合 2025-2026 文献（输入端保信息 Orbis 消融 QoE 跌 168% vs 去掉增强模型仅跌 34%），方向转 codec 元数据（QP/运动矢量/残差）条件化轻量网（H1，零带宽），anchor 稀疏帧 + 残差回加储备为 H2；光流多帧 oracle 降级（AIM2025 学习式双向光流在 270p 主观输给轻量 CNN）。"""
patch("pacnet_progressive_enhancer.md", [(pn_anchor, pn_new)])

print("ALL DOCS PATCHED")
