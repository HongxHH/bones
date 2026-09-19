# H1：codec 元数据条件化增强——数据通路

## 定位

单帧 anytime（PAC-Net 多出口）经全 K 帕累托、hard_survey、LPIPS 复验三重证伪后，瓶颈判为**输入信息不足而非模型容量**（见 `current_state.md` 2026-09-19、`known_issues.md` 感知指标复验条）。H1 把 H.264 解码器免费产出、零额外带宽的码流侧信息（QP、运动矢量，残差见下）作为条件输入喂给单个轻量网，让网络知道"每个块被压得多狠、往哪运动"。

本文件只记录**数据通路**：这些信号在解码侧能否拿到、如何与帧严格对齐。条件网络结构与离线证伪见 `h1_condnet_falsification.md`（结论：端侧小样本下条件无增益、便宜版证伪）。本文件脚本与样例在 `workspace/pacnet/h1_codecmeta/`。

## 结论（2026-09-19，lr_426_crf34.mp4，426×178@24，H.264，MB 栅格 27×12=324）

| 信号 | 解码侧可得性 | 提取方式 | 对齐校验 |
|---|---|---|---|
| 运动矢量 MV | 可得，逐 16×16 MB、1/4 像素精度 | PyAV 打开后置 `flags2 \|= 1<<28`，读帧 `MOTION_VECTORS` side data | 箭头空间连贯、贴合镜头/物体运动；P 帧前向全覆盖 |
| 量化参数 QP | 可得，逐 MB（h264 不导出 QP side data，走 `-debug qp` 文本） | `ffmpeg -threads 1 -debug qp`，严格行解析 | 482/482 帧精确 324 个 QP、与 PyAV 帧类型 0 处不一致 |
| 预测残差 | **标准解码器不导出** | side data 仅 `MOTION_VECTORS`、`SEI_UNREGISTERED` | 需改解码器 / JM·VTM 参考软件，第一版不做 |

## 运动矢量提取

- PyAV（av 16.0.1，SR 环境已装）。`av.open(..., options={'flags2':'+export_mvs'})` **不生效**；必须打开后设 `st.codec_context.flags2 |= (1<<28)`（EXPORT_MVS 位是 `1<<28`，不是 `1<<23`），见 `extract_mv.py:19,26`。
- side data 是原始字节，每条记录为 ffmpeg `AVMotionVector`，native 对齐 **40 字节**，struct 格式 `@iBBhhhh2xqIIH6x`（`extract_mv.py:17`）。字段：`source(i32), w(u8), h(u8), src_x,src_y,dst_x,dst_y(i16), flags(u64), motion_x,motion_y(u32), motion_scale(u16)`。踩坑：w/h 后不能加填充，flags 前补 2x。
- 字段语义（实测）：
  - `source`：−1=past 前向（P 帧全部、B 帧前向分量），+1=future 后向（仅 B 帧），无 0；
  - `dst_x/dst_y`：当前帧 MB 中心像素坐标（网格 8,24,…，末块可越界）；
  - `motion_x/y`：完整运动矢量（整像素+1/4 亚像素），**位移 = motion / motion_scale，scale 恒为 4**；不要用 src−dst（漏掉亚像素且早期错位读出 106px 假值）；
  - I 帧无该 side data，MV 场为 0。
- 建场：以 dst 为中心填 w×h=16×16，取前向 source=−1，`|mv|>48px` 判为帧外/无纹理误配丢弃（`extract_mv.py:20,44-66`）。输出 mvx、mvy、cover 三场。
- 核对图 `mv_check_*_f{297,426,384,339}.png`：top 运动帧中位位移 9~17px、覆盖率 0.57~0.76，箭头大片同向、空间连贯。
- **B 帧注意**：只取前向时 B 帧 cover 仅约 0.20（大量块用后向预测或 skip）。条件网络阶段 B 帧应合并后向分量（建议 MV 给 4 通道：前/后向 mvx、mvy），P 帧后向通道为 0。

## QP 提取

- h264 解码器不导出 QP table side data；官方命令行出口是 `ffmpeg -debug qp`，但**默认多线程解码帧交错**（解析出 107 帧 vs 解码 90 帧、混入非 QP 数字、出现 QP=95 越界）。必须 `-threads 1`（`extract_qp.py:18`）。
- 单线程下每帧格式固定：一行 `New frame, type: X`，随后 nMBH 行纯数字，每行 nMBW 个两位 QP 紧密拼接（本例 12 行 × 54 字符 = 324）。解析只收"去 `[h264 @ addr]` 行首前缀后整行纯数字、偶数长度"的行，其余日志（nal_unit_type、graph、detected cores）自然排除；每帧必须凑满 324 且 max≤51（`extract_qp.py:17-32,69`）。
- reshape(nMBH,nMBW) 后 np.repeat 16 上采样到帧分辨率，最近邻保持块级边界。
- 核对：片头黑底标题卡 mean QP≈21.8/std1.5，火箭引擎复杂运动帧 f275 mean≈37.4/std6.0、QP 24~44（`qp_check_*_f{275,302,303,289}.png`），内容自适应方向正确（平坦区低 QP、复杂/运动区高 QP）。

## 残差与第一版特征集

- 残差=重建帧−运动补偿预测帧，是变换量化后的"新信息"，标准 ffmpeg/PyAV 不导出（`combine_features.py:19-23` 实测 side data 仅两类）。精确获取需 patch libavcodec 或 JM/VTM 参考软件，工程量大；且其信息与 QP 高度重叠（高 QP 即残差被粗量化）。
- 第一版条件特征（零带宽、解码器免费）：**QP map 1 通道 + 前/后向 MV 共 4 通道**，块级最近邻上采样到帧分辨率，与 LR 帧拼接喂单个轻量网（l2 规模，不再做多出口）。
- 残差的低成本替代留作消融候选：直接从重建帧算像素域高频能量 / 块边界不连续（DCT 高频、Sobel），近似"哪里被压坏"，无需解码器内部量。

## 复现入口

- `extract_mv.py`：MV 提取 + 箭头核对，产物 `mv_check_*.png`。
- `extract_qp.py`：QP 解析 + 热力核对（main 内 N 控制帧数），产物 `qp_check_*.png`。
- `combine_features.py`：对同一帧（TARGET）产出 LR+MV+QP 严格对齐的 `codec_features_f275.npz/.png`，并审计 side data 类型。
- 运行用 SR 环境解释器 `E:\dev\env\miniconda3\envs\SR\python.exe`，系统 ffmpeg 6.1.1 在 PATH。

## 后续：条件网证伪已完成

数据通路成立后做了第二步：l2 规模轻量条件网（add 特征相加 / cat 输入拼接两种注入），三臂（无条件 / 真 QP+MV / 乱序条件）对照，BBB 训练、TOS 留出，hard_survey 同口径 VMAF+LPIPS+符号检验。**端侧小样本下条件无任何可测量增益（ΔVMAF≤0.04、ΔLPIPS≈0.001、真条件≈乱序条件），H1 便宜版证伪**，详见 `h1_condnet_falsification.md`。MV 用于显式时序配准（warping，H2）而非当输入通道，是未被证伪的方向。
