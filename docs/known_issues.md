# 已知问题

## 环境与配置

### Conda 环境
- 环境名：`sr`
- 激活：`conda activate sr`
- 必须在 `player/` 目录下运行实验
- scikit-learn 需手动安装：`conda run -n SR pip install scikit-learn`

### 渲染模式
- **headless 模式不加载增强缓冲区**，会影响实验结果（延迟表不准确）
- 推荐用 **opengl 模式 + 注释掉帧渲染输出**，既无头跑实验又保证缓冲区加载
- 注释位置：`player/istream_player/modules/renderer/` 下的 OpenGL 渲染代码

## 仿真器与真实播放器差异

### 仿真器（`control/`）
- 使用固定延迟表，**无 per-segment 延迟变化**
- 内容感知质量预测会导致决策方差增大→振荡/stall
- **不能用于验证内容感知自适应**，只能用于快速算法原型验证
- 仿真器中 oracle（完美预测）QoE 也低于 BONES，说明仿真器环境不适合内容感知

### 真实播放器（`player/`）
- 有真实 GPU 延迟波动、内容相关延迟、fast_complete 机制
- 单次实验 ~12 分钟（159段，RTX 3060Ti）
- 是验证内容感知方法的正确场所

### halt 离线仿真库存判据不能照搬到播放器（阶段二确认）
- 离线 `workspace/pacnet/sim_tos_halt.py` 只有“增强 vs 播放”一个库存递推 `B+=帧间隔−延迟`，没有 fast_complete，库存跌破单帧门槛才降档，贪心库存策略成立。
- 真实播放器另有独立的 fast_complete 保护层，其动态阈值（约 0.8s 余量）远高于 halt 单帧库存门槛（约 0.06s）。halt 若用纯库存水位判据，lead 下降时 fast_complete 先触发、整段截断，e2/e1 档来不及执行（实测死磕 e3 到只做 60/96 帧）。
- 播放器里 halt 应改用可调度性前瞻：逐级判断 `lead >= l_exit×剩余帧×安全系数`，选最高可行档，把 fast_complete 留作“连 e1 都不可行”的最终兜底。
- 离线仿真还省略解码、resize、H2D/D2H、asyncio 调度与下载争抢等每帧固定开销，临界算力下实测帧延迟高于静态延迟表，sim 的 e3 占比偏乐观，只能做趋势/阈值初筛，最终以播放器为准。

### 制造算力紧张必须用乘法负载，不能用固定加法（阶段二确认）
- `_gpuburn.py` 的 `PACNET_BURN_K` 按每帧实测前向 `fwd_ms×(K−1)` 烧 matmul，端到端约 ×K，保持三出口延迟比，halt 阈值不倒挂，PAC/IMDN 同口径公平。
- 旧 `PACNET_EXTRA_MS` 对每帧固定加毫秒，会压缩出口延迟比（如 1.397→1.177），使 e3 阈值反低于 e2、e2 档永远选不到（e2 死档）。仅保留作兼容。

### 纯增强库存仿真不含下载，网络结构类结论须含下载时间线复核（阶段二 e0 证伪）
- `sim_tos_halt.py`、`sim_e0_falsify.py` 只建模增强供给 vs 播放消耗，无网络下载。k=1.5 现状 halt 在其中已 0 掉帧，加虚拟更快出口 e0（加压 5.2~17.5ms 扫遍）逐字节无收益、e0 从不被选中；但真实 0921 仍卡 84.5s，纯增强仿真复现不了。
- 真实 0921 三策略逐段 slack（增强完成相对播放点余量）无一为负，184 段下载墙钟 802s > 视频 736s；卡顿主体是下载供不上 + 增强/gpuburn 对下载解码的资源竞争，不是最浅增强档吞吐（e1=17.5ms 对 24fps DT=41.67ms 本有 2.4 倍余量）。
- 教训：评估更快出口/更轻网络能否减卡顿，必须在含真实段到达时间的时间线或真实播放器上做，纯增强库存仿真只适合趋势初筛，不能据此决定改网络/重训。
- gpuburn 效度：burn=fwd×(K−1) 使选深档额外多烧 GPU、可能拖慢同机下载/解码；真实弱设备是全档等比变慢、无选深档额外烧卡，K=1.5 可能高估深档流畅度惩罚。

### 增强在 asyncio 单线程同步阻塞，但与下载基本并行、卡顿杠杆在码率（阶段二反事实）
- 全部 enhancer 的 torch forward + cuda.synchronize + gpuburn 在 asyncio run 内同步执行，全工程无 run_in_executor/线程池，每帧仅 sleep(0) 让出；trace 下载在同一事件循环按 20000 字节分块 sleep 限速。
- 0921 配对到相同码率的段（halt vs IMDN 155 段）下载墙钟仅差 +3.1s（中位 −0.02s），增强多花 368s 却几乎没拉长同码率下载——增强在下载 chunk sleep 间隙“免费搭车”。halt 多卡 15.5s 主要来自差网选了更多 repr1（下载字节多）。
- 复刻 trace.py 限速的下载反事实（sim_download_counterfactual.py）：固定 repr0(400k) 卡顿 0（下载 615s < 视频 736s），repr1 卡 512s、repr2 卡 1096s。卡顿唯一硬杠杆是码率下载量；仿真系统性低估真实卡顿 15-20s（未建模增强关键路径/解码/起播），但相对顺序正确。


### 共享骨干浅出口被专用轻模型帕累托压制（阶段二全 K 检验，决定 anytime 适用设定）
- 固定 repr0 消除下载瓶颈后扫 K=1..6（sim_pareto_k.py）：halt 仅 K≤1.2（全程 e3、anytime 不换档）赢 IMDN 0.58；K≥1.5 紧张区 halt 掉帧崩到 82.7，IMDN l2（5.1ms、8 倍余量）稳在 88.66，halt 反输 0.8~6.4；时变算力（sim_variable_compute.py）同样输 1~2.7。
- 根因：PAC 共享 nf=32 重骨干的 e1(11.64ms,86.18)/e2(24.19,87.12) 是“深网络半成品”，被专用轻模型 IMDN_RTC l2(5.1ms,88.70) 严格 dominate；e0 更浅但质量更低仍被压制。轻量 CNN 单帧、小倍率超分的低算力甜区已被轻模型占满，anytime 需换到轻模型不够用的设定（高倍率/强混合退化/视频 VSR），见 current_state 下一步。

### 单帧压缩/倍率难度扫描：轻-重质量阶梯 ≤2.5、极端退化归零（阶段二难任务测量）
- `workspace/pacnet/hard_survey/measure_hard_survey.py` 在 7 档退化（426/320 宽即 4.5x/6x × CRF26/34/42，加 426/400k 锚点，每档 32 帧）上测 l1/l2/l3 与 PAC e1/e2/e3：l2(5.1ms)→e3(33.8ms) 的 VMAF 差全程 0~2.52，从不达 3~5 的价值阈值；差距呈倒 U（中等退化 crf26 最大 2.52，极端压缩 crf42 信息被抹除后所有模型一起崩、差距归零 0~0.5）；6x 不大于 4.5x；e3 比 l3(9.8ms) 仅高 0.6~1.7 分却慢 3.4 倍。
- 含义：换更重压缩或更高倍率救不出 anytime 浅档价值区间，单帧 IMDN 系、压缩+下采样任务族的共享骨干 anytime 不成立。视频时序超分（VSR）这一“换信息源”维度尚未排除，但本地缺匹配退化的 VSR 代码/权重（anovsr、CARN-video 仅权重无推理代码，STDF 仅模型定义无权重且不放大），公平对比需训练，见 current_state 下一步。

### 感知指标复验：VMAF 的小幅优势在 LPIPS 下反转为劣势（2026-09-19，H3 评测安全网）
- 起因：MSU 约 12 万人主观标注显示 VMAF 在“压缩再超分”画面上与主观分相关性仅约 0.41（纯压缩约 0.9），需排除“轻≈重”是 VMAF 失灵的假象。
- 脚本 `workspace/pacnet/hard_survey/measure_perceptual.py`（全参考 LPIPS-squeeze / CW-SSIM / VIF，7 档退化×7 方法×32 帧、统一 1920×800 对源）与 `measure_perceptual_pair.py`（l2/e3/l3 逐帧配对 + 双侧符号检验）；结果 `hard_survey_perceptual.json`、`perceptual_pair.json`。pyiqa 0.1.16（无 ERQA，DISTS 需 VGG16 528MB 默认关，`ENABLE_DISTS=1` 开启）。
- LPIPS（越低越好）配对差 d=lpips(l2)−lpips(e3)：7 档中 6 档为负且显著（anchor、426crf26 p=0.0021，426crf34 p=0.0005，426crf42 与 320crf42 p<0.0001，320crf34 p=0.007），e3 更差帧占 75%~91%；唯一均值为正的 320crf26 不显著（p=0.050）且 68.8% 的帧 e3 仍更差（均值被少数帧拉动）。独立 l3 的 LPIPS 始终最低。CW-SSIM 差 ±0.002 属噪声级；VIF 上 e3 仅 +0.001~0.01（锐化带来的高频能量）。
- 判决：不是 VMAF 掩盖了重模型收益，而是 VMAF/VIF 把过锐化的高频计为质量，LPIPS 识别出共享骨干重出口产出不自然纹理、感知上反而偏离 GT。单帧“压缩+下采样”任务的 anytime 深度伸缩彻底盖棺，且不是尺子问题；瓶颈在输入信息而非模型容量（独立重网 l3 相对 l2 的 LPIPS 优势也很小）。CW-SSIM 对 bicubic 平坦帧有除零爆炸（3e4 量级），该列只在模型间对比。



### PACNet 帧级 VMAF 埋点（阶段二修复，易回退）
- analyzer 不能用 BONES 下发动作 `segment.enhance_action`（PACNet 模式恒为 3）查表打分，否则 halt 降 e1/e2 的帧、fast_complete 未增强帧全按 e3 计分，VMAF 虚高约 4 分/段。
- 正确做法：PACNet stats 回传 `exit_frame_counts=[e1,e2,e3]` 与 `fast_complete_triggered`，analyzer 按帧占比加权 real_vmaf，未增强帧按 base 计。改动见 `pacnet_enhancer.py`、`analyzer.py` 与补丁 `patch_vmaf_accounting.py`。

### 运行时 VMAF/QoE 是档位级查表常数，不是逐帧/逐段实测（2026-09-19 确认）
- 播放器运行时不跑 VMAF：每段 VMAF 直接查 5×5 质量表 `quality_table[q_idx,e_idx]`（加载 `player/imdn_div2k_quality.json`，`enhancer_imdn.py:489-516`；取值 `analyzer.py:625/655/670/672`）。每格是离线把该档全部段（BBB 159 段）跑完 libvmaf 再跨段平均的**单个全片标量**，同档位每段同值、与该段实际内容无关。
- 表的来源 `data/bbb/measure.py:55-103`：libvmaf 对每段跑一次，ffmpeg `VMAF score`（`:31`）是段内逐帧均值；逐段数组存在 `imdn_div_quality.json`，播放器加载的 `imdn_div2k_quality.json` 又塌缩成每格单值。
- PAC halt 段内按 e1/e2/e3 真实帧数加权（`analyzer.py:636-643`）、fast_complete 按增强比例插值（`:657`），权重是真实帧数，但被加权的 VMAF 仍是档位常数。
- 会话 `avg_vmaf`=跨段简单平均（`:788-789`），`oscillation`=相邻段 VMAF 绝对差均值（`:793-797`），`QoE=avg_vmaf−oscillation−avg_freeze/10`（`:806`）。后果：oscillation 只反映档位切换、不含内容波动；卡顿惩罚过弱（84.5s 仅扣约 0.046）。
- **结论口径**：端到端多轨迹结果里，卡顿/延迟/出口分布/fast_complete 段数是播放时真实测的、有效；其中 VMAF/QoE 只是档位级估计，不能用于支撑任何内容感知/帧级方法的优劣结论。hard_survey、H3（LPIPS）、H1、H2 等离线脚本走 libvmaf 逐帧/clip 实测 + 符号检验，结论不受此限。
- 真实客户端没有 1080p 参考，全参考 VMAF 运行时本就不可算，只能事后离线评。方向定型后的播放器改造预案：把 5×5 标量表换成 `[段][q][e]` 逐段表（BBB/IMDN 逐段数组已在 `imdn_div_quality.json`，TOS/PAC 用离线脚本补测），analyzer 按段号索引；必要时 headless 落盘实际渲染帧事后逐帧算 VMAF/LPIPS 抽检；QoE 卡顿权重届时按标准线性 QoE（每秒罚 λ）重定并预注册。勿在播放路径内联 libvmaf。

## 算法问题

### CA-BONES oscillation 暴增
- **现象**：所有在质量奖励上做内容感知的模式（base_known/predicted），oscillation 都比 BONES 高 1.8-4.5
- **根因**：内容感知增加决策方差→相邻段质量差异大→oscillation 增大
- **QoE 公式**：`qoe = avg_vmaf - oscillation_vmaf - avg_freeze/10`，oscillation 权重 1:1，影响极大
- **解决方向**：在资源惩罚上做内容感知（CARES），或加入质量切换惩罚

### predicted 模式过度自信
- **现象**：level3 选择从 BONES 的 8 段暴增到 30 段
- **根因**：RandomForest 预测有误差，但决策系统不考虑不确定性→过度自信→选高级别→stall
- **教训**：预测模型 R² 高不代表决策效果好，需要考虑预测误差对决策的影响

### per-seg KNN 延迟预测冷启动
- **现象**：前 10 段 KNN 历史记录不足，预测不准
- **当前处理**：冷启动阈值 10 段，期间用全局 EWMA 延迟
- **待优化**：可以用内容特征直接初始化延迟预测，减少冷启动期

## 实验问题

### 实验耗时
- 单次真实播放器实验 ~12 分钟
- 调参需要多次实验，耗时较长
- 建议：先用仿真器快速验证方向，再用真实播放器确认

### 结果文件命名
- 位置：`player/output/data/`
- 命名：`{nes}_{enhancer}_fast-{use_fast_complete}_video-{video}_trace-{trace}_{timestamp}.json`
- 按时间戳区分，注意不要混淆不同配置的实验

### GPU 温度影响
- 连续多次实验会导致 GPU 温度升高→增强延迟增加→结果波动
- 建议：实验间隔让 GPU 冷却，或在结果中记录 GPU 温度

## 代码坑

### dl_time 单位 bug（已修复）

- **位置**：`control/bones_ca.py:_lookahead_control()`
- **问题**：`dl_time = seg_size / bandwidth * 1000` 把 bytes 当成 Kb
- **正确**：`dl_time = seg_size * 8 / bandwidth`（seg_size 单位 bytes，bandwidth 单位 kbps）
- **影响**：下载时间高估 125 倍，前瞻预测 massive 卡顿，高带宽轨迹 QoE 从 ~96 降到 ~80
- **修复时间**：2026-08
- **教训**：仿真器中手动计算物理量时严格核对单位，特别是 bytes vs bits

### 前瞻规划剩余问题

- **4G_car、hd_fixed 轨迹略输 BONES**：可能原因包括恒定带宽预测不准确、卡顿惩罚偏保守、缓冲区模型简化
- **带宽预测**：当前用 EWMA 单点估计作为未来 N 段恒定预测，未考虑带宽波动
- **N>2 复杂度**：N=3 时 8000 种组合，N=4 时 160000 种，需剪枝或近似
- **仅仿真器验证**：尚未在真实播放器 iStreamPlayer 上验证

### 配置文件路径
- `ca_feature_path` 相对于 `player/` 目录：`data/content_features/bbb_features.json`
- `ca_predictor_path` 相对于 `player/` 目录：`../workspace/vmaf_gain_predictor.pkl`
- `ca_enhance_file` 相对于 `player/` 目录：`../data/bbb/imdn_bbb_3060ti.json`

### 段索引从 1 开始
- 实验结果中 `segment.index` 从 1 开始
- 内容特征 `bbb_features.json` 中 `segment_index` 从 0 开始
- 转换：`feature_index = segment.index - 1`

### enhance JSON 数据结构
- `base_quality_vmaf`: (num_bitrate-1, num_seg) = (4, 159)
- `enhanced_quality_vmaf`: (num_bitrate_enh, num_method, num_seg) = (3, 4, 159)
- 最高码率（第5个）不增强，base VMAF 设为 100
- 无效值用 -inf 表示
