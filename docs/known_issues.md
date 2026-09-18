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

### PACNet 帧级 VMAF 埋点（阶段二修复，易回退）
- analyzer 不能用 BONES 下发动作 `segment.enhance_action`（PACNet 模式恒为 3）查表打分，否则 halt 降 e1/e2 的帧、fast_complete 未增强帧全按 e3 计分，VMAF 虚高约 4 分/段。
- 正确做法：PACNet stats 回传 `exit_frame_counts=[e1,e2,e3]` 与 `fast_complete_triggered`，analyzer 按帧占比加权 real_vmaf，未增强帧按 base 计。改动见 `pacnet_enhancer.py`、`analyzer.py` 与补丁 `patch_vmaf_accounting.py`。

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
