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
