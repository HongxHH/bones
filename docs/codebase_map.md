# 代码地图

## 目录结构

```
bones/
├── player/                    # 真实播放器（iStreamPlayer）
│   ├── play.py                # 实验入口
│   ├── istream_player/
│   │   ├── config/config.py   # 配置（use_fast_complete 默认 True）
│   │   └── modules/
│   │       ├── nes/           # 网络增强调度控制器
│   │       │   ├── bones.py           # 原始 BONES
│   │       │   ├── ca_bones.py        # ★ CA-BONES（base_known/predicted/cares）
│   │       │   ├── content_aware_bones.py
│   │       │   ├── bola.py, mpc.py    # 其他基线
│   │       │   └── drl_nes.py, nas.py # DRL/NAS 控制器
│   │       ├── enhancer/      # 神经增强器
│   │       │   ├── enhancer_imdn_always.py  # ★ IMDN 增强器（EWMA延迟/per-seg预测）
│   │       │   ├── fast_complete_policy.py   # ★ fast_complete 机制
│   │       │   ├── imdn_model.py, imdn_block.py
│   │       │   └── imdn_weight/     # 模型权重
│   │       └── analyzer/analyzer.py # 结果分析（QoE 公式在此）
│   ├── data/
│   │   ├── video_mp4box/       # DASH 视频分片
│   │   ├── 3Glogs/             # 网络带宽轨迹
│   │   └── content_features/   # ★ 预计算内容特征（bbb_features.json）
│   └── output/data/            # 实验结果 JSON
│
├── control/                   # 仿真器（快速迭代）
│   ├── bones.py               # 原始 BONES 仿真
│   ├── bones_ca.py            # ★ CA-BONES 仿真（5种模式）
│   ├── bola.py, mpc.py, pensieve.py
│   └── autotune_*.npz        # 自动调参结果
│
├── workspace/                 # 工作区（脚本/模型/结果）
│   ├── vmaf_gain_predictor.pkl   # ★ 训练好的 VMAF 预测模型（RandomForest, R²=0.909）
│   ├── train_vmaf_predictor.py    # 模型训练脚本
│   ├── run_stage_a_sim.py         # 仿真器对比
│   ├── compare_cares.py           # CARES vs BONES 对比
│   ├── analyze_resource_efficiency.py
│   └── notes/, results/, theory/, experiments/  # 子目录
│
├── honghong/                  # ★ 红红目录（个人ID，CARN模型/FLOPs/延迟测量）
│   ├── archs/                 # CARN 模型架构
│   ├── checkpoints/           # 模型权重
│   ├── FLOPs/                 # FLOPs 计算
│   ├── calc_FLOPs.py
│   └── measure_latency.py
│
├── data/bbb/imdn_bbb_3060ti.json  # ★ 段级 VMAF 数据（159段×5码率×5级别）
├── enhance/                   # 增强模型代码（IMDN/NAS）
├── experiment/                # 实验脚本
└── docs/                      # 本文档目录
```

## 关键模块说明

### CA-BONES 控制器（`player/istream_player/modules/nes/ca_bones.py`）
- 三种模式：`base_known`（段级base+内容因子缩放）、`predicted`（RandomForest预测VMAF）、`cares`（内容感知资源惩罚）
- 延迟表：调用 `enhancer.get_latency_table(segment_index=idx)`，支持 per-segment 预测
- 决策核心：`_bones_control()`，Lyapunov 优化 `min (drift - V*reward) / size`

### IMDN 增强器（`player/istream_player/modules/enhancer/enhancer_imdn_always.py`）
- EWMA 实测延迟反馈：abort/fast_complete/正常完成都更新延迟表
- per-segment KNN 延迟预测：基于内容特征预测延迟缩放因子
- 内容特征加载：`ContentFeatureStore`，从 `bbb_features.json` 读取

### 结果分析器（`player/istream_player/modules/analyzer/analyzer.py`）
- QoE 公式：`qoe = avg_vmaf - oscillation_vmaf - avg_freeze / 10.0`（第783行）
- fast_complete VMAF：加权平均 `base*(1-ratio) + enhance*ratio`（第636行）
