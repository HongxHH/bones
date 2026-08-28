# 快速开始

## 环境

- Conda 环境：`sr`
- 激活：`conda activate sr`
- 工作目录：`player/`

## 运行实验

```bash
cd player/
conda run -n SR python play.py \
  --mod_nes ca_bones \
  --mod_enhancer imdn_always \
  --mod_renderer headless \
  --mod_analyzer data_collector \
  --input data/video_mp4box/bbb.mpd \
  --trace data/3Glogs/report.2010-09-13_1003CEST.json \
  --config ca_cares_config.json
```

### 关键参数

| 参数 | 可选值 | 说明 |
|---|---|---|
| `--mod_nes` | `bones`, `ca_bones` | 网络增强调度控制器 |
| `--mod_enhancer` | `imdn_always` | 神经增强器 |
| `--mod_renderer` | `headless`, `opengl` | 渲染模式（headless 不渲染帧，用于实验） |
| `--mod_analyzer` | `data_collector` | 结果分析器 |
| `--input` | `data/video_mp4box/bbb.mpd` | DASH 视频清单 |
| `--trace` | `data/3Glogs/*.json` | 网络带宽轨迹 |
| `--config` | `*.json` | 自定义配置（CARES 参数、ca_mode 等） |

### 配置文件示例

`player/ca_cares_config.json`：
```json
{
  "ca_mode": "cares",
  "cares_lambda_min": 0.0,
  "cares_lambda_max": 0.003,
  "ca_feature_path": "data/content_features/bbb_features.json",
  "use_content_latency": false
}
```

`ca_mode` 可选值：`base_known`, `predicted`, `cares`

## 结果文件

- 位置：`player/output/data/`
- 命名：`{nes}_{enhancer}_fast-{use_fast_complete}_video-{video}_trace-{trace}_{timestamp}.json`
- 关键字段：`avg_vmaf`, `QOE`, `num_stall`, `dur_stall`, `oscillation_vmaf`, `enhance_level_count`, `gpu_metrics`

## QoE 公式

```
QOE = avg_vmaf - oscillation_vmaf - avg_freeze / 10.0
```

- `oscillation_vmaf`：相邻段 VMAF 差值的平均（权重 1:1）
- `avg_freeze`：平均每段卡顿时长（权重 1/10）

## 单次实验耗时

约 12 分钟（159 段，headless 模式，RTX 3060Ti）
