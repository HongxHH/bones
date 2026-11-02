# player 根目录历史实验散件（2026-08）

从 `player/` 根迁出，避免与运行时入口混放。脚本假定 cwd=`player/`，且常硬编码已删除的 `output/data/...` 路径，**默认不可复现**，仅溯源。

| 类别 | 文件 |
|---|---|
| 对比/分析脚本 | `compare_*.py`、`analyze_*.py` |
| CA / CARES 调参 | `ca_*_config.json`、`tune_a*.json` |
| 当时输出摘录 | `*_comparison.txt`、`workspace_comparison.json` |
| 旧 latency 备份 | `imdn_latency_origin.json`（无代码引用） |

仍留在 `player/` 根、供播放器加载的：`play.py`、`imdn_*.json`、`pacnet_*.json`、`latency_table.json`、`tos.yaml`、`config_bones_no_fast.json`、`config_mt_bones.json`。
