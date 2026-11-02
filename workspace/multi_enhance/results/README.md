# results

| 子目录 | 用途 |
|---|---|
| **`canonical/`** | **权威汇总**——引用数字只认这里 |
| `archive/` | partial、作废跑、带日志的原始 summary、测量 log |
| `runs/` | `run_d3.py` / `run_d4.py` 新跑默认输出 |

原始播放器 JSON 仍在 `player/output/data/d{1,3,4}/`（体积大，不迁入本仓工作区）。

### canonical 文件

| 文件 | 内容 |
|---|---|
| `d3_bbb_0913.json` | BBB×0913 三臂 |
| `d3_bbb_0921.json` | BBB×0921 三臂 |
| `d4_tos_0913.json` | TOS×0913 对齐 Q 表后三臂 |
| `d5_bbb_0913.json` | D5 六臂消融（BBB×0913） |
| `d5_bbb_0921.json` | D5 六臂消融（BBB×0921） |
| `d5_slack_diag_0921.json` | 0921 ours vs no_slack 段级诊断 |
| `d5_kappa_0921.json` | tool_slack_kappa 扫描（0921） |
| `pref_joint_matrix_0921.json` | pref_joint 胆量/口味矩阵（0921） |


`archive/d4/*INVALID*` / 未对齐旧表跑次 **不要引用**。
