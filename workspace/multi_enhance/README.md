# multi_enhance

端侧多类型画质增强的自适应调度（自研 Ours-v0）。

**先看人话结论：** [`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md)

**状态：** D5 完成 → 下一阶段：**下载×增强联合 v1 + 用户旋钮**（半途收工不进段前）。

| 文档 | 内容 |
|---|---|
| [`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md) | 当前结论（推荐入口） |
| [`docs/DESIGN_JOINT_V1.md`](docs/DESIGN_JOINT_V1.md) | **联合调度 v1 详细设计** |
| [`docs/PARAMS_JOINT_V1.md`](docs/PARAMS_JOINT_V1.md) | **联合 v1 默认参数** |
| [`docs/PREF_JOINT_MATRIX.md`](docs/PREF_JOINT_MATRIX.md) | **差网旋钮矩阵结果** |

| [`docs/ANALYSIS_REPORT.md`](docs/ANALYSIS_REPORT.md) | 阶段分析 |
| [`RESEARCH.md`](RESEARCH.md) | 研究问题 |
| [`DESIGN.md`](DESIGN.md) | v0 设计（松耦合） |
| [`SUMMARY.md`](SUMMARY.md) | 目录与命令 |

```text
cd workspace/multi_enhance
python scripts/run_d5.py --video bbb --traces 0913
```
