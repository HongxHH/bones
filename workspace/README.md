# Workspace — 个人实验工作区

本目录用于存放基于 BONES 基础环境的个人扩展实验，与原始 BONES 代码隔离。

## 目录结构

- `algorithms/` — 新算法实现（可 import 原始 `control/` 中的基类或工具）
- `experiments/` — 新实验脚本（批量运行、对比实验等）
- `configs/` — 实验配置文件
- `results/` — 实验结果输出（gitignore，不入库）
- `notes/` — 实验笔记、想法记录、论文阅读笔记

## 约定

- 原始 BONES 代码（`simulation/`、`control/`、`config/`、`experiment/` 等）保持不动，作为基线
- 个人扩展全部收敛在本目录下，便于后续"基线 vs 我们的方法"对比
- 引用原始模块时使用相对路径，例如 `from control import BONES`
