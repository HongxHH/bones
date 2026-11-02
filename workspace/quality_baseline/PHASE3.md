# 阶段 3：段内恢复执行器

**状态：三轨迹 × 四臂跑完（2026-10-27）；判决见文末**

## 配置

| CLI / player 字段 | 蛮干 | bicubic 尾 | F 尾 | 整段 F |
|---|---|---|---|---|
| `fixed_enhance_action` | 3 | 3 | 3 | 1 |
| `recovery_mode` | `none` | `bicubic` | `fallback` | `none` |
| `fallback_enhance_action` | — | — | 1 | — |
| `PACNET_BURN_K` | 5 | 5 | 5 | 5 |
| `latency_adapt` | false | false | false | false |

示例：

```text
python play.py --mod_nes always_greedy --mod_enhancer imdn --mod_renderer opencv \
  --fixed_enhance_action 3 --recovery_mode fallback --fallback_enhance_action 1 \
  --use_fast_complete true --latency_adapt false \
  -t data/3Glogs/report.2010-09-13_1003CEST.json
```

环境变量：`PACNET_BURN_K=5`。

## 行为

- 触发：每 15 帧，`剩余帧 × 已观测每帧耗时 > 距开播 − 一帧裕量`。
- `bicubic`：前缀保留，尾巴 `DecoderWrapper`。
- `fallback`：先换 `fallback_enhance_action` 模型继续；再触发则 bicubic 尾。
- `none`：不工作量截断；播放追上则 **丢前缀**（作废帧对照）。
- 恢复臂播放追上：提交前缀 + bicubic 尾（不丢）。
- `segment_stats` 增加 `frame_enhance_levels` / `recovery_mode` / `fallback_triggered` 等。

## 产物

- 播放器：`config.py`、`always_greedy.py`、`enhancer_imdn_always.py`、`module_composer.py`
- Prism：`experiment_spec.py`、`attach.py`；工作区 `prism/workspaces/recovery_four_arm.yaml`
- 单测：`player/test/test_recovery_work_deadline.py`
- 汇总：`phase3_summary_all.json`（权威）；合并帮手在 `_scratch/`  
- 说明：`player/output` 已按 L0 清空，原始 per-run json 若需重汇只能重跑 `run_phase3_arms.py`

## 结果（K=5, C=3, F=1，3G 三轨迹）

| arm | 0913 QoE / stall | 0914 QoE / stall | 0921 QoE / stall | waste |
|---|---|---|---|---|
| brute_C3 | **60.5** / 5.9s | 49.5 / **927s** | **55.3** / 88.7s | ~3k 帧 |
| bicubic_tail | 52.0 / 10.8s | 48.7 / 474s | 49.2 / 20.0s | 0 |
| fallback_F1 | 54.5 / 21.2s | 50.1 / 522s | 53.6 / **16.2s** | 0 |
| whole_F1 | 54.0 / **1.9s** | **51.8** / 496s | 51.4 / 20.3s | ~50–80 |

F-tail vs whole-F：`dQoE / dStall` = +0.5/+19s（0913）、−1.7/+26s（0914）、+2.2/−4s（0921）。

## 结论（NO-GO 作默认策略）

1. **机制成立**：恢复臂把 abort waste 清到 0，并稳定触发 fast-complete（约 50–85 段/跑）。
2. **F 尾相对整段 F 不稳**：仅 0921 同时更好；0913 小幅 QoE 换来大幅卡顿；重载 0914 全面劣于 whole_F。
3. **蛮干 QoE 常最高**，但浪费算力且在差网下卡顿爆炸（0914），不可当稳健默认。
4. **论文/系统默认**：重载偏 `whole_F`；恢复执行器作可选消废路径，不宜宣称 F-tail 全面优于整段低档。
