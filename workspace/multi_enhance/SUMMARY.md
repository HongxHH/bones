# multi_enhance 查阅入口

**状态：** D0–D4 完成；D5 消融与差网诊断完成（2026-10-29）  

**先读这个（人话结论）：** [`docs/CONCLUSIONS.md`](docs/CONCLUSIONS.md)  
**阶段报告：** [`docs/ANALYSIS_REPORT.md`](docs/ANALYSIS_REPORT.md)  
**研究方向：** [`RESEARCH.md`](RESEARCH.md)

**一句话：** 增强有用；差网暴露旧「先下载再增强」过保守。下一代是 `pref_joint`（下载×增强联合枚举 + 胆量/口味旋钮）。

---

## 1. 结论速查

| 问题 | 人话结论 |
|---|---|
| 要不要做增强调度？ | **要。** 比不增强明显好。 |
| 现在这版够不够好？ | **还不够。** 轻网还行；差网输给「大胆选 + 半途收工」。 |
| 半途收工重要吗？ | **重要。** 差网关掉会更卡。 |
| 高清专用工具？ | 会选到；但还没证明「去掉就变差」。 |
| 下一步？ | 用 `pref_joint` 冒烟 + 差网矩阵验收旋钮；半途收工只当保险丝。 |

权威数字：[`results/canonical/`](results/canonical/)  
消融表：[`docs/D5_RESULTS.md`](docs/D5_RESULTS.md) · 差网诊断：[`docs/D5_SLACK_DIAG.md`](docs/D5_SLACK_DIAG.md)

---

## 2. 目录地图

```text
workspace/multi_enhance/
  README.md / SUMMARY.md / DESIGN.md / RESEARCH.md
  docs/
    CONCLUSIONS.md      ← 给人看的当前结论（从这里开始）
    ANALYSIS_REPORT.md  ← 阶段分析（人话）
    D5_RESULTS.md       ← 消融数字
    D5_SLACK_DIAG.md    ← 差网为什么大胆版更好
    D3.md / D4.md / D5.md
  scripts/              跑实验、建先验、诊断
  results/canonical/    只认这里的权威 JSON
```

播放器侧关键文件：

| 路径 | 用途 |
|---|---|
| `player/multitask_bbb_*.json` / `multitask_tos_*.json` | 质量/耗时表 |
| `player/.../nes/tool_greedy.py` | 旧：先下载再增强 |
| `player/.../nes/pref_joint.py` | **新：联合枚举 + 用户旋钮** |
| `priors/joint_v1_params.json` | 联合默认参数 |
| `player/.../enhancer/enhancer_multitask.py` | 多工具增强执行 |
| `player/output/data/d5/` | 各次实验原始结果 |

---

## 3. 工具编号（需要读日志时）

| 编号 | 意思 | 大致用在 |
|---:|---|---|
| 0 | 不增强 | 任意清晰度 |
| 1–3 | 超分 轻/中/强 | 低清档 |
| 4 | 轻量 ×2 | 720 |
| 5 | 同分辨率增强（现为锐化占位） | 720/1080；1080 尚未正式入表 |

TOS 画面比例和 BBB 不同（显示约 1920×800），别混用几何假设。

---

## 4. 阶段是否通过

| 阶段 | 通过要点 |
|---|---|
| D1 | 能跑，段上有工具编号 |
| D2 | 质量/耗时来自实测，不手填 |
| D3 | BBB 上相对不增强明显更好 |
| D4 | TOS 方向不反；旧表已作废重测 |
| D5 | 消融跑完；差网诊断清楚：太胆小，需把半途收工纳入决策 |

---

## 5. 已知坑

1. 跑 TOS 要用对输入，并显式带片子路径（曾被默认参数盖成 BBB）。  
2. 观感分来自**查表**，对照实验必须同一张表。  
3. 旧 TOS/imdn 表不可信。  
4. 锐化占位会被 VMAF 偏爱，换真模型前别过度宣称画质。  
5. 预计耗时表比真实快很多——改策略时要心里有数。

---

## 6. 常用命令

```text
cd workspace/multi_enhance
python scripts/build_priors.py
python scripts/run_d5.py --video bbb --traces 0913

cd player
python play.py --mod_nes pref_joint --mod_enhancer multitask \
  --mod_renderer headless --mod_downloader trace --mod_status off \
  --user_courage balanced --user_taste balanced \
  -t data/3Glogs/report.2010-09-13_1003CEST.json -i data/video_mp4box/bbb.mpd
```

单测：`cd player && pytest test/test_pref_joint.py test/test_tool_greedy.py`
