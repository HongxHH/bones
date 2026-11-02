# D5 结果：BBB 六臂消融（0913 + 0921）

**人话结论请先看：** [`CONCLUSIONS.md`](CONCLUSIONS.md)  
**状态：两轨迹主表完成（2026-10-29）；1080 / TOS 仍待做**  
**权威 JSON：** [`d5_bbb_0913.json`](../results/canonical/d5_bbb_0913.json) · [`d5_bbb_0921.json`](../results/canonical/d5_bbb_0921.json)

说明：同一张质量表查分；关闭在线改耗时；无窗口 + 网络轨迹回放。

**表头人话：** QOE≈观感分 · stall≈卡顿秒数 · FC≈半途收工次数 · ours=现在这版 · none=不增强 · fixed_high=永远最强 · no_hd=去掉高清工具 · no_slack=选时不管缓冲 · no_fc=不许半途收工

---

## BBB × 0913（较轻网）

| 臂 | QOE | avg_vmaf | stall (s) | FC 段数 | 工具摘要 |
|---|---:|---:|---:|---:|---|
| **ours** | 59.67 | 73.80 | **0.66** | 1 | none14 / L2:45 / L3:79 / restore:21 |
| none | 47.05 | 63.20 | 3.72 | 0 | 全 none |
| fixed_high | **63.63** | 74.17 | 5.33 | 37 | 几乎全 L3 |
| no_hd | 61.91 | 73.77 | 0.65 | 0 | 无 restore；none 更多 |
| no_slack | **64.54** | **75.92** | 1.78 | 28 | 更猛选 L3+restore |
| no_fc | 56.13 | 72.39 | 0.64 | 0 | 有 restore，无 FC |

## BBB × 0921（差网）

| 臂 | QOE | avg_vmaf | stall (s) | FC 段数 | 工具摘要 |
|---|---:|---:|---:|---:|---|
| **ours** | 55.81 | 70.11 | 20.55 | 3 | none26 / L2:45 / L3:65 / restore:22 |
| none | 47.90 | 62.17 | **44.61** | 0 | 全 none |
| fixed_high | **62.33** | 72.50 | 20.66 | 48 | 几乎全 L3 |
| no_hd | 57.46 | 70.11 | 24.64 | 1 | 无 restore；none 更多 |
| no_slack | 61.83 | **73.38** | **11.85** | 41 | 几乎全 L3+restore |
| no_fc | 54.93 | 70.77 | 30.88 | 0 | 工具分布近 ours |

---

## 对照门禁（跨轨迹）

| 期望 | 0913 | 0921 |
|---|---|---|
| ours ≫ none | **成立**（QOE +12.6，stall −3.1） | **成立**（QOE +7.9，stall −24） |
| ours vs fixed：不显著差或 stall 更好 | **stall 大胜**；QOE −4 | stall **持平**；QOE −6.5 → 差网下「会选」未赢 fixed |
| ours > no_hd | QOE **未赢**；stall 持平 | QOE **未赢**；stall **略胜**（−4.1s） |
| ours 优于 no_slack（尤其 stall） | stall **成立**（0.66 vs 1.78） | **反号**：no_slack stall/QOE 都更好 |
| ours 优于 no_fc | QOE **成立**；stall 相近 | **双成立**（QOE +0.9，stall −10.3） |

---

## 综合结论（两轨迹）

1. **相对 none 稳健成立**——增强有用，差网下 stall 收益尤其大。  
2. **相对 fixed_high 不稳健**：轻网靠 stall 叙事；差网 stall 打平、QOE 落后。v0 缓冲门可能过保守，或需与 ABR 协同。  
3. **fast-complete 价值加强**：差网关 FC 后 stall 明显变差——RQ3 证据比 0913 更硬。  
4. **slack 是旋钮，不是「去掉就差」**：轻网无 slack → 更卡；差网无 slack → 反而更流畅且更高 QOE（靠更猛选 + 大量 FC）。当前阈值/可行集规则需复盘，不宜写成「slack 必有益」。  
5. **no_hd 仍未显示「去掉就 QOE 变差」**（两轨迹 QOE 都略高）；差网 stall 上 ours 略好。多类型高清声称应降级为「可用/条件启用」，或改专项指标后再验。

---

## 下一步（仍属 D5）

- [x] BBB × 0913  
- [x] BBB × 0921  
- [x] **复盘 slack** → [`D5_SLACK_DIAG.md`](D5_SLACK_DIAG.md)；κ 扫描见 `d5_kappa_0921.json`（只降 κ 抬 QOE 但 stall 变差；需 FC 感知）  
- [ ] 1080 H0（RQ1）  
- [ ] TOS × 0913 抽检（确认不反号即可）

跑法：`python scripts/run_d5.py --video bbb --traces 0921`
