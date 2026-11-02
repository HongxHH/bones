# 联合调度 v1 — 参数表（实现用）

**日期：** 2026-10-29  
**配套设计：** [`DESIGN_JOINT_V1.md`](DESIGN_JOINT_V1.md)  
**机读副本：** [`../priors/joint_v1_params.json`](../priors/joint_v1_params.json)

这些是**第一期默认值**：先能跑、旋钮有可见差异；差网跑完再微调，不在这里追求「神参数」。

---

## 0. 立场（写进实现注释）

- 候选很少 → **枚举取最高分 = 该分式下的精确最优**；力气花在分式和约束上。  
- 半途收工 **不** 进下列任何参数。  
- 质量表 Q 大约在 **40～100**（BBB multitask）；加减分按「几分 VMAF 量级」设计，才能扳动相邻工具。

---

## 1. 旋钮取值

| 配置名 | 取值 | 默认 |
|---|---|---|
| `user_courage` | `conservative` / `balanced` / `aggressive` | `balanced` |
| `user_taste` | `balanced` / `sharp` / `clean` | `balanced` |

---

## 2. 胆量 → 可行性与风险

段长记为 `T_seg`（来自 MPD，秒）。下面用「几段」表达，实现时乘 `T_seg`。

| 胆量 | 危险线 `danger_seg` | 宽松 `relax_seg` | 风险权重 `risk_weight` | 人话 |
|---|---:|---:|---:|---|
| 保守 `conservative` | **2.0** | 0.0 | **1.0** | 至少想留约 2 段余量；贴边重罚 |
| 均衡 `balanced` | **1.0** | 0.25 | **0.5** | 约 1 段；中等惩罚 |
| 激进 `aggressive` | **0.5** | 0.5 | **0.2** | 约半段；少罚，但仍要「表上做完」 |

### 可行性（能否进候选）

```text
download_s  = 本段该档字节数 / max(带宽估计, bw_floor)
enhance_s   = max( T[d,t] / enhance_safety , plan_floor[t] )   # 见 §3
danger_s    = danger_seg * T_seg
relax_s     = relax_seg * T_seg
budget_s    = (buff_down_s - buff_enh_s) - danger_s + relax_s

能选 当且仅当：
  合法且 Q 有效
  且 enhance_s <= enhance_max_seg * T_seg     # 防表/底荒谬
  且 download_s + enhance_s <= budget_s
```

| 常数 | 默认 | 含义 |
|---|---:|---|
| `enhance_safety` | 0.9 | 与现网一致：表耗时再除以 0.9（略估保守） |
| `enhance_max_seg` | 1.5 | 增强计划耗时不得超过 1.5 段，否则视为坏表 |
| `bw_floor_bps` | 50_000 | 带宽估计下限，避免除零/极端 | 
| `risk_scale` | 10.0 | 见下 |

### 已能选时的卡顿风险扣分

```text
tightness = (download_s + enhance_s) / max(budget_s, 1e-3)   # 可行时通常 ≤ 1
risk_penalty = risk_weight * tightness * risk_scale
```

均衡且贴边时：`0.5 × 1.0 × 10 ≈ 5` 分，和相邻工具质量差同量级，能起作用又不淹没质量。

---

## 3. 耗时「做完」怎么用（表偏乐观时的保护）

当前 `multitask_latency` 合法格往往只有 **毫秒～几十毫秒**，而差网实测增强常到 **1～5 秒**。  
段前按「做完」算时，必须有保护底，否则可行性几乎形同虚设。

**计划耗时：**

```text
enhance_s = max( T_table[d,t] / 0.9 , plan_floor_s[t] )
```

| tool_id | 名 | `plan_floor_s` | 依据（粗） |
|---:|---|---:|---|
| 0 | none | 0.0 | — |
| 1 | sr_low | 0.4 | 轻超分量级 |
| 2 | sr_med | 0.8 | 中超分 |
| 3 | sr_high | 2.0 | 0921 重增强实测常 ≥2s |
| 4 | sr_x2_med | 1.2 | 介于中、强之间 |
| 5 | restore_sharp | 0.3 | 锐化相对便宜 |

说明：这是**实现前的诚实底**，不是手填质量增益；等耗时表重测准了，可以把 floor 全改成 `0`，只靠表。

另：`latency_adapt` 第一期对比实验默认 **false**（可复现）；修好表之前靠 floor。

---

## 4. 口味 → 加分（Q 分点）

加在查表质量上，单位与 Q 相同。

| 口味 | 超分类 (1,2,3,4) | 恢复/去糊类 (5) | none (0) | 人话 |
|---|---:|---:|---:|---|
| 均衡 `balanced` | 0 | 0 | 0 | 纯看表 |
| 更清晰 `sharp` | **+3.0** | 0 | 0 | 够扳动「中 vs 强」一类差距 |
| 更干净 `clean` | 0 | **+4.0** | 0 | 720 上 restore(96.4) vs x2(93.7) 差额约 2.7，+4 足够稳偏 restore |

以后若加真·去噪工具，归入「恢复/去糊类」同一加成即可。

---

## 5. 总分

```text
score(d,t) = Q[d,t] + taste_bonus(t) - risk_penalty(d,t)
```

- 在所有「能选」的 `(d,t)` 里取 score 最大。  
- 平局：优先更低下载档，再优先更轻工具（偏稳）。  
- 一个都不能选：`(d=0, t=0)`。

第一段（无历史）：与现在一样，可强制 `(0, 0)` 或只允许最低档，避免冷启动误伤。

---

## 6. 下载字节怎么估

```text
segment_bytes ≈ bitrate_bps[d] * T_seg / 8
download_s    = segment_bytes / max(bw_estimate_bps, bw_floor_bps)
```

`bitrate_bps[d]` 来自 MPD representation；`bw_estimate` 来自现有 BandwidthMeter（若无则用 `bw_floor`）。

---

## 7. 第一期建议实验矩阵（参数验收）

固定片 BBB、轨迹 0921，增强器 multitask，半途收工仍开（只统计）：

| 跑次 | courage | taste | 看什么 |
|---|---|---|---|
| A | balanced | balanced | 默认联合 vs 旧 tool_greedy |
| B | conservative | balanced | 卡顿是否下降、工具是否变轻 |
| C | aggressive | balanced | 质量/工具是否变重、卡顿是否可接受 |
| D | balanced | sharp | 超分类占比是否上升 |
| E | balanced | clean | restore 占比是否上升 |
| F | — | — | none / 旧 tool_greedy 对照 |

**旋钮合格线（第一期）：**

- B 的 stall ≤ A，或明显更偏轻工具；  
- C 的平均 Q 或重工具占比 > A，stall 不爆炸（例如不超过旧 tool_greedy 的 1.3× 太多）；  
- D/E 工具类型分布肉眼可分。

---

## 8. 以后允许动的旋钮（调参次序）

1. `plan_floor_s[sr_high]`（表未准时影响最大）  
2. `danger_seg` 三档  
3. `taste` 加减分  
4. `risk_weight` / `risk_scale`  
5. 表准了之后：把 floor 降为 0  

不要一上来同时拧一堆。

---

## 9. 和旧 slack 公式的关系

旧：`slack = down - enh - 1*T_seg`，再要求 `slack ≥ T_table`（表还偏小）。  

新：危险线由**胆量**显式给出；预算里同时扣**下载+增强**；增强用 **floor 后的做完时间**。  
不是把 κ 再拧一格，而是换联合预算。
