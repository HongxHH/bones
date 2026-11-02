# 设计：多模型增强 + 段前调度（v0）

**日期：** 2026-10-27 · **实现状态：D1–D4 GO（2026-10-28）**  
**查阅：** 主题与 RQ 见 [`RESEARCH.md`](RESEARCH.md)；结论与目录见 [`SUMMARY.md`](SUMMARY.md)；阶段细节 [`docs/`](docs/)。  
**地位：** 接口与原则仍以本文为准；实现冲突先改本文再改代码。

---

## 1. 目标与非目标

### 目标

在现有 istream 播放器上，建立：

1. **可插拔多类型增强工具箱**（超分 / 画面恢复·锐化·去块 / 日后去噪），每段至多选一个工具；
2. **段前可行性调度**：在截止时间允许的工具里选期望质量更高者；
3. **v0 与 ABR 松耦合**：下载档由缓冲类逻辑决定；增强侧做自有可行集选择与段中执行。  
   **后续可演进**为自有 ABR×增强协同；**不**仿照/沿用既有联合优化实现当本方法（对照除外）。

### 非目标（v0）

- **不**把既有联合优化算法当作本方向方法去仿制或「接着改」；
- 不做 SR+恢复+插帧等 **级联组合**（组合爆炸与旧数据坑）；
- 不在线更新 VMAF 质量表；
- 不以 RL / 深度策略网络为 v0；
- 不把 PACNet halt 当段前决策（它属于执行层段中改深度）。

---

## 2. 与现有代码的对齐

| 本设计概念 | 现有模块 |
|---|---|
| 观测 | `DownloadBufferImpl` / `EnhanceBufferImpl`（continuous）、`BandwidthMeter`、`StatusSnapshot`、`Analyzer.segment_stats` |
| 工具表 | 今日 5×5 `latency_table` / `quality_table` 的 **列语义** 改为 tool_id；非法格继续用 `−inf` / 极大延迟 |
| 段前决策 | 新 `mod_nes`（建议名 `tool_greedy`）；契约仍是 `update_selection → (download_actions, enhance_actions)` |
| 执行 | **`mod_enhancer=multitask`**：Always 子类，业务在独立模块；Prism 默认 `imdn` 不改行为。旧 USM 五动作 → `multitask_legacy` |
| 延迟自适应 | 复用 `latency_adapt` → `_update_latency_ewma` + 未访问格缩放；可选 `use_content_latency` KNN |
| 下载 | v0 复用缓冲映射 ABR；与增强分决策。未来自有协同可改此边界 |

Scheduler 契约不变：NES 打标 → enqueue → Enhancer `dequeue` → `download_buffer.replace`。

---

## 3. 五层架构

```text
① Observation   缓冲、带宽、分辨率、段索引、GPU（可选）、上一段 stats
② Tool Registry tool_id → 算子实现、合法分辨率、先验 T/Q、能力标签
③ Policy (NES)  段前：feasible 过滤 + argmax Q；输出 enhance_action=tool_id
④ Execution     Enhancer：按 tool 跑完整段或段中降级/截断
⑤ Adaptation    仅延迟在线 EWMA；质量表离线；实验可关 latency_adapt
```

### ③ vs ④ 的边界（重要）

- **段前**：只决定本段承诺的 `tool_id`。
- **段中**：超时、playback 追上 → 走 Always 已有 recovery（bicubic / fallback tool），写入 `frame_enhance_levels`；**不**回调 NES 重选整段工具。
- PACNet 式 halt：若未来接入渐进模型，仍算 ④，不进 ③。

---

## 4. 工具注册表（多类型，非仅 SR）

### 4.1 v0 菜单（列）

| tool_id | 类型 | 含义 | 合法分辨率（初值） | 证据 |
|---|---|---|---|---|
| 0 `none` | — | 不增强 | 全部 | — |
| 1 `sr_low` | SR | IMDN low | 240/360/480 | Phase1 / 既有表 |
| 2 `sr_med` | SR | IMDN medium | 240/360/480 | 同上 |
| 3 `sr_high` | SR | IMDN high | 240/360/480 | 同上 |
| 4 `sr_x2_med` | SR（高清轻量） | IMDN×2→对齐 1080 | **720** | H0 10 段，均 Δ≈+2.4，9/10≥+1 |
| 5 `restore_sharp` | 画面增强 | 同分辨率锐化/去糊代理（先 unsharp，可换模型） | **720/1080** | H0 restore 均 Δ≈+5.1；**1080 Q 仍未测 → 表内 −inf，不可选** |
| 6+ | 去噪 / 去块 | 预留 | 待 H0 同类门禁 | 无实测前不得填正增益、不得被策略选中 |

先验文件：`player/multitask_{bbb,tos}_quality.json`、`multitask_latency{,_tos}.json`（由 `scripts/build_priors.py` 生成；来源见 `priors/meta_*.json`）。

> ID 与列下标一致，便于沿用二维表；实现可用枚举+注册 dict，表只是 NES 视图。

### 4.2 表轴

- **行** = `download_action`（分辨率档），与今日一致。  
- **列** = `tool_id`。  
- 非法：`Q[r,t] = −inf`，`T[r,t]` 极大或同等屏蔽（对齐 multitask 对高清 SR 的写法，但 **开放** 高清 restore / `sr_x2_med`）。

### 4.3 注册表字段（每个 tool）

```text
id, name, kind ∈ {none, sr, restore, denoise, ...}
legal_download_actions: set[int]
run(segment, frames) -> frames          # 执行钩子
t_prior_s_per_frame: 先验每帧秒       # 来自测量 JSON
q_prior: 段或分辨率级质量/增益         # 来自 data/quality 或探针
```

质量数字 **禁止手填随机**；来源写进 json `_metadata`（脚本、时间、口径）。

---

## 5. 观测与输入处理

### 5.1 段前决策默认输入（按优先级）

1. `buff_enh_s` = `enhance_buffer.buffer_level(continuous=True)`  
2. `buff_down_s` = `download_buffer.buffer_level(continuous=True)`  
3. 本段将采用的 `download_action` / 分辨率（先由下载逻辑算出，再选 tool）  
4. `T[r, :]` = `enhancer.get_latency_table(segment_index?)` 已含 safe_factor 与可选内容缩放  
5. `Q[r, :]` = `enhancer.get_quality_table()`  

可选后加：带宽、离线内容特征、GPU EWMA——**不**进 v0 必选。

### 5.2 可修时间（slack）

对齐并显式化 `buffer_greedy` 思想：

```text
# 单位：秒；T_seg 为段时长（如 4s）
T_tool = latency_table[r, tool]           # 已是「整段增强秒」量级（Always 的 get_latency_table）
slack  = buff_down_s - buff_enh_s - T_seg # 与 buffer_greedy 条件同构
feasible(tool) =
    legal(r, tool)
    and Q[r,tool] > −inf
    and T_tool < T_seg / enhance_safety     # 单段墙钟粗门（可选保留）
    and slack >= T_tool                     # 缓冲松弛足够
```

`enhance_safety`（如 0.9）继续表示「把时间估保守」。

若 `feasible` 空集 → `none`。

### 5.3 决策规则（算法 v0）

```text
download = BufferMap(buff_down)            # 或既有 always_greedy 下载支路
r = download
pick = argmax_{t in feasible(r) ∪ {none}} Q[r, t]
# none 的 Q 定义为该分辨率 base（bicubic/原片）质量
return download, pick
```

**理论归属（v0）：** deadline / 缓冲约束下的可行集贪心；下载与增强分决策。协同/联合形态留待自研演进。

**学习：** v0 不用；有多轨迹日志后，才考虑 contextual bandit 替代 argmax Q。

---

## 6. 延迟动态更新

| 模式 | 行为 |
|---|---|
| `latency_adapt=false` | 冻结先验表；对比实验 / 可复现跑 |
| `latency_adapt=true` | 段完成或 abort 后 EWMA 更新 `(r,t)` 每帧延迟；未访问格用已访问平均缩放；可选内容 KNN |

质量表始终离线。系统演示默认建议 **打开** `latency_adapt`；论文对照表可关。

Multitask 里「每段直接覆盖一格、无 EWMA/recovery」的路径 **不作为** v0 主路径。

---

## 7. 执行层要点

1. `enhance_action` 解释为 `tool_id`，分发到 SR 模型池或 restore/denoise 算子。  
2. `scale==1` 时 **禁止** 误走 ×2/×3/×4 SR；走 restore 或 `none`。  
3. 720 的 `sr_x2_med`：×2 后再 bicubic 对齐显示分辨率（与 H0 一致）。  
4. 超时：优先 Always 的 `recovery_mode`（fallback 到更轻 tool 或 bicubic 尾），统计进 `segment_stats`。  
5. Analyzer / telemetry 字段尽量复用；新增 `tool_name` / `tool_kind` 便于复盘。

---

## 8. 素材与模型策略

| 阶段 | 素材 | 模型 |
|---|---|---|
| 实现与调通 | BBB | 现有 IMDN + `restore_sharp` 占位 |
| 泛化验收 | + TOS | 同一注册表；若高清 tool 全灭则收窄声称 |
| 补工具 | 按缺口 | 仅当缺「压缩恢复/去噪」正增益时再找或微调；leave-out 防泄漏 |

---

## 9. 实现阶段与门禁

| 阶段 | 产出 | GO |
|---|---|---|
| D0 | 本文 + 枚举/表 JSON schema | 2026-10-27 已写 |
| D1 | `tool_greedy` NES + `imdn_tools` 分发（restore + sr_x2_med） | **GO**：单测通过；headless 已选 `sr_med`/`restore_sharp` 并完成增强；`segment_stats` 含 `tool_id` |
| D2 | 先验 T/Q 从测量写入（禁手填） | **GO**（2026-10-27）：`player/multitask_bbb_quality.json` + `multitask_latency.json`；脚本 `scripts/build_priors.py`；720 Δ(restore/x2) 与 H0 对账 |
| D3 | 1～2 条 3G 轨迹：vs `fixed sr_high` / vs `none` | **GO**（见 `D3.md`）：0913 大胜；0921 QoE 持平、stall +4s 可接受 |
| D4 | TOS 冒烟 | **GO**（见 `docs/D4.md`）：对齐 Q 表后 vs fixed ΔQOE≈−1.6、stall 持平；远好于 none |
| D5 | 消融 + 1080 闭环 | BBB×0913+0921 见 `docs/D5_RESULTS.md`；1080/TOS 待做 |

---

## 10. 明确拒绝的做法

1. 再生成一份「随机增益」multitask json；  
2. 用估计值开放 `−inf` 格；  
3. v0 上 SR|restore|denoise 任意笛卡尔积；  
4. 段中 NES 重规划整段工具；  
5. 为调度单独再写一套缓冲语义。

---

## 11. 开放问题（实现前可拍板）

1. `restore_sharp` 最终算子：先 ffmpeg/torch USM，还是等压缩恢复模型？  
2. 下载侧固定用 `always_greedy` 缓冲映射，还是 `buffer_greedy` 全套？  
3. `latency_adapt` 默认开还是关（建议：系统 on，A/B 可切）？  
4. tool 列是否突破 5 列（建议直接扩表宽，避免塞进旧 0–4 语义打架）？

---

## 12. 参考

- 播放器：`modules/nes/buffer_greedy.py`（可行集贪心）、`enhancer_imdn_always.py`（EWMA/recovery）、`enhancer_multitask.py`（多类型 ID 先例，执行不完整）  
- 探针：`../hd_enhance_probe/README.md`  
- 旧线教训：`../vbr_sims/multitask_usm/README.md`（机制未证伪，数据与第二算子曾缺位）
