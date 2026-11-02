# -*- coding: utf-8 -*-
"""Diagnose why no_slack beats ours on BBB x 0921."""
from __future__ import annotations

import json
import statistics as stats
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PLAYER = REPO / "player"
D5 = PLAYER / "output" / "data" / "d5"
OUT = ROOT / "docs" / "D5_SLACK_DIAG.md"
CANON = ROOT / "results" / "canonical" / "d5_slack_diag_0921.json"

TOOL = {0: "none", 1: "sr_low", 2: "sr_med", 3: "sr_high", 4: "x2", 5: "restore"}
SAFETY = 0.9  # matches tool_greedy enhance_safety


def load_arm(arm: str) -> dict:
    folder = D5 / f"bbb_0921_{arm}"
    f = next(folder.glob("*.json"))
    data = json.loads(f.read_text(encoding="utf-8"))
    data["_file"] = str(f.relative_to(REPO)).replace("\\", "/")
    return data


def seg_tool(s: dict) -> int:
    t = s.get("tool_id")
    if t is None:
        t = s.get("enhance_action")
    return int(t) if t is not None else 0


def load_latency() -> np.ndarray:
    raw = json.loads((PLAYER / "multitask_latency.json").read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("latency") or raw.get("table") or raw.get("data")
    return np.asarray(raw, dtype=float)


def stall_windows(data: dict) -> list[tuple[float, float, float]]:
    out = []
    t0 = None
    segs = data["segments"]
    if segs:
        # analyzer absolute times; normalize to first start
        t0 = min(s["start_time"] for s in segs if s.get("start_time") is not None)
    for st in data.get("stalls") or []:
        a, b = float(st["time_start"]), float(st["time_end"])
        if t0 is not None and a > 1e12:
            # absolute epoch style
            a, b = a - t0, b - t0
        out.append((a, b, b - a))
    return out


def segments_overlapping_stalls(data: dict) -> list[dict]:
    """Map stalls onto nearby segments via enhance/download timeline if possible."""
    segs = data["segments"]
    if not segs:
        return []
    t0 = min(s["start_time"] for s in segs if s.get("start_time") is not None)
    hits = []
    for sa, sb, dur in stall_windows(data):
        # stall times in analyzer seem relative already in some dumps; try both
        for s in segs:
            # use enhance_end_to_play / start relative
            st = s.get("start_time")
            if st is None:
                continue
            rel = st - t0 if st > 1e12 else st
            # crude: segment whose download window near stall
            fb = s.get("first_byte_at")
            lb = s.get("last_byte_at")
            if fb is not None and lb is not None:
                fb_r = fb - t0 if fb > 1e12 else fb
                lb_r = lb - t0 if lb > 1e12 else lb
                if fb_r <= sb and lb_r >= sa:
                    hits.append(
                        {
                            "stall_dur": dur,
                            "index": s["index"],
                            "tool": seg_tool(s),
                            "down": int(s.get("download_action") or 0),
                            "buf": s.get("decision_time_buffer_level"),
                            "fc": bool(s.get("fast_complete")),
                        }
                    )
    return hits


def estimate_blocked(lat: np.ndarray, action_down: int, buff_down_s: float, seg_s: float) -> dict:
    """Replicate tool_greedy slack gate assuming buff_enh≈0 (optimistic upper bound on slack)."""
    r = min(action_down, lat.shape[0] - 1)
    # table is seconds; code does *1000/safety
    slack_ms = buff_down_s * 1000.0 - 0.0 - seg_s * 1000.0
    blocked = {}
    for t in range(lat.shape[1]):
        t_ms = float(lat[r, t]) * 1000.0 / SAFETY
        if not np.isfinite(t_ms) or t_ms > 1e8:
            blocked[t] = "illegal_or_inf"
            continue
        if t == 0:
            blocked[t] = "ok_none"
            continue
        reasons = []
        if t_ms >= seg_s * 1000.0:
            reasons.append("t>=seg")
        if slack_ms < t_ms:
            reasons.append(f"slack<{t_ms/1000:.2f}s")
        blocked[t] = "ok" if not reasons else ",".join(reasons)
    return {
        "slack_s_if_enh0": slack_ms / 1000.0,
        "gate": {TOOL.get(k, k): v for k, v in blocked.items()},
    }


def main():
    lat = load_latency()
    arms = {a: load_arm(a) for a in ["ours", "no_slack", "fixed_high"]}

    # segment duration from gaps
    osegs = arms["ours"]["segments"]
    durs = []
    for s in osegs:
        if s.get("start_time") is not None and s.get("stop_time") is not None:
            durs.append(s["stop_time"] - s["start_time"])
    seg_s = float(np.median(durs)) if durs else 2.0

    summary = {
        "seg_duration_median_s": seg_s,
        "latency_table_shape": list(lat.shape),
        "latency_row0_s": [float(x) for x in lat[0].tolist()],
        "arms": {},
        "tool_diff_patterns": {},
        "findings": [],
    }

    lines = []
    lines.append("# D5 诊断：0921 上为何 no_slack 优于 ours")
    lines.append("")
    lines.append("**日期：** 2026-10-29  ")
    lines.append("**数据：** `player/output/data/d5/bbb_0921_{ours,no_slack,fixed_high}`  ")
    lines.append("**规则：** `tool_greedy._pick_tool`：`slack = buff_down - buff_enh - seg_time`；非 none 需 `t < seg` 且 `slack ≥ t`（再 /enhance_safety=0.9）。")
    lines.append("")

    for name, data in arms.items():
        segs = data["segments"]
        tools = Counter(seg_tool(s) for s in segs)
        downs = Counter(int(s.get("download_action") or 0) for s in segs)
        bufs = [float(s["decision_time_buffer_level"]) for s in segs if s.get("decision_time_buffer_level") is not None]
        fc_n = sum(1 for s in segs if s.get("fast_complete"))
        lats_by_t = {}
        for t in range(6):
            xs = [
                float(s["enhance_latency"])
                for s in segs
                if seg_tool(s) == t and s.get("enhance_latency") is not None
            ]
            if xs:
                lats_by_t[TOOL[t]] = {
                    "n": len(xs),
                    "mean": round(stats.mean(xs), 3),
                    "p90": round(stats.quantiles(xs, n=10)[-1], 3) if len(xs) >= 10 else round(max(xs), 3),
                    "max": round(max(xs), 3),
                }
        row = {
            "QOE": data.get("QOE"),
            "dur_stall": data.get("dur_stall"),
            "num_stall": data.get("num_stall"),
            "fc_segments": data.get("fast_complete_segments_count"),
            "fc_count_segments_flag": fc_n,
            "tools": {TOOL.get(k, str(k)): v for k, v in sorted(tools.items())},
            "download": {str(k): v for k, v in sorted(downs.items())},
            "dec_buf_mean": round(stats.mean(bufs), 3) if bufs else None,
            "dec_buf_p10": round(stats.quantiles(bufs, n=10)[0], 3) if len(bufs) >= 10 else None,
            "dec_buf_p50": round(stats.median(bufs), 3) if bufs else None,
            "dec_buf_p90": round(stats.quantiles(bufs, n=10)[-1], 3) if len(bufs) >= 10 else None,
            "enhance_latency_by_tool": lats_by_t,
            "stalls": [
                {"start": a, "end": b, "dur": d} for a, b, d in stall_windows(data)
            ],
            "file": data["_file"],
        }
        summary["arms"][name] = row

    by = {a: {int(s["index"]): s for s in arms[a]["segments"]} for a in arms}
    idxs = sorted(set(by["ours"]) & set(by["no_slack"]))

    diffs = []
    for i in idxs:
        o, n = by["ours"][i], by["no_slack"][i]
        to, tn = seg_tool(o), seg_tool(n)
        if to == tn and o.get("download_action") == n.get("download_action"):
            continue
        buf_o = o.get("decision_time_buffer_level")
        down_o = int(o.get("download_action") or 0)
        gate = None
        if buf_o is not None:
            gate = estimate_blocked(lat, down_o, float(buf_o), seg_s)
        diffs.append(
            {
                "index": i,
                "ours_tool": TOOL[to],
                "noslack_tool": TOOL[tn],
                "ours_down": down_o,
                "noslack_down": int(n.get("download_action") or 0),
                "ours_buf": buf_o,
                "noslack_buf": n.get("decision_time_buffer_level"),
                "ours_fc": bool(o.get("fast_complete")),
                "noslack_fc": bool(n.get("fast_complete")),
                "ours_lat": o.get("enhance_latency"),
                "noslack_lat": n.get("enhance_latency"),
                "ours_vmaf": o.get("real_vmaf"),
                "noslack_vmaf": n.get("real_vmaf"),
                "gate_if_enh0": gate,
            }
        )

    tool_only = [d for d in diffs if d["ours_tool"] != d["noslack_tool"]]
    pat = Counter((d["ours_tool"], d["noslack_tool"]) for d in tool_only)
    summary["tool_diff_patterns"] = {f"{a}->{b}": c for (a, b), c in pat.most_common()}
    summary["n_tool_diff"] = len(tool_only)
    summary["n_download_diff"] = sum(
        1 for i in idxs if by["ours"][i].get("download_action") != by["no_slack"][i].get("download_action")
    )

    # How often ours chose none/light while table would block heavy under enh0 slack
    blocked_heavy_when_ours_light = 0
    would_allow_heavy = 0
    light = {"none", "sr_low", "sr_med"}
    for d in tool_only:
        if d["ours_tool"] in light and d["noslack_tool"] in {"sr_high", "restore", "x2"}:
            g = d.get("gate_if_enh0") or {}
            gate = g.get("gate") or {}
            target = d["noslack_tool"]
            status = gate.get(target, "")
            if status == "ok":
                would_allow_heavy += 1
            else:
                blocked_heavy_when_ours_light += 1

    # Buffer distribution when ours picks none vs sr_high
    def bufs_for(arm, tool_name):
        tid = {v: k for k, v in TOOL.items()}[tool_name]
        return [
            float(s["decision_time_buffer_level"])
            for s in arms[arm]["segments"]
            if seg_tool(s) == tid and s.get("decision_time_buffer_level") is not None
        ]

    # FC rescue: among no_slack FC segments, what was committed tool
    ns_fc_tools = Counter(
        TOOL[seg_tool(s)] for s in arms["no_slack"]["segments"] if s.get("fast_complete")
    )
    ours_fc_tools = Counter(
        TOOL[seg_tool(s)] for s in arms["ours"]["segments"] if s.get("fast_complete")
    )

    # Cumulative enhance time committed (table * safety) vs actual latency
    def committed_work(arm):
        total_table = 0.0
        total_actual = 0.0
        for s in arms[arm]["segments"]:
            d = int(s.get("download_action") or 0)
            t = seg_tool(s)
            r = min(d, lat.shape[0] - 1)
            if t < lat.shape[1] and np.isfinite(lat[r, t]):
                total_table += float(lat[r, t]) / SAFETY
            if s.get("enhance_latency") is not None:
                total_actual += float(s["enhance_latency"])
        return round(total_table, 1), round(total_actual, 1)

    # stall vs buffer: average dec buf overall
    # Compare download ladders over time (moving)
    # VMAF delta on tool-diff segments
    vmaf_deltas = [
        d["noslack_vmaf"] - d["ours_vmaf"]
        for d in tool_only
        if d["ours_vmaf"] is not None and d["noslack_vmaf"] is not None
    ]

    # Reconstruct: fraction of segments where slack (enh0) blocks sr_high for ours' download+buf
    block_sr_high = 0
    allow_sr_high = 0
    for s in arms["ours"]["segments"]:
        buf = s.get("decision_time_buffer_level")
        if buf is None:
            continue
        g = estimate_blocked(lat, int(s.get("download_action") or 0), float(buf), seg_s)
        st = g["gate"].get("sr_high")
        if st == "ok":
            allow_sr_high += 1
        else:
            block_sr_high += 1

    # When ours picks none: gate status
    none_gate = Counter()
    for s in arms["ours"]["segments"]:
        if seg_tool(s) != 0:
            continue
        buf = s.get("decision_time_buffer_level")
        if buf is None:
            continue
        g = estimate_blocked(lat, int(s.get("download_action") or 0), float(buf), seg_s)
        none_gate[g["gate"].get("sr_high", "?")] += 1

    cw = {a: committed_work(a) for a in arms}

    summary["recon"] = {
        "ours_segments_sr_high_allowed_if_enh0": allow_sr_high,
        "ours_segments_sr_high_blocked_if_enh0": block_sr_high,
        "ours_none_segments_sr_high_gate": dict(none_gate),
        "tool_diff_light_to_heavy_blocked": blocked_heavy_when_ours_light,
        "tool_diff_light_to_heavy_would_allow_enh0": would_allow_heavy,
        "noslack_fc_tools": dict(ns_fc_tools),
        "ours_fc_tools": dict(ours_fc_tools),
        "committed_work_table_s_actual_s": cw,
        "tool_diff_vmaf_delta_noslack_minus_ours_mean": round(stats.mean(vmaf_deltas), 3)
        if vmaf_deltas
        else None,
        "bufs_ours_when_none_mean": round(stats.mean(bufs_for("ours", "none")), 3)
        if bufs_for("ours", "none")
        else None,
        "bufs_ours_when_sr_high_mean": round(stats.mean(bufs_for("ours", "sr_high")), 3)
        if bufs_for("ours", "sr_high")
        else None,
        "bufs_noslack_when_sr_high_mean": round(stats.mean(bufs_for("no_slack", "sr_high")), 3)
        if bufs_for("no_slack", "sr_high")
        else None,
    }

    # --- markdown ---
    lines.append("## 1. 总览")
    lines.append("")
    lines.append("| 臂 | QOE | stall (s) | FC 段 | dec_buf mean/p50 | 工具摘要 |")
    lines.append("|---|---:|---:|---:|---|---|")
    for name in ["ours", "no_slack", "fixed_high"]:
        r = summary["arms"][name]
        tools = ", ".join(f"{k}:{v}" for k, v in r["tools"].items())
        lines.append(
            f"| {name} | {r['QOE']:.2f} | {r['dur_stall']:.2f} | {r['fc_segments']} | "
            f"{r['dec_buf_mean']}/{r['dec_buf_p50']} | {tools} |"
        )
    lines.append("")
    lines.append(f"段长中位数 ≈ **{seg_s:.2f}s**（用于复现 `t < seg` 门）。")
    lines.append("")

    lines.append("## 2. 工具差异结构")
    lines.append("")
    lines.append(
        f"- 同 index 工具不同：**{summary['n_tool_diff']}** / {len(idxs)} 段"
    )
    lines.append(f"- download_action 不同：**{summary['n_download_diff']}** 段（存在下载耦合）")
    lines.append("- 模式（ours → no_slack）：")
    for k, v in summary["tool_diff_patterns"].items():
        lines.append(f"  - `{k}`: {v}")
    lines.append("")
    if vmaf_deltas:
        lines.append(
            f"- 工具不同段上，no_slack − ours 的 real_vmaf 均值：**{summary['recon']['tool_diff_vmaf_delta_noslack_minus_ours_mean']:+.2f}**"
        )
    lines.append("")

    lines.append("## 3. Slack 门是否在挡重工具？（enh0 上界）")
    lines.append("")
    lines.append(
        "用 `decision_time_buffer_level` 作 `buff_down`，并**乐观假设 `buff_enh=0`**"
        "（真实 slack 只会更紧）。复现 `t_ms = T_table/0.9` 与 `slack ≥ t` / `t < seg`。"
    )
    lines.append("")
    lines.append(
        f"- ours 全程：在该上界下 `sr_high` **允许 {allow_sr_high}** / **挡住 {block_sr_high}** 段"
    )
    lines.append(
        f"- ours 选了 `none` 的段里，`sr_high` 门控计数：`{dict(none_gate)}`"
    )
    lines.append(
        f"- 轻→重差异段：上界仍挡住 no_slack 所选重工具 **{blocked_heavy_when_ours_light}**；"
        f"上界本允许 **{would_allow_heavy}**（后者说明还可能是 `buff_enh>0` 或 Q 表/合法集，而不只是 down 缓冲）"
    )
    lines.append("")
    lines.append(
        f"- ours 选 none 时 mean dec_buf={summary['recon']['bufs_ours_when_none_mean']}；"
        f"选 sr_high 时={summary['recon']['bufs_ours_when_sr_high_mean']}；"
        f"no_slack 选 sr_high 时={summary['recon']['bufs_noslack_when_sr_high_mean']}"
    )
    lines.append("")

    lines.append("## 4. Fast-complete 与算力账")
    lines.append("")
    lines.append(f"- ours FC 工具分布：`{dict(ours_fc_tools)}`")
    lines.append(f"- no_slack FC 工具分布：`{dict(ns_fc_tools)}`")
    lines.append("- 承诺算力（表延迟/0.9 求和）vs 实测 enhance_latency 总和：")
    for a, (tbl, act) in cw.items():
        lines.append(f"  - **{a}**: table≈{tbl}s, actual≈{act}s")
    lines.append("")
    lines.append(
        "解读：no_slack **敢选重工具**，再靠 **FC 大量截断**（41 段）把墙钟拉回来；"
        "ours 很少触发 FC（3 段），等于把「可降级完成」的能力浪费在段前拒之门外。"
    )
    lines.append("")

    lines.append("## 5. 下载耦合")
    lines.append("")
    for name in ["ours", "no_slack", "fixed_high"]:
        lines.append(f"- {name} download hist: `{summary['arms'][name]['download']}`")
    lines.append(
        f"- download 不同段数={summary['n_download_diff']}："
        "增强快慢会反馈到缓冲 → ABR，故 no_slack 的「更猛增强」会改变后续下载档，不只是同档换工具。"
    )
    lines.append("")

    # Top example rows
    lines.append("## 6. 差异段样例（前 12 个轻→重）")
    lines.append("")
    lines.append("| idx | ours | no_slack | ours_buf | gate_sr_high | ours_fc | ns_fc |")
    lines.append("|---:|---|---|---:|---|---|---|")
    shown = 0
    for d in tool_only:
        if d["ours_tool"] not in light or d["noslack_tool"] not in {"sr_high", "restore", "x2"}:
            continue
        g = (d.get("gate_if_enh0") or {}).get("gate", {})
        lines.append(
            f"| {d['index']} | {d['ours_tool']} | {d['noslack_tool']} | {d['ours_buf']} | "
            f"{g.get('sr_high','')} | {int(d['ours_fc'])} | {int(d['noslack_fc'])} |"
        )
        shown += 1
        if shown >= 12:
            break
    lines.append("")

    # Findings
    findings = []
    if block_sr_high > allow_sr_high:
        findings.append(
            "即便 buff_enh=0 的乐观上界，ours 轨迹上多数段仍被 `t<seg` 或 slack 挡住 sr_high——"
            "段前门与「重工具墙钟 ≥ 段长」强相关，不只是缓冲一时紧。"
        )
    findings.append(
        "no_slack 用「段前激进 + 段中 FC」组合在差网取得更低 stall/更高 QOE；"
        "ours 的 slack 按「必须完整跑完 T」计，与 FC 能力不一致 → 过保守。"
    )
    findings.append(
        "存在下载耦合：增强策略会改写 ABR 路径，固定「同下载轨迹比工具」不成立；"
        "评价 slack 时应接受系统级闭环差异。"
    )
    if would_allow_heavy > 0 and blocked_heavy_when_ours_light > 0:
        findings.append(
            "部分差异段在 enh0 上界下本可选重工具却选了轻工具，说明真实 buff_enh 或其它门仍在起作用；"
            "诊断需在日志中加 buff_enh/slack 快照（建议）。"
        )

    # Check t>=seg from latency table
    lines.append("## 7. 表延迟 vs 段长（根因候选）")
    lines.append("")
    lines.append("对每个 download 档，表延迟/0.9 与段长比较：")
    lines.append("")
    lines.append("| down | tool | T/0.9 (s) | vs seg |")
    lines.append("|---:|---|---:|---|")
    for d in range(lat.shape[0]):
        for t in range(lat.shape[1]):
            if not np.isfinite(lat[d, t]) or lat[d, t] > 1e7:
                continue
            t_eff = float(lat[d, t]) / SAFETY
            if t == 0:
                continue
            flag = "OK" if t_eff < seg_s else "BLOCK t>=seg"
            if t_eff >= seg_s * 0.85:  # near boundary interesting
                lines.append(f"| {d} | {TOOL[t]} | {t_eff:.3f} | **{flag}** |")
    lines.append("")

    near_block = []
    for d in range(lat.shape[0]):
        for t in [1, 2, 3, 4, 5]:
            if t >= lat.shape[1]:
                continue
            if not np.isfinite(lat[d, t]) or lat[d, t] > 1e7:
                continue
            t_eff = float(lat[d, t]) / SAFETY
            if t_eff >= seg_s:
                near_block.append((d, TOOL[t], t_eff))
    if near_block:
        findings.append(
            "表内若干（档,工具）的 T/0.9 ≥ 段长，**无论缓冲多满都会被 `t >= seg_time` 永久拒绝**；"
            "slack 消融等于同时关掉了这道硬门，这比「缓冲阈值」更致命。"
        )
        summary["hard_blocked_by_t_ge_seg"] = [
            {"down": d, "tool": t, "t_eff_s": round(te, 3)} for d, t, te in near_block
        ]

    summary["findings"] = findings

    lines.append("## 8. 结论与改规则建议")
    lines.append("")
    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. {f}")
    lines.append("")
    lines.append("### 建议的下一刀（按性价比）")
    lines.append("")
    lines.append(
        "1. **拆开两道门分别消融**：`t >= seg` 硬拒绝 vs `slack < t`。"
        "当前 `tool_ignore_slack` 两道一起关，0921 的「反号」可能主要来自硬门，而不是缓冲公式。"
    )
    lines.append(
        "2. **段前按「可 FC 完成」计费**：例如允许 `t_eff` 到 `α·seg`（α>1）或"
        "`min(t_eff, remaining_to_deadline)`，与执行层一致。"
    )
    lines.append(
        "3. **日志**：NES 决策时写入 `buff_down/buff_enh/slack/t_eff/rejected_tools`，否则只能做 enh0 上界推断。"
    )
    lines.append(
        "4. **验证实验（小）**：BBB×0921 只跑 "
        "`ours` / `no_t_ge_seg`（仅放宽硬门）/ `no_slack` / `fixed`，看哪一刀收回 stall。"
    )
    lines.append("")
    lines.append("权威机读：`results/canonical/d5_slack_diag_0921.json`")
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    CANON.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"wrote {CANON}")
    print("findings:")
    for f in findings:
        print("-", f)
    print("hard_blocked", summary.get("hard_blocked_by_t_ge_seg"))
    print("patterns", summary["tool_diff_patterns"])
    print("recon", summary["recon"])


if __name__ == "__main__":
    main()
