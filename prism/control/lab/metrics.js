/**
 * Prism Lab metric formatters — pure helpers shared by Lab UI.
 * Browser: globalThis.PrismLabMetrics
 * Node: module.exports (for unit tests)
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root) {
    root.PrismLabMetrics = api;
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function fmtInvNum(v, digits) {
    if (v == null || v === "" || Number.isNaN(Number(v))) return "—";
    return Number(v).toFixed(digits);
  }

  function fmtMem(mb) {
    if (mb == null || mb === "") return "—";
    const n = Number(mb);
    if (Number.isNaN(n)) return "—";
    if (n >= 1024) return (n / 1024).toFixed(1) + "G";
    return Math.round(n) + "M";
  }

  function statusLabel(status) {
    if (status === "enhancing") return "增强中";
    if (status === "queued") return "排队";
    if (status === "ready") return "就绪";
    return status || "—";
  }

  function pendingEnhanceS(sample) {
    if (!sample) return null;
    if (sample.enhance_pending_s != null) return Number(sample.enhance_pending_s);
    if (sample.enhance_queue_s != null) return Number(sample.enhance_queue_s);
    if (sample.enhance_buffer_s != null) return Number(sample.enhance_buffer_s);
    return 0;
  }

  function formatSeconds(v, digits) {
    // Number(null) === 0 — must reject null/undefined before coercing.
    if (v == null || v === "" || Number.isNaN(Number(v))) return "—";
    return Number(v).toFixed(digits == null ? 1 : digits) + " s";
  }

  function formatBwMbps(bps) {
    if (bps == null || bps === "" || Number.isNaN(Number(bps))) return "—";
    return (Number(bps) / 1e6).toFixed(1) + " Mbps";
  }

  /** CPU line: "10% · 14% · 6845M"  (cpu · mem% · rss) */
  function formatCpuLine(sample) {
    if (!sample) return "—";
    const cpu = sample.cpu_pct != null ? Number(sample.cpu_pct) : null;
    const memPct = sample.mem_pct != null ? Number(sample.mem_pct) : null;
    const rss = sample.rss_mb != null ? Number(sample.rss_mb) : null;
    const parts = [];
    if (cpu != null && !Number.isNaN(cpu)) parts.push(cpu.toFixed(0) + "%");
    if (memPct != null && !Number.isNaN(memPct)) parts.push(memPct.toFixed(0) + "%");
    if (rss != null && !Number.isNaN(rss)) parts.push(Math.round(rss) + "M");
    return parts.length ? parts.join(" · ") : "—";
  }

  function formatCpuTitle() {
    return "CPU：进程算力% · 进程内存占整机% · RSS";
  }

  /**
   * GPU line: "8% · 3.0G · 进程212M"
   * (util · device used ≈ Task Manager · this-process estimate)
   */
  function formatGpuLine(sample) {
    if (!sample) return "—";
    const gpu = sample.gpu_util_pct != null ? Number(sample.gpu_util_pct) : null;
    const deviceUsed = sample.gpu_mem_used_mb != null ? Number(sample.gpu_mem_used_mb) : null;
    const gpuProcMb = sample.gpu_proc_mem_mb != null ? Number(sample.gpu_proc_mem_mb) : null;
    const parts = [];
    if (gpu != null && !Number.isNaN(gpu)) parts.push(gpu.toFixed(0) + "%");
    if (deviceUsed != null && !Number.isNaN(deviceUsed)) parts.push(fmtMem(deviceUsed));
    if (gpuProcMb != null && !Number.isNaN(gpuProcMb) && gpuProcMb > 1) {
      parts.push("进程" + fmtMem(gpuProcMb));
    }
    return parts.length ? parts.join(" · ") : "—";
  }

  function formatGpuTitle(sample) {
    const src = sample && sample.gpu_proc_mem_src ? String(sample.gpu_proc_mem_src) : "";
    const pct = sample && sample.gpu_mem_pct != null ? Number(sample.gpu_mem_pct) : null;
    const pctTxt = pct != null && !Number.isNaN(pct) ? pct.toFixed(0) + "%" : "—";
    let title = "GPU：算力% · 整卡已用（≈任务管理器） · 本进程显存";
    title += " · 整卡占用 " + pctTxt;
    if (src === "torch") {
      title += " · 进程口径 torch reserved（Windows 上 NVML 按 PID 常为 N/A，模型加载后往往持平）";
    } else if (src) {
      title += " · 进程口径 " + src;
    }
    return title;
  }

  function formatSysExtra(sample) {
    const cpuSys = sample && sample.cpu_system_pct != null ? Number(sample.cpu_system_pct) : null;
    const gpuPct = sample && sample.gpu_mem_pct != null ? Number(sample.gpu_mem_pct) : null;
    const src = sample && sample.gpu_proc_mem_src ? String(sample.gpu_proc_mem_src) : "";
    const parts = [
      "整机CPU " + (cpuSys == null || Number.isNaN(cpuSys) ? "—" : cpuSys.toFixed(0) + "%"),
    ];
    if (gpuPct != null && !Number.isNaN(gpuPct)) parts.push("整卡显存 " + gpuPct.toFixed(0) + "%");
    if (src) parts.push("进程显存 " + src);
    return parts.join(" · ");
  }

  function formatCardParts(card, value) {
    const label = (card && card.label) || "";
    const digits = card && card.digits != null ? Number(card.digits) : 2;
    const empty = value == null || value === "" || Number.isNaN(Number(value));
    return {
      label: label,
      value: empty ? "—" : Number(value).toFixed(digits),
      suffix: empty || !(card && card.suffix) ? "" : String(card.suffix),
      empty: empty,
    };
  }

  function formatCard(card, value) {
    const parts = formatCardParts(card, value);
    if (parts.empty) return parts.label + " —";
    return parts.label + " " + parts.value + parts.suffix;
  }

  function cardGroupName(card) {
    if (card && card.group) return String(card.group);
    const key = String((card && card.key) || "");
    if (key.indexOf("fc_") === 0) return "fast-complete";
    return "播放质量";
  }

  function groupCards(cards) {
    const order = [];
    const buckets = {};
    (cards || []).forEach(function (card) {
      const name = cardGroupName(card);
      if (!buckets[name]) {
        buckets[name] = [];
        order.push(name);
      }
      buckets[name].push(card);
    });
    return order.map(function (name) {
      return { name: name, cards: buckets[name] };
    });
  }

  function formatRebuffer(rebufferS) {
    if (rebufferS == null || rebufferS === "" || Number.isNaN(Number(rebufferS))) {
      return "重缓冲 —";
    }
    return "重缓冲 " + Number(rebufferS).toFixed(1) + " s";
  }

  function formatFixed(label, value, digits) {
    if (value == null || value === "" || Number.isNaN(Number(value))) return label + " —";
    return label + " " + Number(value).toFixed(digits);
  }

  function formatQoe(value) {
    return formatFixed("QoE", value, 2);
  }

  function formatVmaf(value) {
    return formatFixed("VMAF", value, 2);
  }

  function formatOsc(value) {
    return formatFixed("振荡", value, 2);
  }

  function formatFreeze(value) {
    return formatFixed("均卡顿", value, 2) + (value == null || value === "" || Number.isNaN(Number(value)) ? "" : " s");
  }

  /**
   * Normalize ExperimentSpec document or its inner `spec` object.
   * Full: { apiVersion, kind, metadata, spec: { arms, ... } }
   * Inner: { arms, workspace, content, collect, ... }
   */
  function recipeBodyFromSpec(spec) {
    const root = spec || {};
    if (root.spec && typeof root.spec === "object" && Array.isArray(root.spec.arms)) {
      return root.spec;
    }
    if (Array.isArray(root.arms)) return root;
    return root.spec && typeof root.spec === "object" ? root.spec : root;
  }

  /** First arm's player block from an ExperimentSpec (or its .spec inner). */
  function armPlayerFromSpec(spec) {
    const body = recipeBodyFromSpec(spec);
    const arm = ((body && body.arms) || [])[0] || {};
    return (arm && arm.player) || {};
  }

  function traceFileLabel(path) {
    const s = String(path || "").replace(/\\/g, "/");
    if (!s) return "";
    const parts = s.split("/");
    return parts[parts.length - 1] || s;
  }

  function clipText(text, max) {
    const t = String(text || "").trim();
    if (!max || t.length <= max) return t;
    return t.slice(0, Math.max(1, max - 1)) + "…";
  }

  /**
   * Human-readable recipe summary for history / chips / run names.
   * Accepts full ExperimentSpec `{ spec: {...} }` or the inner `spec` object.
   */
  function recipePartsFromSpec(spec) {
    const body = recipeBodyFromSpec(spec);
    const player = armPlayerFromSpec(spec);
    const workspace = String(body.workspace || "").trim();
    const content = String((body.content && body.content.source) || "").trim();
    const nes = String(player.mod_nes || "").trim();
    const rawEnhancer = String(player.mod_enhancer || "").trim();
    const enhancer = rawEnhancer === "imdn_always" ? "imdn" : rawEnhancer;
    const renderer = String(player.mod_renderer || "").trim();
    const trace = String(player.trace || "").trim();
    const downloader = String(player.mod_downloader || "").trim();
    const net =
      trace || downloader === "trace"
        ? (traceFileLabel(trace) || "轨迹")
        : "TCP";
    const collect = body.collect || {};
    const systemMetrics = !!(collect.system_metrics || player.collect_system_metrics);
    return {
      workspace: workspace,
      content: content,
      nes: nes,
      enhancer: enhancer,
      renderer: renderer,
      net: net,
      trace: trace,
      useFast: !!player.use_fast_complete,
      latencyAdapt: !!player.latency_adapt,
      frameCatalog: player.quality_source === "frame_catalog",
      systemMetrics: systemMetrics,
    };
  }

  function recipeSummaryFromSpec(spec, opts) {
    const parts = recipePartsFromSpec(spec);
    const partMax = opts && opts.partMax != null ? Number(opts.partMax) : 18;
    const bits = [parts.workspace, parts.nes, parts.enhancer, parts.renderer, parts.net]
      .filter(Boolean)
      .map(function (bit) {
        return clipText(bit, partMax);
      });
    const line = bits.join(" · ") || "—";
    const titleBits = [
      parts.workspace ? "工作区 " + parts.workspace : "",
      parts.content ? "内容 " + parts.content : "",
      parts.nes ? "NES " + parts.nes : "",
      parts.enhancer ? "增强 " + parts.enhancer : "",
      parts.renderer ? "渲染 " + parts.renderer : "",
      parts.trace ? "轨迹 " + parts.trace : "网络 TCP",
      parts.useFast ? "fast-complete 开" : "",
      parts.frameCatalog ? "逐段测量" : "",
      parts.latencyAdapt ? "latency_adapt 开" : "",
      parts.systemMetrics ? "系统指标 开" : "系统指标 关",
    ].filter(Boolean);
    const shortName = [parts.workspace || "run", parts.nes || "", parts.net || ""]
      .filter(Boolean)
      .join(" · ");
    return {
      line: line,
      title: titleBits.join("\n"),
      shortName: shortName || "run",
      parts: parts,
    };
  }

  const ABORT_LABELS = {
    "Remaining time less than min threshold": "剩余时间不足",
    "Already played before enhance": "播放前已过",
    "Already played in enhance": "增强中已开始播放",
    "Max resolution": "已是最高分辨率",
    "No enhancement": "不增强",
    "UnknownReason": "原因未记录",
  };

  function formatAbortSummary(reasons) {
    if (!reasons || typeof reasons !== "object") return null;
    const parts = Object.entries(reasons)
      .filter(([, value]) => value != null && value !== "")
      .map(([key, value]) => (ABORT_LABELS[key] || key) + ":" + value);
    return parts.length ? parts.join("；") : null;
  }

  /** Merge /analyzer report fields onto a flat arm metrics object for compare. */
  function enrichArmFromAnalyzer(arm, report) {
    const out = Object.assign({}, arm || {});
    if (!report || report.available === false) return out;
    const summary = report.summary || {};
    // Prefer player-analyzer QoE/VMAF when present so compare matches the
    // playback report (and both columns share the same estimate label).
    if (summary.qoe != null) out.qoe = summary.qoe;
    if (summary.avg_vmaf != null) out.avg_vmaf = summary.avg_vmaf;
    if (summary.oscillation_vmaf != null) out.oscillation_vmaf = summary.oscillation_vmaf;
    if (summary.avg_freeze != null) out.avg_freeze = summary.avg_freeze;
    const map = {
      enhancer_abort_total: "abort_total",
      abort_wasted_frames_total: "abort_wasted_frames",
      num_stall: "num_stall",
      num_quality_switches: "num_quality_switches",
      avg_bitrate: "avg_bitrate",
      dur_stall: "stall_s",
      rebuffer_s: "rebuffer_s",
      fast_complete_segments_count: "fc_segments",
      fast_complete_saved_frames_total: "fc_saved_frames",
      fast_complete_quality_drop_avg: "fc_quality_drop_avg",
      segments: "segments",
    };
    Object.keys(map).forEach((src) => {
      if (summary[src] != null) out[map[src]] = summary[src];
    });
    if (summary.dur_stall != null && out.rebuffer_s == null) out.rebuffer_s = summary.dur_stall;
    (report.enhance_levels || []).forEach((row) => {
      if (!row || row.level == null) return;
      const lv = String(row.level);
      if (row.count != null) out["enh_l" + lv + "_count"] = row.count;
      if (row.completed != null) out["enh_l" + lv + "_completed"] = row.completed;
      if (row.completion_rate != null) out["enh_l" + lv + "_rate"] = row.completion_rate;
      if (row.avg_latency != null) out["enh_l" + lv + "_latency"] = row.avg_latency;
    });
    if (report.abort_reasons && typeof report.abort_reasons === "object") {
      out.abort_reasons = report.abort_reasons;
    }
    const abortSummary = formatAbortSummary(out.abort_reasons);
    if (abortSummary) out.abort_summary = abortSummary;
    if (report.file || summary.qoe != null || (report.enhance_levels || []).length) {
      out.quality_source = "analyzer";
      out.outcome_source = "analyzer";
      out.metrics_status = "partial";
      out.metrics_reason =
        "画质来自播放器质量表推算（与 analyzer 一致；abort/FC 按播出结局），不是实测 libvmaf";
    }
    return out;
  }

  /** Shell left column: what to show for a run status. Browser Lab ignores this. */
  function shellChrome(state) {
    const s = state || "idle";
    const live = s === "running" || s === "preparing";
    const settled = s === "stopped" || s === "failed";
    return {
      recipeTitle: s === "idle" ? "配方" : "下次配方",
      expandRecipe: s === "idle",
      collapseRecipe: live || settled,
      collapseHistory: live,
      showMonitor: live || settled,
      showDupMeters: live || settled,
      showSparkCharts: live || settled,
    };
  }

  return {
    fmtInvNum,
    fmtMem,
    statusLabel,
    pendingEnhanceS,
    formatSeconds,
    formatBwMbps,
    formatCpuLine,
    formatCpuTitle,
    formatGpuLine,
    formatGpuTitle,
    formatSysExtra,
    formatRebuffer,
    formatCard,
    formatCardParts,
    cardGroupName,
    groupCards,
    formatQoe,
    formatVmaf,
    formatOsc,
    formatFreeze,
    recipeBodyFromSpec,
    armPlayerFromSpec,
    traceFileLabel,
    clipText,
    recipePartsFromSpec,
    recipeSummaryFromSpec,
    enrichArmFromAnalyzer,
    formatAbortSummary,
    ABORT_LABELS,
    shellChrome,
  };
});
