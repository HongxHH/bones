"use strict";

const assert = require("assert");
const M = require("./metrics.js");

assert.strictEqual(M.fmtMem(212), "212M");
assert.strictEqual(M.fmtMem(1536), "1.5G");
assert.strictEqual(M.fmtMem(null), "—");

assert.strictEqual(M.fmtInvNum(1.234, 2), "1.23");
assert.strictEqual(M.fmtInvNum(null, 2), "—");

assert.strictEqual(M.statusLabel("enhancing"), "增强中");
assert.strictEqual(M.statusLabel("queued"), "排队");
assert.strictEqual(M.statusLabel("ready"), "就绪");

assert.strictEqual(M.pendingEnhanceS({ enhance_pending_s: 3 }), 3);
assert.strictEqual(M.pendingEnhanceS({ enhance_queue_s: 2 }), 2);
assert.strictEqual(M.pendingEnhanceS({ enhance_buffer_s: 1 }), 1);
assert.strictEqual(M.pendingEnhanceS({}), 0);

assert.strictEqual(M.formatSeconds(4.56), "4.6 s");
assert.strictEqual(M.formatSeconds(null), "—");
assert.strictEqual(M.formatBwMbps(2.5e6), "2.5 Mbps");
assert.strictEqual(M.formatBwMbps(null), "—");

assert.strictEqual(
  M.formatCpuLine({ cpu_pct: 10.4, mem_pct: 14.2, rss_mb: 6845.2 }),
  "10% · 14% · 6845M",
);
assert.strictEqual(M.formatCpuLine({}), "—");
assert.strictEqual(M.formatCpuLine(null), "—");
assert.ok(M.formatCpuTitle().includes("进程算力"));

assert.strictEqual(
  M.formatGpuLine({
    gpu_util_pct: 5.2,
    gpu_mem_used_mb: 3048,
    gpu_proc_mem_mb: 212,
    gpu_mem_pct: 18.7,
  }),
  "5% · 3.0G · 进程212M",
);
assert.strictEqual(
  M.formatGpuLine({ gpu_util_pct: 5, gpu_mem_used_mb: 500, gpu_proc_mem_mb: 0.5 }),
  "5% · 500M",
);

assert.ok(M.formatGpuTitle({ gpu_proc_mem_src: "torch", gpu_mem_pct: 19 }).includes("torch"));
assert.ok(M.formatGpuTitle({ gpu_proc_mem_src: "nvml" }).includes("nvml"));
assert.ok(M.formatGpuTitle({}).includes("整卡已用"));

assert.strictEqual(
  M.formatSysExtra({ cpu_system_pct: 14, gpu_mem_pct: 19, gpu_proc_mem_src: "torch" }),
  "整机CPU 14% · 整卡显存 19% · 进程显存 torch",
);
assert.strictEqual(M.formatSysExtra({}), "整机CPU —");

assert.strictEqual(M.formatRebuffer(1.25), "重缓冲 1.3 s");
assert.strictEqual(M.formatRebuffer(null), "重缓冲 —");
assert.strictEqual(M.formatCard({ label: "QoE", digits: 2 }, 74.5), "QoE 74.50");
assert.strictEqual(M.formatCard({ label: "触发段数", digits: 0 }, 3), "触发段数 3");
assert.strictEqual(M.formatCard({ label: "均卡顿", digits: 2, suffix: " s" }, 0.2), "均卡顿 0.20 s");
assert.strictEqual(M.formatCard({ label: "质量下降", digits: 2 }, null), "质量下降 —");
assert.deepStrictEqual(M.formatCardParts({ label: "均卡顿", digits: 2, suffix: " s" }, 0.2), {
  label: "均卡顿",
  value: "0.20",
  suffix: " s",
  empty: false,
});
assert.strictEqual(M.formatCardParts({ label: "质量下降", digits: 2 }, null).empty, true);
assert.deepStrictEqual(
  M.groupCards([
    { key: "qoe", label: "QoE" },
    { key: "fc_segments", label: "触发段数" },
    { key: "avg_vmaf", label: "VMAF", group: "播放质量" },
  ]).map((group) => [group.name, group.cards.map((card) => card.key)]),
  [
    ["播放质量", ["qoe", "avg_vmaf"]],
    ["fast-complete", ["fc_segments"]],
  ],
);
assert.strictEqual(M.formatQoe(74.5), "QoE 74.50");
assert.strictEqual(M.formatQoe(null), "QoE —");
assert.strictEqual(M.formatVmaf(80), "VMAF 80.00");
assert.strictEqual(M.formatOsc(1.5), "振荡 1.50");
assert.strictEqual(M.formatFreeze(0.25), "均卡顿 0.25 s");
assert.strictEqual(M.formatFreeze(null), "均卡顿 —");

assert.strictEqual(
  M.formatAbortSummary({
    "Already played in enhance": 6,
    "Already played before enhance": 6,
  }),
  "增强中已开始播放:6；播放前已过:6",
);
assert.strictEqual(M.formatAbortSummary(null), null);

const enriched = M.enrichArmFromAnalyzer(
  { qoe: 57.3 },
  {
    available: true,
    summary: {
      enhancer_abort_total: 12,
      abort_wasted_frames_total: 363,
      num_stall: 9,
      num_quality_switches: 40,
      avg_bitrate: 681993,
    },
    enhance_levels: [
      { level: "2", count: 141, completed: 129, completion_rate: 0.9149, avg_latency: 1.03 },
      { level: "1", count: 10, completed: 9, completion_rate: 0.9 },
    ],
    abort_reasons: { "Already played in enhance": 6 },
  },
);
assert.strictEqual(
  M.enrichArmFromAnalyzer(
    { qoe: 99 },
    { available: true, summary: { qoe: 57.3, avg_vmaf: 64.1 }, enhance_levels: [], abort_reasons: {} },
  ).qoe,
  57.3,
);
assert.strictEqual(enriched.qoe, 57.3);
assert.strictEqual(enriched.abort_total, 12);
assert.strictEqual(enriched.abort_wasted_frames, 363);
assert.strictEqual(enriched.enh_l2_count, 141);
assert.strictEqual(enriched.enh_l2_completed, 129);
assert.strictEqual(enriched.enh_l2_rate, 0.9149);
assert.strictEqual(enriched.abort_summary, "增强中已开始播放:6");
assert.strictEqual(enriched.metrics_status, "partial");
assert.strictEqual(enriched.quality_source, "analyzer");

const sampleSpec = {
  apiVersion: "prism.lab/v1",
  kind: "ExperimentSpec",
  metadata: { name: "bbb-play" },
  spec: {
    workspace: "baseline",
    content: { source: "path:player/data/video_mp4box" },
    arms: [
      {
        id: "play",
        player: {
          mod_nes: "always_greedy",
          mod_enhancer: "imdn_always",
          mod_renderer: "opencv",
          mod_downloader: "trace",
          trace: "data/3Glogs/report.2010-09-13_1003CEST.json",
          use_fast_complete: false,
          collect_system_metrics: false,
        },
      },
    ],
    collect: { client_metrics: true, system_metrics: false },
  },
};
const summary = M.recipeSummaryFromSpec(sampleSpec, { partMax: 40 });
assert.ok(summary.line.includes("baseline"));
assert.ok(summary.line.includes("always_greedy"));
assert.ok(summary.line.includes("opencv"));
assert.ok(summary.line.includes("report.2010-09-13_1003CEST.json"));
assert.ok(summary.title.includes("轨迹 data/3Glogs/"));
assert.ok(summary.title.includes("系统指标 关"));
assert.strictEqual(summary.parts.net, "report.2010-09-13_1003CEST.json");
assert.strictEqual(summary.shortName, "baseline · always_greedy · report.2010-09-13_1003CEST.json");
assert.strictEqual(M.recipePartsFromSpec(sampleSpec.spec).workspace, "baseline");
assert.strictEqual(M.traceFileLabel("data/3Glogs/a.json"), "a.json");
assert.strictEqual(M.armPlayerFromSpec(sampleSpec).mod_nes, "always_greedy");
const clipped = M.recipeSummaryFromSpec(sampleSpec, { partMax: 12 });
assert.ok(clipped.line.includes("…"));

const tcpSummary = M.recipeSummaryFromSpec({
  spec: {
    workspace: "fast_complete",
    arms: [{ id: "a", player: { mod_nes: "bones", mod_downloader: "tcp", use_fast_complete: true } }],
    collect: { system_metrics: true },
  },
});
assert.ok(tcpSummary.line.includes("TCP"));
assert.ok(tcpSummary.title.includes("fast-complete 开"));
assert.ok(tcpSummary.title.includes("系统指标 开"));

const idleChrome = M.shellChrome("idle");
assert.strictEqual(idleChrome.recipeTitle, "配方");
assert.strictEqual(idleChrome.expandRecipe, true);
assert.strictEqual(idleChrome.collapseRecipe, false);
assert.strictEqual(idleChrome.showMonitor, false);
assert.strictEqual(idleChrome.showDupMeters, false);
assert.strictEqual(idleChrome.showSparkCharts, false);

const runningChrome = M.shellChrome("running");
assert.strictEqual(runningChrome.recipeTitle, "下次配方");
assert.strictEqual(runningChrome.collapseRecipe, true);
assert.strictEqual(runningChrome.collapseHistory, true);
assert.strictEqual(runningChrome.showMonitor, true);
assert.strictEqual(runningChrome.showSparkCharts, true);
assert.strictEqual(runningChrome.showDupMeters, true);

const preparingChrome = M.shellChrome("preparing");
assert.strictEqual(preparingChrome.showSparkCharts, true);
assert.strictEqual(preparingChrome.collapseHistory, true);

const stoppedChrome = M.shellChrome("stopped");
assert.strictEqual(stoppedChrome.recipeTitle, "下次配方");
assert.strictEqual(stoppedChrome.collapseRecipe, true);
assert.strictEqual(stoppedChrome.expandRecipe, false);
assert.strictEqual(stoppedChrome.showDupMeters, true);
assert.strictEqual(stoppedChrome.showSparkCharts, true);
assert.strictEqual(stoppedChrome.collapseHistory, false);
assert.strictEqual(stoppedChrome.showMonitor, true);

const failedChrome = M.shellChrome("failed");
assert.strictEqual(failedChrome.showSparkCharts, true);
assert.strictEqual(failedChrome.collapseRecipe, true);

console.log("metrics.test.js: ok");
