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

console.log("metrics.test.js: ok");
