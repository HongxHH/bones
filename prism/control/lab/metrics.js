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

  function formatRebuffer(rebufferS) {
    if (rebufferS == null || rebufferS === "" || Number.isNaN(Number(rebufferS))) {
      return "重缓冲 —";
    }
    return "重缓冲 " + Number(rebufferS).toFixed(1) + " s";
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
  };
});
