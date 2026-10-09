# -*- coding: utf-8 -*-
"""追加阶段二两条仿真器/播放器差异到 known_issues.md（幂等，保持 CRLF）。"""
import io
p = r"E:\dev\Code\Python\bones\docs\known_issues.md"
raw = io.open(p, "r", encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in raw else "\n"
s = raw.replace("\r\n", "\n")

anchor = "- 是验证内容感知方法的正确场所\n"
block = (
"- 是验证内容感知方法的正确场所\n"
"\n"
"### halt 离线仿真库存判据不能照搬到播放器（阶段二确认）\n"
"- 离线 `workspace/pacnet/sim_tos_halt.py` 只有“增强 vs 播放”一个库存递推 `B+=帧间隔−延迟`，没有 fast_complete，库存跌破单帧门槛才降档，贪心库存策略成立。\n"
"- 真实播放器另有独立的 fast_complete 保护层，其动态阈值（约 0.8s 余量）远高于 halt 单帧库存门槛（约 0.06s）。halt 若用纯库存水位判据，lead 下降时 fast_complete 先触发、整段截断，e2/e1 档来不及执行（实测死磕 e3 到只做 60/96 帧）。\n"
"- 播放器里 halt 应改用可调度性前瞻：逐级判断 `lead >= l_exit×剩余帧×安全系数`，选最高可行档，把 fast_complete 留作“连 e1 都不可行”的最终兜底。\n"
"- 离线仿真还省略解码、resize、H2D/D2H、asyncio 调度与下载争抢等每帧固定开销，临界算力下实测帧延迟高于静态延迟表，sim 的 e3 占比偏乐观，只能做趋势/阈值初筛，最终以播放器为准。\n"
"\n"
"### 制造算力紧张必须用乘法负载，不能用固定加法（阶段二确认）\n"
"- `_gpuburn.py` 的 `PACNET_BURN_K` 按每帧实测前向 `fwd_ms×(K−1)` 烧 matmul，端到端约 ×K，保持三出口延迟比，halt 阈值不倒挂，PAC/IMDN 同口径公平。\n"
"- 旧 `PACNET_EXTRA_MS` 对每帧固定加毫秒，会压缩出口延迟比（如 1.397→1.177），使 e3 阈值反低于 e2、e2 档永远选不到（e2 死档）。仅保留作兼容。\n"
"\n"
"### PACNet 帧级 VMAF 埋点（阶段二修复，易回退）\n"
"- analyzer 不能用 BONES 下发动作 `segment.enhance_action`（PACNet 模式恒为 3）查表打分，否则 halt 降 e1/e2 的帧、fast_complete 未增强帧全按 e3 计分，VMAF 虚高约 4 分/段。\n"
"- 正确做法：PACNet stats 回传 `exit_frame_counts=[e1,e2,e3]` 与 `fast_complete_triggered`，analyzer 按帧占比加权 real_vmaf，未增强帧按 base 计。改动见 `pacnet_enhancer.py`、`analyzer.py` 与补丁 `patch_vmaf_accounting.py`。\n"
)

if "halt 离线仿真库存判据不能照搬" in s:
    print("[skip] already inserted")
elif anchor not in s:
    raise SystemExit("[error] anchor not found")
else:
    s = s.replace(anchor, block, 1)
    io.open(p, "w", encoding="utf-8", newline="").write(s.replace("\n", nl))
    print("[ok] known_issues.md updated, newline=%r" % nl)
