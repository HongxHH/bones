# -*- coding: utf-8 -*-
"""
0921 卡顿归因：把 dur_stall 拆成"增强产能不足"与"下载/缓冲"两部分，
判定轻量 e0（最浅档 17.5ms->8ms）的真实收益上限。

同一条 trace、同一 BONES 码率选择，三策略只有增强速度不同：
  force_e3(单帧~50.7ms) / halt(混合,最浅e1~17.5ms) / IMDN(l2~7.7ms)
纯增强库存仿真(sim_e0_falsify.py)里 e1 对 24fps 已有 2.4x 余量、e0 零收益，
故 e0 的价值只可能来自"差网晚到压缩了段增强窗口"——本脚本用真实 json 量化。

逐段量（仅 is_enhance 段）：
  latency   = enhance_latency              该段增强实际墙钟耗时(s)
  lead      = enhance_start_to_play_time   增强开始时刻距该段播放点的提前量(s)
  slack     = lead - latency               增强完成相对播放点的余量;<0=做完已过播放点
  未增强帧   = 96 - enhanced_cnt            fc/截断导致按 base 播的帧(增强产能不足,e0可救)
"""
import json, os, statistics as st

D = r'E:\dev\Code\Python\bones\player\output\data'
FILES = [
    ('force_e3', 'bones_pacnet_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-123213.json'),
    ('halt',     'bones_pacnet_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-121806.json'),
    ('IMDN',     'bones_imdn_always_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-124549.json'),
]
SEG_MS = 4000.0
FRAME_N = 96

print('='*108)
print('0921 卡顿归因（TOS, K=1.5）')
print('='*108)
hdr = f"{'策略':>9} {'总卡顿s':>8} {'次数':>4} {'增强段':>6} {'fc段':>5} {'未增强帧':>8} {'slack<0段':>9} {'负slack和s':>10} {'中位slack':>9}"
print(hdr)
print('-'*108)

for name, fn in FILES:
    d = json.load(open(os.path.join(D, fn), encoding='utf-8'))
    dur = d['dur_stall']; nstall = d['num_stall']
    esegs = [s for s in d['segments'] if s.get('is_enhance')]
    fc = [s for s in esegs if s.get('fast_complete')]
    missing = sum(max(0, FRAME_N - (s.get('enhanced_cnt') or 0)) for s in esegs)
    slacks = []
    for s in esegs:
        lat = s.get('enhance_latency'); lead = s.get('enhance_start_to_play_time')
        if lat is None or lead is None:
            continue
        slacks.append(lead - lat)
    neg = [x for x in slacks if x < 0]
    med = st.median(slacks) if slacks else float('nan')
    print(f"{name:>9} {dur:>8.2f} {nstall:>4} {len(esegs):>6} {len(fc):>5} {missing:>8} "
          f"{len(neg):>9} {sum(-x for x in neg):>10.2f} {med:>9.2f}")

print('-'*108)
print('\n逐段 slack 分布（halt，秒；负数=增强做完已过该段播放点）：')
d = json.load(open(os.path.join(D, FILES[1][1]), encoding='utf-8'))
sl = [(s['index'], s['enhance_start_to_play_time']-s['enhance_latency'], s.get('enhanced_cnt'),
       s.get('fast_complete'), s.get('exit_frame_counts'))
      for s in d['segments'] if s.get('is_enhance') and s.get('enhance_latency') is not None]
sl.sort(key=lambda x: x[1])
print('  最紧张的 15 段:  segIdx  slack(s)  enhCnt  fc  exit[e1,e2,e3]')
for idx, sval, cnt, f, ex in sl[:15]:
    print(f"    seg{idx:>4}  {sval:>7.2f}  {cnt:>4}  {str(f):>5}  {ex}")

# 下载侧：每段下载线程耗时与"播放点时是否已下载完"
print('\n下载侧统计（halt，每段 stop-start，秒）：')
dls = []
for s in d['segments']:
    if s.get('stop_time') and s.get('start_time'):
        dls.append(s['stop_time']-s['start_time'])
ds = sorted(dls)
p90 = ds[int(len(ds)*0.9)]
print(f"  段数={len(dls)} 下载耗时 中位={st.median(dls):.2f}s p90={p90:.2f}s "
      f"max={max(dls):.2f}s 总下载墙钟={sum(dls):.1f}s（视频总长 {184*4}s）")
