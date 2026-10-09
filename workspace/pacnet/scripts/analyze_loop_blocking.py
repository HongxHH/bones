# -*- coding: utf-8 -*-
"""
量化"增强阻塞 asyncio 事件循环 -> 拖慢 trace 分块下载"的量级（0921，K=1.5）。

机制：trace 下载器在事件循环线程里按 20KB 分块 asyncio.sleep 限速；
增强 torch forward + gpuburn 是同步阻塞调用、也在同一线程，每帧仅 sleep(0) 让出。
增强帧阻塞期间下载协程得不到调度 -> 段实际下载墙钟被拉长 -> 卡顿。

用真实 json 的逐段墙钟做经验归因（不重放 asyncio、不写可能再失真的仿真）：
- 增强总墙钟 sum(enhance_latency)
- 每段下载墙钟 stop-start，配对 halt vs IMDN 同字节段，其差即 loop 占用净效应
- 视频总墙钟 = 736 + dur_stall
"""
import json, os, statistics as st

D = r'E:\dev\Code\Python\bones\player\output\data'
FILES = [
    ('force_e3', 'bones_pacnet_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-123213.json'),
    ('halt',     'bones_pacnet_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-121806.json'),
    ('IMDN',     'bones_imdn_always_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-124549.json'),
]
VIDEO_S = 184 * 4

def load(fn):
    d = json.load(open(os.path.join(D, fn), encoding='utf-8'))
    segs = {s['index']: s for s in d['segments']}
    enh = sum(s.get('enhance_latency') or 0 for s in d['segments'])
    nenh = sum(1 for s in d['segments'] if s.get('is_enhance'))
    dl = {}
    for i, s in segs.items():
        if s.get('stop_time') and s.get('start_time'):
            dl[i] = (s['stop_time'] - s['start_time'], s.get('total_bytes') or 0, s.get('repr_id'))
    return d, enh, nenh, dl

print('='*96)
print('0921 事件循环阻塞归因（K=1.5）')
print('='*96)
data = {}
for name, fn in FILES:
    d, enh, nenh, dl = load(fn)
    data[name] = (d, enh, nenh, dl)
    total_dl = sum(x[0] for x in dl.values())
    print(f"{name:>9}: 卡顿 {d['dur_stall']:6.1f}s/{d['num_stall']:>2}次  视频墙钟 {VIDEO_S+d['dur_stall']:6.1f}s  "
          f"增强总墙钟 {enh:6.1f}s({nenh}段)  下载墙钟合计 {total_dl:6.1f}s")

# 配对 halt vs IMDN：同 index、同字节（同码率决策）段，比较下载墙钟
def pair_diff(a, b):
    da = data[a][3]; db = data[b][3]
    same = [(i, da[i][0], db[i][0]) for i in da if i in db and da[i][1] == db[i][1] and da[i][1] > 0]
    diff = [(i, x-y) for i, x, y in same]
    nz = [x for _, x in diff]
    return same, diff, nz

for a, b in [('halt', 'IMDN'), ('force_e3', 'IMDN')]:
    same, diff, nz = pair_diff(a, b)
    tot = sum(nz)
    print(f"\n配对 {a} vs {b}：同码率段 {len(same)} 个，下载墙钟差之和（{a}-{b}）= {tot:+.1f}s "
          f"（中位 {st.median(nz):+.3f}s/段）")

# 码率决策是否被增强改变：repr_id 分布对比
print('\n各策略 repr_id（码率档）段数分布：')
for name, fn in FILES:
    d, _, _, _ = data[name]
    cnt = {}
    for s in d['segments']:
        r = s.get('repr_id')
        cnt[r] = cnt.get(r, 0) + 1
    print(f"  {name:>9}: {dict(sorted(cnt.items(), key=lambda x:(x[0] is None,x[0])))}")

# 增强墙钟差（halt/force_e3 比 IMDN 多占用 loop 的时间上界）
print('\n增强总墙钟差（相对 IMDN，= 多阻塞事件循环的时间上界）：')
base = data['IMDN'][1]
for name in ['force_e3', 'halt']:
    print(f"  {name:>9}-IMDN = {data[name][1]-base:+.1f}s")
