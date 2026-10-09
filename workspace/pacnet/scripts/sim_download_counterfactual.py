# -*- coding: utf-8 -*-
"""
下载侧反事实仿真（0921 等 3G trace）。

只建模"串行下载 + 播放队列消费"，增强在旁路并行（已由 slack 全正、同码率段下载墙钟
仅 +3s 证明增强基本不进卡顿关键路径）。下载限速严格复刻 trace.py 的 throttled_download/
download_time/go_by（含其 go_by 把秒当毫秒的既有行为），分 20000 字节块。

第一步：用三策略真实码率序列复现 dur_stall，校准模型。
第二步：反事实码率策略（固定低码率等）算卡顿，质量用质量表平均上界。
"""
import json, os, glob, math
import numpy as np

PLAYER = r'E:\dev\Code\Python\bones\player'
TRACES = {
    '0921': 'report.2010-09-21_1001CEST.json',
    '0914': 'report.2010-09-14_1038CEST.json',
    '0913': 'report.2010-09-13_1003CEST.json',
}
SEG_MS = 4000
N_SEG = 184
CHUNK = 20_000

# ---- 每档每段真实字节（VBR m4s）----
def load_seg_bytes():
    base = os.path.join(PLAYER, 'data', 'tears_of_steel')
    b = {}
    for r in range(5):
        files = sorted(glob.glob(os.path.join(base, f'stream{r}', 'segment_*.m4s')),
                       key=lambda p: int(p.split('segment_')[1].split('.m4s')[0]))
        b[r] = [os.path.getsize(f) for f in files]
    return b

# ---- trace 下载时钟（严格复刻 trace.py）----
class TraceClock:
    def __init__(self, trace_fn):
        tr = json.load(open(os.path.join(PLAYER, 'data', '3Glogs', trace_fn)))
        self.bw = np.array([x['bandwidth_kbps'] for x in tr], dtype=float)  # kbps=bits/ms
        self.du = np.array([x['duration_ms'] for x in tr], dtype=float)
        self.idx = 0
        self.rest = self.du[0]
        self.wall = 0.0
        self.last = None

    def _go_by(self, time_pass):  # 原样复刻：time_pass 数值直接与 ms 的 rest 比较
        while self.rest < time_pass:
            time_pass -= self.rest
            self.idx = (self.idx + 1) % len(self.bw)
            self.rest = self.du[self.idx]
        self.rest -= time_pass

    def _transfer_ms(self, size):  # download_time 的传输计算，返回 ms 并推进 rest/idx
        remain = size * 8.0
        bw = self.bw[self.idx]
        total = 0.0
        while remain > bw * self.rest:
            remain -= bw * self.rest
            self.idx = (self.idx + 1) % len(self.bw)
            self.rest = self.du[self.idx]
            bw = self.bw[self.idx]
            total += self.rest
        self.rest -= remain / bw
        total += remain / bw
        return total

    def download_segment(self, nbytes):
        """串行下载一段，返回该段完成后的累计墙钟（秒）。"""
        n_chunks = math.ceil(nbytes / CHUNK)
        for c in range(n_chunks):
            size = min(CHUNK, nbytes - c * CHUNK)
            # download_time 开头
            if self.last is None:
                self.last = self.wall
            else:
                elapsed = self.wall - self.last
                self.last = self.wall
                self._go_by(elapsed)
            total_s = self._transfer_ms(size) / 1000.0
            self.wall += total_s  # asyncio.sleep
        return self.wall

def ready_timeline(repr_seq, seg_bytes, trace_fn):
    clk = TraceClock(trace_fn)
    ready = []
    for i, r in enumerate(repr_seq):
        clk.download_segment(seg_bytes[r][i])
        ready.append(clk.wall)
    return ready

def total_stall(ready, seg_s=4.0):
    """零起播缓冲队列模型：s_i=max(ready_i, s_{i-1}+seg)，stall 累加。"""
    stall = 0.0
    s_prev = ready[0]  # 第一段下完开播
    stalls = []
    for i in range(1, len(ready)):
        due = s_prev + seg_s
        if ready[i] > due:
            d = ready[i] - due
            stall += d
            stalls.append(d)
            s_prev = ready[i]
        else:
            s_prev = due
    return stall, stalls

def real_repr_seq(fn):
    d = json.load(open(os.path.join(PLAYER, 'output', 'data', fn), encoding='utf-8'))
    segs = sorted(d['segments'], key=lambda s: s['index'])
    seq = [(s.get('repr_id') if s.get('repr_id') is not None else s.get('download_action')) for s in segs]
    return d, seq

FILES = {
    '0921': {
        'halt': 'bones_pacnet_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-121806.json',
        'force_e3': 'bones_pacnet_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-123213.json',
        'IMDN': 'bones_imdn_always_fast-True_video-tos_trace-report.2010-09-21_1001CEST_20260918-124549.json',
    },
}

if __name__ == '__main__':
    seg_bytes = load_seg_bytes()
    print('每档段均字节 KB:', {r: round(np.mean(seg_bytes[r])/1024) for r in range(5)})
    for tag, strategies in FILES.items():
        print('='*90)
        print(f'校准 trace {tag}')
        print('='*90)
        for name, fn in strategies.items():
            d, seq = real_repr_seq(fn)
            seq = [0 if x is None else x for x in seq]
            ready = ready_timeline(seq, seg_bytes, TRACES[tag])
            stall, _ = total_stall(ready)
            print(f'{name:>9}: 真实卡顿 {d["dur_stall"]:6.1f}s | 仿真重放卡顿 {stall:6.1f}s | '
                  f'下载总墙钟 {ready[-1]:.1f}s（视频 {N_SEG*4}s）')

        # 反事实：固定码率档
        print('-'*90)
        for r in range(3):
            seq = [r]*N_SEG
            ready = ready_timeline(seq, seg_bytes, TRACES[tag])
            stall, _ = total_stall(ready)
            print(f'固定 repr{r} 全程: 仿真卡顿 {stall:6.1f}s | 下载总墙钟 {ready[-1]:.1f}s')
