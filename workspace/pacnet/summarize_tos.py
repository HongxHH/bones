# -*- coding: utf-8 -*-
"""汇总 TOS 跨视频播放器实验：PACNet vs IMDN 基线，3 条 3G trace。"""
import json, glob, os
from collections import Counter

OUT = r'E:\dev\Code\Python\bones\player\output\data'
traces = {
    '09-13': 'report.2010-09-13_1003CEST',
    '09-14': 'report.2010-09-14_1038CEST',
    '09-21': 'report.2010-09-21_1001CEST',
}
methods = {'pacnet': 'bones_pacnet', 'imdn': 'bones_imdn'}

def latest(method_pat, trace):
    fs = glob.glob(os.path.join(OUT, f'{method_pat}*video-tos*trace-{trace}_*.json'))
    fs = [f for f in fs if 'latency' not in f]
    if not fs:
        return None
    return max(fs, key=os.path.getmtime)

rows = {}
for tk, tv in traces.items():
    rows[tk] = {}
    for mk, mp in methods.items():
        f = latest(mp, tv)
        if not f:
            rows[tk][mk] = None
            continue
        d = json.load(open(f, encoding='utf-8'))
        segs = d['segments']
        dl = Counter(s['download_action'] for s in segs)
        en = Counter(s['enhance_action'] for s in segs)
        rows[tk][mk] = dict(
            qoe=d['QOE'], vmaf=d['avg_vmaf'], osc=d['oscillation_vmaf'],
            nstall=d['num_stall'], dstall=d['dur_stall'], sw=d['num_quality_switches'],
            dl=dict(sorted(dl.items())), en=dict(sorted(en.items())),
            fc=sum(1 for s in segs if s.get('fast_complete')),
            n=len(segs),
        )

def fmt(x, n=2):
    return f'{x:.{n}f}' if isinstance(x, (int, float)) else str(x)

print('='*100)
print('TOS（Tears of Steel，24fps，184段）PACNet vs IMDN基线，3条3G trace')
print('='*100)
hdr = f"{'trace':<8}{'方法':<8}{'QoE':>8}{'VMAF':>8}{'osc':>7}{'卡顿次':>7}{'卡顿s':>8}{'切换':>6}{'fast':>6}"
print(hdr)
agg = {'pacnet': [], 'imdn': []}
for tk in traces:
    for mk in ['pacnet', 'imdn']:
        r = rows[tk][mk]
        if r is None:
            print(f'{tk:<8}{mk:<8}  缺失')
            continue
        agg[mk].append(r)
        print(f'{tk:<8}{mk:<8}{r["qoe"]:>8.2f}{r["vmaf"]:>8.2f}{r["osc"]:>7.2f}'
              f'{r["nstall"]:>7}{r["dstall"]:>8.2f}{r["sw"]:>6}{r["fc"]:>6}')
    # 差值
    p, i = rows[tk]['pacnet'], rows[tk]['imdn']
    if p and i:
        win = 'PAC胜' if p['qoe'] > i['qoe'] else ('基线胜' if p['qoe'] < i['qoe'] else '平')
        print(f'  -> ΔQoE={p["qoe"]-i["qoe"]:+.2f}  ΔVMAF={p["vmaf"]-i["vmaf"]:+.2f}  '
              f'Δ卡顿s={p["dstall"]-i["dstall"]:+.2f}  {win}')
    print('-'*100)

# 平均
print('='*100)
print('三条trace平均')
print('='*100)
def avg(lst, k):
    return sum(r[k] for r in lst)/len(lst)
for mk in ['pacnet', 'imdn']:
    lst = agg[mk]
    print(f'{mk:<8} QoE={avg(lst,"qoe"):>7.2f}  VMAF={avg(lst,"vmaf"):.2f}  osc={avg(lst,"osc"):.2f}  '
          f'卡顿次={avg(lst,"nstall"):.1f}  卡顿s={avg(lst,"dstall"):.2f}  切换={avg(lst,"sw"):.1f}  fast={avg(lst,"fc"):.1f}')
pl, il = agg['pacnet'], agg['imdn']
print(f'\n平均ΔQoE={avg(pl,"qoe")-avg(il,"qoe"):+.2f}  平均ΔVMAF={avg(pl,"vmaf")-avg(il,"vmaf"):+.2f}  '
      f'平均Δ卡顿s={avg(pl,"dstall")-avg(il,"dstall"):+.2f}')
wins = sum(1 for tk in traces if rows[tk]['pacnet'] and rows[tk]['imdn'] and rows[tk]['pacnet']['qoe'] > rows[tk]['imdn']['qoe'])
print(f'QoE胜率: PACNet {wins}/{len(traces)}')

print('\n' + '='*100)
print('决策分布（下载档0=240p..4=1080p / 增强档0=不增强,1=e1/l1,2=e2/l2,3=e3/l3）')
print('='*100)
for tk in traces:
    for mk in ['pacnet', 'imdn']:
        r = rows[tk][mk]
        if r:
            print(f'{tk} {mk:<8} 下载{r["dl"]}  增强{r["en"]}')
    print('-'*60)
