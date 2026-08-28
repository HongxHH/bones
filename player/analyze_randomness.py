import json

# 第一次实验（level2为主）
with open('output/data/ca_bones_imdn_always_fast-True_video-bbb_trace-report.2010-09-13_1003CEST_20260828-151339.json') as f:
    d1 = json.load(f)

# 第二次实验（level1为主）
with open('output/data/ca_bones_imdn_always_fast-True_video-bbb_trace-report.2010-09-13_1003CEST_20260828-152848.json') as f:
    d2 = json.load(f)

print('=== 前20段增强级别对比 ===')
print('{:<6} {:<10} {:<10} {:<10}'.format('段号', '第一次', '第二次', '差异'))
print('-' * 40)

segs1 = d1.get('segments', [])
segs2 = d2.get('segments', [])

for i in range(min(20, len(segs1), len(segs2))):
    s1 = segs1[i]
    s2 = segs2[i]
    e1 = s1.get('enhance_action', '?')
    e2 = s2.get('enhance_action', '?')
    v1 = s1.get('real_vmaf', 0)
    v2 = s2.get('real_vmaf', 0)
    diff = '不同' if e1 != e2 else ''
    print('{:<6} {:<10} {:<10} {:<10}'.format(
        i+1, 
        'L{} vmaf={:.1f}'.format(e1, v1) if v1 else 'L{}'.format(e1),
        'L{} vmaf={:.1f}'.format(e2, v2) if v2 else 'L{}'.format(e2),
        diff
    ))

print()
print('=== 第一次实验 level 切换次数 ===')
level_transitions_1 = 0
for i in range(1, len(segs1)):
    if segs1[i].get('enhance_action') != segs1[i-1].get('enhance_action'):
        level_transitions_1 += 1
print('level切换次数:', level_transitions_1, '/', len(segs1)-1, '段')

print()
print('=== 第二次实验 level 切换次数 ===')
level_transitions_2 = 0
for i in range(1, len(segs2)):
    if segs2[i].get('enhance_action') != segs2[i-1].get('enhance_action'):
        level_transitions_2 += 1
print('level切换次数:', level_transitions_2, '/', len(segs2)-1, '段')

print()
print('=== 第一次实验各 level 连续段长度统计 ===')
from collections import defaultdict
runs_1 = defaultdict(list)
current_level = segs1[0].get('enhance_action')
run_len = 1
for i in range(1, len(segs1)):
    if segs1[i].get('enhance_action') == current_level:
        run_len += 1
    else:
        runs_1[current_level].append(run_len)
        current_level = segs1[i].get('enhance_action')
        run_len = 1
runs_1[current_level].append(run_len)
for level, runs in sorted(runs_1.items()):
    print('  L{}: 出现{}次, 平均连续{:.1f}段, 最长{}段'.format(
        level, len(runs), sum(runs)/len(runs), max(runs)))

print()
print('=== 第二次实验各 level 连续段长度统计 ===')
runs_2 = defaultdict(list)
current_level = segs2[0].get('enhance_action')
run_len = 1
for i in range(1, len(segs2)):
    if segs2[i].get('enhance_action') == current_level:
        run_len += 1
    else:
        runs_2[current_level].append(run_len)
        current_level = segs2[i].get('enhance_action')
        run_len = 1
runs_2[current_level].append(run_len)
for level, runs in sorted(runs_2.items()):
    print('  L{}: 出现{}次, 平均连续{:.1f}段, 最长{}段'.format(
        level, len(runs), sum(runs)/len(runs), max(runs)))
