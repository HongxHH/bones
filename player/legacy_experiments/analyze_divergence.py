import json

with open('output/data/ca_bones_imdn_always_fast-True_video-bbb_trace-report.2010-09-13_1003CEST_20260828-151339.json') as f:
    d1 = json.load(f)
with open('output/data/ca_bones_imdn_always_fast-True_video-bbb_trace-report.2010-09-13_1003CEST_20260828-152848.json') as f:
    d2 = json.load(f)

segs1 = d1.get('segments', [])
segs2 = d2.get('segments', [])

# 找出第一次出现分歧的段
first_diff = None
for i in range(min(len(segs1), len(segs2))):
    e1 = segs1[i].get('enhance_action')
    e2 = segs2[i].get('enhance_action')
    if e1 != e2:
        first_diff = i
        break

print('第一次分歧出现在第', first_diff + 1 if first_diff is not None else '无', '段')

if first_diff is not None:
    print()
    print('=== 分歧前后10段对比 ===')
    start = max(0, first_diff - 5)
    end = min(len(segs1), len(segs2), first_diff + 15)
    print('{:<6} {:<12} {:<12} {:<8}'.format('段号', '第一次', '第二次', '分歧'))
    print('-' * 45)
    for i in range(start, end):
        s1 = segs1[i]
        s2 = segs2[i]
        e1 = s1.get('enhance_action', '?')
        e2 = s2.get('enhance_action', '?')
        v1 = s1.get('real_vmaf', 0)
        v2 = s2.get('real_vmaf', 0)
        b1 = s1.get('buffer_level', 0) if s1.get('buffer_level') else 0
        b2 = s2.get('buffer_level', 0) if s2.get('buffer_level') else 0
        diff = '***' if e1 != e2 else ''
        print('{:<6} L{} v={:.1f}  L{} v={:.1f}  {}'.format(
            i+1, e1, v1, e2, v2, diff))

# 统计分歧点之后的 level 分布差异
print()
print('=== 分歧点之后的 level 分布 ===')
from collections import Counter
if first_diff is not None:
    levels1_after = Counter()
    levels2_after = Counter()
    for i in range(first_diff, len(segs1)):
        levels1_after[segs1[i].get('enhance_action')] += 1
    for i in range(first_diff, len(segs2)):
        levels2_after[segs2[i].get('enhance_action')] += 1
    print('第一次(分歧后{}段):'.format(len(segs1)-first_diff), dict(levels1_after))
    print('第二次(分歧后{}段):'.format(len(segs2)-first_diff), dict(levels2_after))

# 找出最长连续段的位置
print()
print('=== 第一次实验最长连续段位置 ===')
max_run = 0
max_start = 0
max_level = None
current_level = segs1[0].get('enhance_action')
run_start = 0
for i in range(1, len(segs1)):
    if segs1[i].get('enhance_action') != current_level:
        run_len = i - run_start
        if run_len > max_run:
            max_run = run_len
            max_start = run_start
            max_level = current_level
        current_level = segs1[i].get('enhance_action')
        run_start = i
run_len = len(segs1) - run_start
if run_len > max_run:
    max_run = run_len
    max_start = run_start
    max_level = current_level
print('最长连续段: L{}, 第{}-{}段, 共{}段'.format(max_level, max_start+1, max_start+max_run, max_run))

print()
print('=== 第二次实验最长连续段位置 ===')
max_run = 0
max_start = 0
max_level = None
current_level = segs2[0].get('enhance_action')
run_start = 0
for i in range(1, len(segs2)):
    if segs2[i].get('enhance_action') != current_level:
        run_len = i - run_start
        if run_len > max_run:
            max_run = run_len
            max_start = run_start
            max_level = current_level
        current_level = segs2[i].get('enhance_action')
        run_start = i
run_len = len(segs2) - run_start
if run_len > max_run:
    max_run = run_len
    max_start = run_start
    max_level = current_level
print('最长连续段: L{}, 第{}-{}段, 共{}段'.format(max_level, max_start+1, max_start+max_run, max_run))
