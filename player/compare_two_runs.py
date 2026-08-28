import json

# 第一次实验
with open('output/data/ca_bones_imdn_always_fast-True_video-bbb_trace-report.2010-09-13_1003CEST_20260828-151339.json') as f:
    d1 = json.load(f)

# 第二次（重复）实验
with open('output/data/ca_bones_imdn_always_fast-True_video-bbb_trace-report.2010-09-13_1003CEST_20260828-152848.json') as f:
    d2 = json.load(f)

print('=== 两次实验对比（CARES+VMAF平滑, λ=0.003, α=0.3）===')
print('{:<20} {:>10} {:>10} {:>10}'.format('指标', '第一次', '第二次', '差值'))
print('-' * 55)
for key, label in [('avg_vmaf','avg_vmaf'), ('QOE','QOE'), ('oscillation_vmaf','oscillation'), ('num_stall','num_stall'), ('dur_stall','dur_stall')]:
    v1 = d1.get(key, 0)
    v2 = d2.get(key, 0)
    diff = v2 - v1
    print('{:<20} {:>10.2f} {:>10.2f} {:>+10.2f}'.format(label, v1, v2, diff))

print()
print('level分布 第一次:', d1.get('enhance_level_count'))
print('level分布 第二次:', d2.get('enhance_level_count'))

gpu1 = d1.get('gpu_metrics', {})
gpu2 = d2.get('gpu_metrics', {})
print()
print('{:<20} {:>10} {:>10} {:>10}'.format('GPU指标', '第一次', '第二次', '差值'))
print('-' * 55)
for key, label in [('avg_power_w','avg_power(W)'), ('total_energy_kj','total_energy(kJ)'), ('peak_power_w','peak_power(W)'), ('peak_temp_c','peak_temp(C)')]:
    v1 = gpu1.get(key, 0)
    v2 = gpu2.get(key, 0)
    diff = v2 - v1
    print('{:<20} {:>10.2f} {:>10.2f} {:>+10.2f}'.format(label, v1, v2, diff))
