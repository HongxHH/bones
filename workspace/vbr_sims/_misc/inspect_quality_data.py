import json

# 检查质量数据结构
d = json.load(open(r'E:\dev\Code\Python\bones\data\bbb\imdn_bbb_quality.json'))
print('Keys:', list(d.keys())[:10])
for k in list(d.keys())[:5]:
    v = d[k]
    print(f'  {k}: type={type(v).__name__}', end='')
    if isinstance(v, list):
        print(f', len={len(v)}', end='')
        if len(v) > 0:
            print(f', first_type={type(v[0]).__name__}', end='')
            if isinstance(v[0], (int, float)):
                print(f', first={v[0]:.2f}', end='')
            elif isinstance(v[0], list):
                print(f', first_len={len(v[0])}', end='')
        print()
    elif isinstance(v, dict):
        print(f', subkeys={list(v.keys())[:5]}')
    else:
        print(f', value={v}')

# 检查是否有vmaf相关的key
print('\nVMAF-related keys:')
for k in d.keys():
    if 'vmaf' in k.lower() or 'quality' in k.lower():
        print(f'  {k}')
