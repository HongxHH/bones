# -*- coding: utf-8 -*-
"""基类 measure_latency 按数据集切换延迟缓存。"""
p = r'E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\enhancer_imdn_always.py'
s = open(p, 'rb').read().decode('utf-8')
nl = '\r\n' if '\r\n' in s else '\n'
marker = 'imdn_latency_tos.json'
if marker in s:
    print('skip: already patched')
else:
    anchor = ('        # 如果延迟文件已存在，直接加载\n'
              '        if os.path.exists(file_name):\n')
    inject = ('        if getattr(self, "video_dataset", "bbb") == "tos":\n'
              '            file_name = "imdn_latency_tos.json"\n'
              '        # 如果延迟文件已存在，直接加载\n'
              '        if os.path.exists(file_name):\n')
    anchor = anchor.replace('\n', nl)
    inject = inject.replace('\n', nl)
    assert s.count(anchor) == 1, f'anchor count={s.count(anchor)}'
    s = s.replace(anchor, inject, 1)
    open(p, 'wb').write(s.encode('utf-8'))
    print('patched base measure_latency')
import ast
ast.parse(open(p, encoding='utf-8').read())
print('syntax OK')
