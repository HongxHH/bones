# -*- coding: utf-8 -*-
"""修复 get_frame_rate 支持 DASH 分数帧率（24/1）。"""
p = r'E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\enhancer_imdn_always.py'
with open(p, 'rb') as f:
    s = f.read().decode('utf-8')
nl = '\r\n' if '\r\n' in s else '\n'

marker = 'DASH 帧率支持'
old = ('            # 提取帧率并转换为浮点数\n'
       '            frame_rate = np.array(as_obj.frame_rate).astype(float)\n'
       '            return frame_rate\n')
new = ('            # 提取帧率并转换为浮点数（DASH 帧率支持 "24/1" 分数格式）\n'
       '            fr = str(as_obj.frame_rate)\n'
       '            if "/" in fr:\n'
       '                _num, _den = fr.split("/")\n'
       '                return float(_num) / float(_den) if float(_den) != 0 else float(_num)\n'
       '            return float(fr)\n')

if marker in s:
    print('skip: frame rate already patched')
else:
    old_nl = old.replace('\n', nl)
    new_nl = new.replace('\n', nl)
    assert s.count(old_nl) == 1, f'anchor count={s.count(old_nl)}'
    s = s.replace(old_nl, new_nl, 1)
    with open(p, 'wb') as f:
        f.write(s.encode('utf-8'))
    print('patched: frame rate fractional support')
