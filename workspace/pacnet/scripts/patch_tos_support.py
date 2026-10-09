# -*- coding: utf-8 -*-
"""为多视频（TOS）支持修改播放器代码。幂等：已打过的补丁会跳过。"""

def read(p):
    with open(p, 'rb') as f:
        return f.read().decode('utf-8')

def write(p, s):
    with open(p, 'wb') as f:
        f.write(s.encode('utf-8'))

def nl_of(s):
    return '\r\n' if '\r\n' in s else '\n'

def simple_patch(path, marker, old, new):
    s = read(path)
    nl = nl_of(s)
    if marker.replace('\n', nl) in s:
        print(f'skip (patched): {path} :: {marker[:40]!r}')
        return
    old = old.replace('\n', nl)
    new = new.replace('\n', nl)
    n = s.count(old)
    assert n == 1, f'{path}: expected 1 match, got {n} for {old[:60]!r}'
    write(path, s.replace(old, new, 1))
    print(f'patched: {path} :: {marker[:40]!r}')

def anchor_patch(path, marker, anchor, inject):
    s = read(path)
    nl = nl_of(s)
    if marker.replace('\n', nl) in s:
        print(f'skip (patched): {path} :: {marker[:40]!r}')
        return
    anchor = anchor.replace('\n', nl)
    inject = inject.replace('\n', nl)
    assert anchor in s, f'{path}: anchor not found for {marker[:40]!r}'
    write(path, s.replace(anchor, inject, 1))
    print(f'patched: {path} :: {marker[:40]!r}')

# 1. config.py
cfg = r'E:\dev\Code\Python\bones\player\istream_player\config\config.py'
simple_patch(cfg, 'video_dataset: str = "bbb"',
    '    trace: str = ""\n',
    '    trace: str = ""\n'
    '    # 视频数据集标识（bbb/tos），据此切换质量表、per-seg 内容难度与延迟缓存\n'
    '    video_dataset: str = "bbb"\n')

# 2/3. 基类
base = r'E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\enhancer_imdn_always.py'
simple_patch(base, 'self.video_dataset = getattr(config',
    '        self.content_aware = config.content_aware\n',
    '        self.content_aware = config.content_aware\n'
    '        # 视频数据集标识（bbb/tos），切换质量表等数据集相关资源\n'
    '        self.video_dataset = getattr(config, "video_dataset", "bbb")\n')
simple_patch(base, 'imdn_tos_quality.json',
    '        if self.content_aware:\n'
    '            file_name = "imdn_bbb_quality.json"\n'
    '        else:\n'
    '            file_name = "imdn_div2k_quality.json"\n',
    '        if self.content_aware:\n'
    '            file_name = "imdn_bbb_quality.json"\n'
    '        elif getattr(self, "video_dataset", "bbb") == "tos":\n'
    '            file_name = "imdn_tos_quality.json"\n'
    '        else:\n'
    '            file_name = "imdn_div2k_quality.json"\n')

# 4. PACNetEnhancer
pac = r'E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\pacnet_enhancer.py'
anchor_patch(pac, 'file_name = "pacnet_latency_tos.json"',
    '        if os.path.exists(file_name):\n'
    '            saved = json.load(open(file_name))',
    '        if getattr(self, "video_dataset", "bbb") == "tos":\n'
    '            file_name = "pacnet_latency_tos.json"\n'
    '        if os.path.exists(file_name):\n'
    '            saved = json.load(open(file_name))')
anchor_patch(pac, 'gain_file = "per_seg_psnr_tos.json"',
    '        gain_candidates = [\n'
    '            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",\n'
    '                           "workspace", "pacnet", "per_seg_psnr.json"),\n'
    '        ]',
    '        gain_file = "per_seg_psnr_tos.json" if getattr(self, "video_dataset", "bbb") == "tos" else "per_seg_psnr.json"\n'
    '        gain_candidates = [\n'
    '            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",\n'
    '                           "workspace", "pacnet", gain_file),\n'
    '        ]')
anchor_patch(pac, '_qname = "pacnet_tos_quality.json"',
    '        qpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "pacnet_quality.json")',
    '        _player_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))\n'
    '        _qname = "pacnet_tos_quality.json" if getattr(self, "video_dataset", "bbb") == "tos" else "pacnet_quality.json"\n'
    '        qpath = os.path.join(_player_dir, _qname)')

print('ALL PATCH DONE')
