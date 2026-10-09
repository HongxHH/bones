# -*- coding: utf-8 -*-
"""幂等补丁：PAC 与 IMDN 的 enhance_one_frame 在 interpolate 后调用 _gpuburn.burn()。自动检测换行。"""
import io

def patch(path, old, new, tag):
    s = io.open(path, "r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in s else "\n"
    if "_gpuburn import burn" in s:
        print(f"[skip] {tag} already patched"); return
    old = old.replace("\\n", nl)  # not used; build below
    # 用 nl 重建
    raise RuntimeError("see below")

def patch2(path, old_lines, new_lines, tag):
    s = io.open(path, "r", encoding="utf-8", newline="").read()
    nl = "\r\n" if "\r\n" in s else "\n"
    if "_gpuburn import burn" in s:
        print(f"[skip] {tag} already patched"); return
    old = nl.join(old_lines)
    new = nl.join(new_lines)
    assert old in s, f"{tag}: anchor not found"
    s = s.replace(old, new, 1)
    io.open(path, "w", encoding="utf-8", newline="").write(s)
    print(f"[ok] {tag} (nl={'CRLF' if nl==chr(13)+chr(10) else 'LF'})")

pac = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\pacnet_enhancer.py"
patch2(pac,
    ['            tensor = F.interpolate(tensor, size=(self.display_H, self.display_W),',
     '                                   mode="bicubic", align_corners=False)',
     '            await asyncio_sleep()',
     '        tensor = tensor.cpu()'],
    ['            tensor = F.interpolate(tensor, size=(self.display_H, self.display_W),',
     '                                   mode="bicubic", align_corners=False)',
     '            await asyncio_sleep()',
     '            from ._gpuburn import burn as _burn; _burn()',
     '        tensor = tensor.cpu()'],
    "pacnet_enhancer")

im = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\enhancer_imdn_always.py"
patch2(im,
    ["            tensor = torch.nn.functional.interpolate(tensor, size=(self.display_H, self.display_W), mode='bicubic', align_corners=False)",
     "            # 让出控制权给其他任务",
     "            await asyncio.sleep(0)",
     "        # 将张量移回CPU以节省GPU内存",
     "        tensor = tensor.cpu()  # save on CPU to save GPU memory"],
    ["            tensor = torch.nn.functional.interpolate(tensor, size=(self.display_H, self.display_W), mode='bicubic', align_corners=False)",
     "            # 让出控制权给其他任务",
     "            await asyncio.sleep(0)",
     "            from ._gpuburn import burn as _burn; _burn()",
     "        # 将张量移回CPU以节省GPU内存",
     "        tensor = tensor.cpu()  # save on CPU to save GPU memory"],
    "imdn_always")
print("done")
