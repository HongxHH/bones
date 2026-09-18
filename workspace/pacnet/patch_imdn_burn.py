# -*- coding: utf-8 -*-
"""补丁：IMDN enhancer 乘法 gpuburn（幂等，保持原换行符）。"""
import io, os

PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "player", "istream_player", "modules", "enhancer", "enhancer_imdn_always.py")

with io.open(PATH, "r", encoding="utf-8", newline="") as f:
    raw = f.read()
nl = "\r\n" if "\r\n" in raw else "\n"
s = raw.replace("\r\n", "\n")

# 1) model(tensor) 前后加同步计时
OLD1 = (
"        with torch.no_grad():\n"
"            # 模型前向传播\n"
"            tensor = model(tensor)\n"
)
NEW1 = (
"        with torch.no_grad():\n"
"            # 模型前向传播\n"
"            import time as _time\n"
"            torch.cuda.synchronize()\n"
"            _t0 = _time.perf_counter()\n"
"            tensor = model(tensor)\n"
"            torch.cuda.synchronize()\n"
"            fwd_ms = (_time.perf_counter() - _t0) * 1000.0\n"
)

# 2) burn() 改为 burn(fwd_ms)
OLD2 = "            from ._gpuburn import burn as _burn; _burn()\n"
NEW2 = "            from ._gpuburn import burn as _burn; _burn(fwd_ms)\n"

changed = []
if "fwd_ms = (_time.perf_counter() - _t0)" in s:
    changed.append("timing already present")
else:
    if OLD1 not in s:
        raise SystemExit("[error] OLD1 anchor not found")
    s = s.replace(OLD1, NEW1, 1)
    changed.append("timing inserted")

if "_burn(fwd_ms)" in s:
    changed.append("burn(fwd_ms) already present")
else:
    if OLD2 not in s:
        raise SystemExit("[error] OLD2 anchor not found")
    s = s.replace(OLD2, NEW2, 1)
    changed.append("burn arg updated")

with io.open(PATH, "w", encoding="utf-8", newline="") as f:
    f.write(s.replace("\n", nl))
print("[ok] enhancer_imdn_always.py:", ", ".join(changed), "newline=%r" % nl)
