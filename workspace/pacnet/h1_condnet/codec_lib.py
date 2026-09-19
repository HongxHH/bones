# -*- coding: utf-8 -*-
"""H1 条件网：码流元数据（QP + 前/后向运动矢量）提取与条件张量组装。

逻辑沿用已验证的 h1_codecmeta/extract_mv.py、extract_qp.py（见 docs/h1_codec_metadata.md）：
- MV：PyAV 置 flags2|=1<<28 读 MOTION_VECTORS side data，AVMotionVector 40 字节，
  位移=motion/motion_scale(=4)，source=-1 前向 / +1 后向，dst 为当前帧 16x16 宏块中心；
- QP：h264 不导出 side data，解析 `ffmpeg -threads 1 -debug qp`，单线程下逐帧 nMBW*nMBH 个 QP；
- 帧序：PyAV 解码序 == ffmpeg 单线程序 == 显示序（已在 482 帧上 0 类型错位验证）。

条件张量 7 通道（与 LR 同分辨率，块级最近邻上采样由 QP reshape+repeat 完成）：
  [0]   qp/40
  [1:3] 前向 mvx,mvy /16
  [3]   前向 cover（像素是否被有效前向 MV 覆盖）
  [4:6] 后向 mvx,mvy /16
  [6]   后向 cover
I 帧 MV/cover 全 0，仅 QP 有效。
"""
import os
import re
import struct
import subprocess

import numpy as np
import av

MV_STRUCT = struct.Struct("@iBBhhhh2xqIIH6x")  # 40 字节
assert MV_STRUCT.size == 40
EXPORT_MVS = 1 << 28
MAX_DISP = 48.0
MB = 16
QP_SCALE = 40.0
MV_SCALE = 16.0
NCOND = 7


def _field_from_recs(recs, H, W, source):
    vx = np.zeros((H, W), np.float32)
    vy = np.zeros((H, W), np.float32)
    cover = np.zeros((H, W), np.float32)
    for source_, w, h, sx, sy, dx, dy, flags, mx, my, scale in recs:
        if source_ != source:
            continue
        sc = float(scale) if scale else 4.0
        ax, ay = mx / sc, my / sc
        if np.hypot(ax, ay) > MAX_DISP:
            continue
        x0 = max(0, dx - w // 2); x1 = min(W, dx + (w + 1) // 2)
        y0 = max(0, dy - h // 2); y1 = min(H, dy + (h + 1) // 2)
        if x1 <= x0 or y1 <= y0:
            continue
        vx[y0:y1, x0:x1] = ax
        vy[y0:y1, x0:x1] = ay
        cover[y0:y1, x0:x1] = 1.0
    return vx, vy, cover


def scan_mv(path, wanted):
    """整段解码一次，返回 {idx: (rgb, fwd(vx,vy,cover), bwd(vx,vy,cover))}，仅保留 wanted 帧。

    rgb 直接取自同一 PyAV 帧，保证 LR 与 MV 天然同帧对齐。
    """
    wanted = set(int(i) for i in wanted)
    cont = av.open(path)
    st = cont.streams.video[0]
    st.codec_context.flags2 = int(getattr(st.codec_context, "flags2", 0)) | EXPORT_MVS
    W, H = st.codec_context.width, st.codec_context.height
    out = {}
    for fi, frame in enumerate(cont.decode(st)):
        if fi > max(wanted):
            break
        if fi in wanted:
            rgb = frame.to_ndarray(format="rgb24")
            recs = []
            for sd in frame.side_data:
                if sd.type.name == "MOTION_VECTORS":
                    buf = bytes(memoryview(sd))
                    for o in range(0, len(buf), 40):
                        recs.append(MV_STRUCT.unpack(buf[o:o + 40]))
            out[fi] = (rgb,
                       _field_from_recs(recs, H, W, -1),
                       _field_from_recs(recs, H, W, +1))
    cont.close()
    return out, W, H


def parse_qp(path, wanted, H, W, max_frames=None):
    """ffmpeg 单线程 -debug qp 解析，返回 {idx: QP 像素图(H,W)}，仅保留 wanted 帧。"""
    wanted = set(int(i) for i in wanted)
    nmbw, nmbh = int(np.ceil(W / MB)), int(np.ceil(H / MB))
    nmb = nmbw * nmbh
    cmd = ["ffmpeg", "-hide_banner", "-threads", "1", "-debug", "qp", "-i", path, "-an"]
    if max_frames is not None:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "null", "-"]
    p = subprocess.run(cmd, capture_output=True, text=True, errors="ignore")
    out, fi, cur = {}, -1, None
    for raw in p.stderr.splitlines():
        s = re.sub(r"^\[[^\]]*\]\s*", "", raw).strip()
        m = re.match(r"New frame,\s*type:\s*(\w+)", s)
        if m:
            fi += 1
            cur = []
            continue
        if cur is not None and s.isdigit() and len(s) >= 2 and len(s) % 2 == 0:
            cur.extend(int(s[j:j + 2]) for j in range(0, len(s), 2))
            if len(cur) >= nmb and fi in wanted and len(cur) == nmb and max(cur) <= 51:
                qpmb = np.array(cur, np.float32).reshape(nmbh, nmbw)
                out[fi] = np.repeat(np.repeat(qpmb, MB, 0), MB, 1)[:H, :W]
    return out


def build_frame_cache(video_path, wanted):
    """对一个退化视频，返回 {idx: (rgb, cond(7,H,W) float32)} 与 W,H，QP/MV 与帧严格对齐。"""
    mv, W, H = scan_mv(video_path, wanted)
    qp = parse_qp(video_path, wanted, H, W)
    cache = {}
    for i in wanted:
        rgb, fwd, bwd = mv[i]
        q = qp.get(i)
        if q is None:  # 极端情况下该帧 QP 缺失，用全 0 QP（仍保留 MV）
            q = np.zeros((H, W), np.float32)
        c = np.stack([
            q / QP_SCALE,
            fwd[0] / MV_SCALE, fwd[1] / MV_SCALE, fwd[2],
            bwd[0] / MV_SCALE, bwd[1] / MV_SCALE, bwd[2],
        ], axis=0).astype(np.float32)
        cache[int(i)] = (rgb, c)
    return cache, W, H
