# -*- coding: utf-8 -*-
"""H1 数据通路：从 H.264 解码帧 side data 提取运动矢量，构建与帧对齐的 MV 场。

AVMotionVector（ffmpeg libavutil/motion_vector.h，native 对齐，40 字节）：
  int32 source; uint8 w,h; int16 src_x,src_y,dst_x,dst_y; uint64 flags;
  uint32 motion_x,motion_y; uint16 motion_scale; (+6B 尾填充)
字段语义（实测 ffmpeg 6.1 / h264）：
  - source: -1=past 前向（P 帧全部、B 帧前向分量），+1=future 后向（仅 B 帧）；
  - dst_x/dst_y：当前帧 16x16 宏块中心（像素，网格 8,24,...，末块可越界）；
  - motion_x/y：完整运动矢量（含整像素+1/4 亚像素），位移 = motion / motion_scale(=4)；
  - I 帧无 side data，MV 场为 0。
"""
import os, struct, io
import numpy as np
import av

MV_STRUCT = struct.Struct("@iBBhhhh2xqIIH6x")  # 40 字节
assert MV_STRUCT.size == 40, MV_STRUCT.size
EXPORT_MVS = 1 << 28
MAX_DISP = 48.0  # px，超过视为帧外/无纹理误配丢弃（x264 搜索范围内单帧位移远小于此）


def decode_with_mv(path, max_frames=600):
    cont = av.open(path)
    st = cont.streams.video[0]
    st.codec_context.flags2 = int(getattr(st.codec_context, "flags2", 0)) | EXPORT_MVS
    W, H = st.codec_context.width, st.codec_context.height
    out = []
    for fi, frame in enumerate(cont.decode(st)):
        if fi >= max_frames:
            break
        rgb = frame.to_ndarray(format="rgb24")
        recs = []
        for sd in frame.side_data:
            if sd.type.name == "MOTION_VECTORS":
                buf = bytes(memoryview(sd))
                for o in range(0, len(buf), 40):
                    recs.append(MV_STRUCT.unpack(buf[o:o+40]))
        out.append((fi, str(frame.pict_type), rgb, recs))
    cont.close()
    return out, W, H


def mv_field(H, W, recs, source_pref=-1):
    """返回 fx,fy（像素位移，前向分量）与 cover（该像素是否被有效 MV 块覆盖）。"""
    fx = np.zeros((H, W), np.float32)
    fy = np.zeros((H, W), np.float32)
    cover = np.zeros((H, W), np.float32)
    cand = [r for r in recs if r[0] == source_pref]
    if not cand:
        cand = [r for r in recs if r[0] == -1]
    for source, w, h, sx, sy, dx, dy, flags, mx, my, scale in cand:
        sc = float(scale) if scale else 4.0
        vx, vy = mx / sc, my / sc
        if np.hypot(vx, vy) > MAX_DISP:  # 帧外/场景切换误配
            continue
        x0 = max(0, dx - w // 2); x1 = min(W, dx + (w + 1) // 2)
        y0 = max(0, dy - h // 2); y1 = min(H, dy + (h + 1) // 2)
        if x1 <= x0 or y1 <= y0:
            continue
        fx[y0:y1, x0:x1] = vx
        fy[y0:y1, x0:x1] = vy
        cover[y0:y1, x0:x1] = 1.0
    return fx, fy, cover


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = r"E:\dev\Code\Python\bones\workspace\pacnet\hard_survey\lr_426_crf34.mp4"
    outdir = r"E:\dev\Code\Python\bones\workspace\pacnet\h1_codecmeta"
    frames, W, H = decode_with_mv(path, max_frames=900)
    log = io.open(os.path.join(outdir, "extract_mv.txt"), "w", encoding="utf-8")
    scored = []
    fields = {}
    for fi, ptype, rgb, recs in frames:
        fx, fy, cover = mv_field(H, W, recs)
        mag = np.hypot(fx, fy)
        cov = cover.mean()
        active = mag[cover > 0]
        med = float(np.median(active)) if active.size else 0.0  # 中位数抗异常
        scored.append((med, cov, fi, ptype))
        fields[fi] = (rgb, fx, fy, mag, cover)
    # 要求覆盖率足够，按中位位移排序
    cand = [t for t in scored if t[1] > 0.3]
    cand.sort(reverse=True)
    log.write(f"frames={len(frames)} W={W} H={H}\n")
    log.write("top by median|MV| (med px, coverage, fi, type):\n")
    for t in cand[:10]:
        log.write("  med=%.3f cov=%.2f fi=%d %s\n" % t)
    picks = []
    for med, cov, fi, ptype in cand:
        if all(abs(fi - q) > 40 for q in picks):
            picks.append(fi)
        if len(picks) == 4:
            break
    for k, fi in enumerate(picks):
        rgb, fx, fy, mag, cover = fields[fi]
        fig, ax = plt.subplots(1, 1, figsize=(10, 4.2), dpi=110)
        ax.imshow(rgb)
        step = 8
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        vx, vy, cv = fx[ys, xs], fy[ys, xs], cover[ys, xs]
        m = cv > 0
        ax.quiver(xs[m], ys[m], vx[m], vy[m], color="lime", scale=80, width=0.004, headwidth=4)
        ax.set_title(f"frame {fi}: forward MV (lime, px), median={np.median(np.hypot(vx[m],vy[m])):.2f}")
        ax.axis("off"); fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"mv_check_{k}_f{fi}.png")); plt.close(fig)
    fi = picks[0]
    rgb, fx, fy, mag, cover = fields[fi]
    np.savez_compressed(os.path.join(outdir, "mv_align_sample.npz"),
                        rgb=rgb, mvx=fx, mvy=fy, cover=cover, frame_idx=fi, W=W, H=H)
    log.write(f"sample fi={fi} rgb{rgb.shape} mvx{fx.shape} x[{fx.min():.2f},{fx.max():.2f}] "
              f"y[{fy.min():.2f},{fy.max():.2f}] cover={cover.mean():.2f}\nDONE\n")
    log.close(); print("done picks", picks)


if __name__ == "__main__":
    main()
