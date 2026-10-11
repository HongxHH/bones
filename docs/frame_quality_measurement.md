# 帧级质量度量规范

全项目唯一的逐帧画质测量标准。所有工作线的离线质量数据、端到端画质评分均从本规范产出的权威数据集读取；禁止各线自建测量口径。

建立时间：2026-10-11。背景：档案 #2（`workspace/vbr_sims/frame_level/`，cv2 打包口径）与档案 #13（`workspace/pacnet/sr_aware_alloc/`，连续 rawvideo 口径）两套指标并存，前者绝对值不可信。

## 1. 金标准管道

逐帧连续测量，不抽帧、不打包：

1. 输入压缩视频经 ffmpeg 解码为连续 rawvideo（`-f rawvideo -pixel_format rgb24`），按显示时间序逐帧产出；
2. 学习类方法：帧转 torch 张量（RGB，[0,1]）→ 模型 forward → 输出尺寸与目标不一致时 `F.interpolate(mode="bicubic", align_corners=False)` 对齐；
3. 逐帧连续喂入 libvmaf，参考视频作为第二输入，`log_fmt=json` 输出**逐帧**分数。两种等价拓扑：
   - 每臂一个 ffmpeg（rawvideo stdin + 参考 + 单 libvmaf），feeder 线程各自喂帧；bicubic 臂可纯 ffmpeg 无 stdin。**默认采用**，实现见 `workspace/quality_baseline/measure_quality_baseline.py`；
   - 同帧各臂纵向堆叠为单条 rawvideo，一次 ffmpeg 内 crop+split 出多个 libvmaf。2 臂可行（范本 `workspace/pacnet/sr_aware_alloc/measure_seg_rd.py:69-145`）；4 臂堆叠+split+多 map 的 filtergraph 在本机调度死锁（2026-10-11 实测），禁止用于 ≥4 臂。

固定项，测量时记录：libvmaf 模型版本、ffmpeg 版本、参考视频路径、输入视频路径、模型权重路径与 MD5、帧率、线程数。

工具链前置：必须使用编进 libvmaf 的 ffmpeg。PATH 中 `E:\dev\tool\ffmpeg-6.1.1-full_build-shared` 为残缺构建（bin 仅 11 文件、无第三方 DLL），conda-forge Windows ffmpeg 不编 libvmaf；用 gyan.dev full 构建（`workspace/ffmpeg_gyan/ffmpeg-9.0.2-full_build/`，版本 9.0.2）并以 `--ffmpeg` 显式指定。曾出现"libvmaf 慢 20 余倍"的误判，根因是命令只对第一输入限时长、参考流被排空全片，非构建问题（见第 3 节）。

帧数限制规则：以文件为输入、只测片段时，必须在**输出位置**加 `-frames:v N`（或对每个输入分别限时长）。只对第一输入加 `-t` 时，参考输入在失真流结束后被 ffmpeg 继续解码排空；120 帧冒烟实测墙钟由 0.7s 膨胀到 80~129s（2026-10-11）。rawvideo stdin 臂同样需要输出级 `-frames:v N` 及时终止参考解码。

## 2. bicubic 上采样

统一在 ffmpeg YUV 域完成：`scale=<W>:<H>:flags=bicubic`，独立 ffmpeg 进程产出 rawvideo。

禁止用 `cv2.resize(INTER_CUBIC)` 或 torch RGB 域 bicubic 代替：RGB 域与 YUV 域结果存在系统性差异，档案 #14 实测同帧段相差约 9~10 VMAF（`workspace/pacnet/modern_sr/probe_modern_sr.py:241` 为 RGB 域路径）。

## 3. 禁止做法

| 禁止项 | 原因 | 出处 |
|---|---|---|
| cv2 VideoWriter mp4v 打包成临时 mp4 再测 | 重编码改变时域加权与帧序，绝对值失真（曾虚高 20~28 分） | `docs/known_issues.md:79-83` |
| 全片稀疏抽帧代表整体 | 段内/帧间方差大，抽帧结论不可复现 | 档案 #8 F6、档案 #9 |
| cv2 RGB 域 bicubic 作基线 | 与 ffmpeg YUV 域系统偏差约 9~10 分 | 档案 #14 P3 |
| 段均值离线单独维护 | 段均值必须从逐帧分数现算，避免两套数 | 本规范第 4 节 |
| 只对第一输入限时长、参考流不限制 | 参考流被排空全片，墙钟膨胀约 20 倍，曾误判为 libvmaf 故障 | 本规范第 1 节帧数限制规则 |

## 4. 权威数据集

目录：`data/quality/`。全项目唯一，所有控制器、仿真器、工作线只读这一份。

每个输入视频一个文件 `data/quality/<video>_<input_res>.json`，schema：

```json
{
  "video": "bbb",
  "measured_at": "2026-10-11T..",
  "ref_video": "data/video_mp4box/output_1080p.mp4",
  "input_video": "data/video_mp4box/output_240p.mp4",
  "input_size": [426, 240], "target_size": [1920, 1080],
  "scale": 4, "fps": 30,
  "frames_total": 19039, "full_segments": 158, "frames_per_seg": 120,
  "libvmaf_model": "vmaf_v0.6.1", "libvmaf_n_threads": 1,
  "methods": {
    "bicubic": {"kind": "ffmpeg_scale_bicubic_yuv", "weights": null,
                 "weights_md5": null, "frames": [19039 个逐帧 float]},
    "sr_low":  {"kind": "IMDN_RTC x4 num_modules=3 nf=6",
                 "weights": "player/.../IMDN_low_bbb_x4.pth",
                 "weights_md5": "<md5>", "frames": []}
  }
}
```

约束：

- 只存逐帧分数；段均值、全片均值、卡顿外的统计量一律从 `frames` 现算；
- 尾段不足 `frames_per_seg` 的帧逐帧保留，不计入完整段统计；
- 学习方法必须记录权重路径与 MD5，权重变更须重测并保留旧文件版本溯源。

## 5. 对齐校验

每次测量前做对齐验证：参考与失真按 offset ∈ {−2,…,2} 平移匹配，正确结果应在 offset=0 处 PSNR 处处最大；范本 `workspace/pacnet/modern_sr/audit_alignment.py`。帧数断言：所有臂帧数等于参考帧数，不等即判失败，禁止用旧数据填充。

## 6. 扩展规则

- **新增方法**：同一管道加一条臂，在同一 `<video>_<input_res>.json` 的 `methods` 内追加，不新建文件、不新建口径；
- **新增输入视频**：先在本规范登记为基准视频，按既有方法网格测全；跨内容验证视频（如 Tears of Steel）只用于泛化结论，不混入基线；
- **新增指标**（LPIPS 等）：可逐帧张量直算（不经打包，口径成立，见档案 #9 G2），结果并入同一文件或同目录并列文件，须在本文件补记口径；
- 任何例外须在本文件写明原因与影响范围。
