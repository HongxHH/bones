# Prism control plane

Prism 的 DASH 控制面 + Origin。编排器只调度插件，不拼 ffmpeg 命令。

产品入口（Lab + 画面窗口）：

```
python -m prism
```

打开 http://127.0.0.1:8080/lab/

## 仅启动控制面

在仓库根目录：

```
python -m pip install -r prism/control/requirements.txt
python -m prism serve --port 8080
```

OpenAPI：http://127.0.0.1:8080/docs  
Lab：http://127.0.0.1:8080/lab/

## Phase 0：导入现有 DASH 并给播放器拉

```
curl -X POST http://127.0.0.1:8080/v1/bootstrap/import-dash ^
  -H "Content-Type: application/json" ^
  -d "{\"name\":\"bbb\",\"path\":\"player/data/video_mp4box\",\"experiment_name\":\"local-bbb\"}"
```

轮询 `GET /v1/jobs/{id}` 至 `succeeded`，再：

```
GET /v1/publications
GET /v1/publications/{id}/play
```

播放器（`player/` 目录）：

```
python play.py -i http://127.0.0.1:8080/origin/publications/{id}/bbb.mpd --mod_downloader tcp --mod_status status
```

## 会话

`POST /v1/sessions` 绑定 publication，返回后用 `GET /v1/sessions/{id}/play` 拿会话级 MPD。
`PATCH /v1/sessions/{id}/policy` 可过滤 representation 或切到 dynamic MPD。
会话清单带 `Cache-Control: no-store` 与 `X-Bones-Policy-Version`。

## 多机队列

```
set BONES_REDIS_URL=redis://127.0.0.1:6379/0
python -m prism serve --port 8080
```

未设置时使用进程内队列。需 `pip install redis`。

## 测量 / 实验档案

- 停跑后控制面用本地质量表算 `bones_qoe_v1`（`avg_vmaf − oscillation − stall/段数/10`），写入 `metrics_bundle.json`
- 质量分按 content + 分辨率档 + 模型 + 增强级别查找；没有表时 Bundle 标 `unavailable`，播放照常归档
- 配方里 `quality.prefer` 可加上 `remote_catalog`（拉 `{url}/{content}.json`）和 `compute`（参考片 + 失真片 libvmaf，结果回写目录）
- VMAF 作业：`POST /v1/jobs` 且 `plugin=vmaf`，params 含 `reference` 与 `distorted`；带 `run_id` 时结果并入该 Run 的 Bundle
- Experiment：job 结束后 `archive` hook 写入 `prism/var/experiments/{id}/`（fingerprint + manifest + metrics）

## Lab / 仪器控制台

- Prism Lab：http://127.0.0.1:8080/lab/
- 仪器调试页：player 开启 `--mod_status status` 后访问 http://127.0.0.1:8765

## 声明式 Run

```
python -m prism run prism/experiments/bbb_play.yaml
```

## 验收

控制面已在 `:8080` 时：

```
python prism/scripts/accept_prism.py
pytest prism/control/tests -q
```

## 精确实验清单

- 推荐 workspace：`baseline`（fast-complete 关、OpenCV 出画）。只要段计时、不要画面时用 `headless`。
- `opengl` 渲染器只按段时长 `sleep`，不是 GPU 出画。跨次对比不要把它和 `opencv` 混用。
- 默认可复现项：`use_fast_complete=false`、`collect_system_metrics=false`。`latency_adapt` 默认开启：本机首次（或缺 GPU 指纹）重测 `imdn_latency.json`，之后每次跑完 EWMA 回写；精确对比请显式设 `latency_adapt=false`。
- Lab 运行中轮询 `GET /v1/runs/{id}/live`（尾部样本 + 标量摘要）。完整 Bundle 在停跑或 `GET /metrics` 时计算，不要挂在 1Hz 上。
- BBB + IMDN 的 Lab QoE 画质优先用 `data/quality` 逐帧真测，按段聚合并按播出结局（含 fast-complete 比例）加权。没有权威数据时才回退 analyzer 旧表推算。
- `quality_source` 默认 `static`。设为 `frame_catalog` 会让 BONES 按当前段的权威均值选下载档和增强档，默认实验不要开。`always_greedy` 不读这张表。
- `prefer: compute` 会跑 ffmpeg，只走完整 metrics，不走 live。
- 声明式 Spec 只有一个 arm。`POST /runs/{id}/advance` 与 `arm_mode=parallel` 不是 Lab 路径。
- 带宽：默认 `bw_meter`（段级）。可选 `bw_cont`（连续窗口），类名 `ContinuousBandwidthMeter`。
- NES 里论文主路径是 BONES / always_greedy / IMDN always；PACNet、DRL、multitask 是实验性模块。

## 测试

```
pytest prism/control/tests -q
```
