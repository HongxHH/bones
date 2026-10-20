# 流媒体实验平台架构（控制面 / Origin / Player）

> 产品形态与实验工作台设计见 **[Prism 产品设计](./prism_product_design.md)**（产品名 Prism）。

面向研究与实验的 Neural-Enhanced Streaming 平台：服务端产出/分发 DASH，客户端继续作为可插拔播放内核。编排器不拼 ffmpeg CLI；Origin 只读 Store 并生成清单。

## 产品目标

1. 可复现实验：同一 job 配方可重跑，产物可追溯（fingerprint + manifest）
2. 可插拔：转码配方、打包、分发、测量均为插件，pipeline 由插件自描述
3. 多会话分发与 Origin 并发
4. 动态/近实时打包（per-session MPD + policy_version）
5. 水平扩展：`BONES_REDIS_URL` 启用 RedisQueue；worker 按 labels 认领

## 仓库落位

- `prism/control/`：控制面、队列、worker、Origin、插件
- `prism/cli/`、`prism/desk/`：CLI 与桌面窗
- `prism/var/`：控制面本地数据目录
- `player/istream_player/modules/status/`：播放进程内 StatusExporter
- `player/console/`：外置 Web 控制台静态页

## 阶段验收

### Phase 0

- `python -m prism serve --port 8080`
- `POST /v1/bootstrap/import-dash` 导入现有 DASH 树
- player：`python play.py -i http://127.0.0.1:8080/origin/publications/{id}/….mpd --mod_downloader tcp`

### Phase 1

- `POST /v1/sessions` 后两个 player 可同时拉 origin
- `GET /v1/origin/logs?session_id=` 可查访问

### Phase 2

- `PATCH /v1/sessions/{id}/policy` 过滤 representation 或 `mpd_type=dynamic`
- 会话 MPD：`/origin/sessions/{id}/manifest.mpd`（`Cache-Control: no-store`，`X-Bones-Policy-Version`）
- player Scheduler 在 representation 消失时 clamp；MPDProvider `update(force=True)` 强制刷新

### Phase 3

- 本地 worker 带 labels（cpu/transcode/measure）
- `POST /v1/workers/register` + `/claim` + `/complete` 供远程 worker
- `BONES_REDIS_URL=redis://…` → `RedisQueue`
- VMAF：`plugin=vmaf`，params 需 `reference` + `distorted`（或 `distorted_glob`），管道 `setpts+libvmaf`
- Experiment archive hook：`job_*.json` + `manifest.json` + `metrics/`

## 客户端 UI

视频仍在 player 进程（OpenCV/headless）渲染。`--mod_status status` 在 `127.0.0.1:8765` 开控制面：

- `GET /api/status`、`POST /api/pause|resume|stop`、`/ws`
- **U2**：`POST /api/switch_source`（改当前 MPD）、`POST /api/launch`（另起 player）
- 浏览器打开 http://127.0.0.1:8765；经 `/api/server/*` 代理到 `status_server_url`（默认 8080）
- OpenCV HUD：`display_overlay=True`（默认开启）

## 元数据持久化

`prism/var/meta/{collection}/{id}.json` 按实体增量写入；启动时兼容旧的整表 `{collection}.json`。

## 测试

```bash
pytest prism/control/tests -q
```

Seek / 手动锁码率属于播放内核，不在本控制面第一期。
