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

- VMAF：`POST /v1/jobs` 且 `plugin=vmaf`，params 含 `reference` 与 `distorted`
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

## 测试

```
pytest prism/control/tests -q
```
