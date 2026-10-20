# Prism

本地神经增强流媒体**实验工作台**。

## 布局

```
prism/
  cli/          # CLI、launcher、agent、frame_sink
  control/      # 控制面（API / Lab / Origin / Run）
  desk/         # 桌面窗（左 Lab + 右画面）
  experiments/  # 声明式配方 YAML
  scripts/      # restart_ui / accept 等工具
  var/          # 运行时数据（meta / runs / archives）
player/         # 仪器（本机播放 / 增强），独立目录
```

## 启动

在仓库根目录：

```
python -m prism
```

会打开 Prism 窗口：左侧实验台，右侧画面。控制面没在运行时会先拉起。

只在系统浏览器里打开实验台：

```
python -m prism lab
```

仅启动控制面：

```
python -m prism serve --port 8080
```

常用参数：

```
python -m prism --port 8080 --no-server
python -m prism lab --no-browser
```

## Lab / CLI

```
python -m prism run prism/experiments/bbb_play.yaml
python -m prism status <run_id>
python -m prism export <run_id> -o out.csv
python -m prism stop <run_id>
python -m prism agent --port 8079
```

工具脚本（在 `prism/scripts/`）：

```
restart_ui.bat                 # 仓库根：只重启桌面窗
python prism/scripts/accept_prism.py
```

测试：

```
pytest prism/control/tests -q
```

产品设计见 [`docs/prism_product_design.md`](../docs/prism_product_design.md)。
