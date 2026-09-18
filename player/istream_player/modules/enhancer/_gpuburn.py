# -*- coding: utf-8 -*-
"""每帧额外 GPU 计算负载（阶段二实验用），两种模式：

1. 乘法模式（推荐，与离线仿真算力系数 k 同口径）：环境变量 PACNET_BURN_K=1.5 时，
   按"该帧已发生的模型处理耗时 model_ms"额外烧 model_ms*(k-1) 的 matmul，
   使该帧端到端延迟近似放大到 k 倍。出口越深 / 分辨率越高，烧得越多，
   三出口延迟比值保持不变，halt 阈值不会倒挂。对 PAC / IMDN 一视同仁。
2. 加法模式（兼容旧实验）：环境变量 PACNET_EXTRA_MS=17 时，每帧固定烧指定毫秒。

两者都不设置则不启用。burn(model_ms) 的 model_ms 为该帧到推理为止的处理耗时（毫秒）。
"""
import os
import time

import torch

_state = {"inited": False, "k": 1.0, "extra": 0.0,
          "single": 0.5, "a": None, "b": None}


def _ensure():
    global _state
    if _state["inited"]:
        return
    try:
        _state["k"] = float(os.environ.get("PACNET_BURN_K", "0") or 0)
    except ValueError:
        _state["k"] = 0.0
    try:
        _state["extra"] = float(os.environ.get("PACNET_EXTRA_MS", "0") or 0)
    except ValueError:
        _state["extra"] = 0.0
    if _state["k"] > 1.0 or _state["extra"] > 0:
        a = torch.randn(2048, 2048, device="cuda")
        b = torch.randn(2048, 2048, device="cuda")
        with torch.no_grad():
            for _ in range(10):
                a @ b
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(30):
                a @ b
            torch.cuda.synchronize()
            single = (time.perf_counter() - t0) / 30 * 1000
        _state["a"], _state["b"], _state["single"] = a, b, single
        if _state["k"] > 1.0:
            print(f"[gpuburn] MULT mode K={_state['k']:.2f}x, "
                  f"single_matmul={single:.3f}ms, burn=model_ms*{_state['k']-1:.2f}", flush=True)
        else:
            _iters = max(1, round(_state["extra"] / single))
            print(f"[gpuburn] ADD mode EXTRA={_state['extra']:.1f}ms, "
                  f"single_matmul={single:.3f}ms, iters={_iters}", flush=True)
    _state["inited"] = True


def burn(model_ms: float = 0.0):
    """每帧调用一次。model_ms：该帧到推理为止已发生的处理耗时（毫秒）。"""
    _ensure()
    if _state["k"] > 1.0:
        target = float(model_ms) * (_state["k"] - 1.0)
    elif _state["extra"] > 0:
        target = _state["extra"]
    else:
        return
    single = _state["single"]
    iters = int(round(target / single)) if single > 0 else 0
    if iters <= 0:
        return
    a, b = _state["a"], _state["b"]
    with torch.no_grad():
        for _ in range(iters):
            a @ b
