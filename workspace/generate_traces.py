"""生成多带宽场景的网络轨迹"""
import json
import numpy as np
import os

TRACE_DIR = r"E:\dev\Code\Python\bones\data\trace"

def generate_trace(name, bandwidths, duration_per_period=30000, latency_range=(50, 200)):
    """生成轨迹"""
    trace = []
    for bw in bandwidths:
        latency = np.random.randint(latency_range[0], latency_range[1])
        trace.append({
            "duration_ms": duration_per_period,
            "bandwidth_kbps": int(bw),
            "latency_ms": int(latency)
        })
    path = os.path.join(TRACE_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Generated {name}: {len(trace)} periods, avg bw={np.mean(bandwidths):.0f} kbps")
    return path

np.random.seed(42)

# 3G: 低带宽，500-2000 kbps
bw_3g = np.random.uniform(500, 2000, 20)
generate_trace("3g_low", bw_3g)

# 4G: 中带宽，2000-6000 kbps
bw_4g = np.random.uniform(2000, 6000, 20)
generate_trace("4g_medium", bw_4g)

# WiFi: 高带宽，5000-15000 kbps
bw_wifi = np.random.uniform(5000, 15000, 20)
generate_trace("wifi_high", bw_wifi)

# 波动: 带宽剧烈波动，500-10000 kbps
bw_volatile = []
for _ in range(20):
    if np.random.random() < 0.3:
        bw_volatile.append(np.random.uniform(500, 1500))  # 低谷
    else:
        bw_volatile.append(np.random.uniform(4000, 10000))  # 高峰
generate_trace("volatile", bw_volatile)

# 原始dummy也保留
print("\nAll traces generated.")
