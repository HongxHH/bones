"""Multi-task enhancement latency measurement: NAFNet (deblur) + baseline"""
import sys, os, time
import torch
import numpy as np

# --- NAFNet (Deblur) ---
sys.path.insert(0, r'E:\dev\Code\Python\bones\enhance\NAFNet')
from basicsr.models.archs.NAFNet_arch import NAFNet

print("=== NAFNet (Deblur) ===", flush=True)

# Standard NAFNet-GoPro config (~1.7M params)
nafnet = NAFNet(img_channel=3, width=32, middle_blk_num=1,
                enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
nafnet = nafnet.cuda().eval()
naf_params = sum(p.numel() for p in nafnet.parameters())
print(f"  Params: {naf_params:,}", flush=True)
print(f"  GPU mem: {torch.cuda.memory_allocated()/1024/1024:.1f} MB", flush=True)

# Warmup at 1080p
dummy = torch.randn(1, 3, 1080, 1920).cuda()
with torch.no_grad():
    for _ in range(3):
        _ = nafnet(dummy)
    torch.cuda.synchronize()

# Measure at different resolutions
for h, w in [(540, 960), (720, 1280), (1080, 1920)]:
    x = torch.randn(1, 3, h, w).cuda()
    lats = []
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = nafnet(x)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
    print(f"  {h}x{w}: {np.mean(lats):.1f} +/- {np.std(lats):.1f} ms "
          f"({1000/np.mean(lats):.1f} FPS)", flush=True)

del nafnet
torch.cuda.empty_cache()

# --- NAFNet-Small (lighter deblur) ---
print("\n=== NAFNet-Small (Deblur, width=16) ===", flush=True)
naf_small = NAFNet(img_channel=3, width=16, middle_blk_num=1,
                    enc_blk_nums=[1, 1, 1, 14], dec_blk_nums=[1, 1, 1, 1])
naf_small = naf_small.cuda().eval()
small_params = sum(p.numel() for p in naf_small.parameters())
print(f"  Params: {small_params:,}", flush=True)
print(f"  GPU mem: {torch.cuda.memory_allocated()/1024/1024:.1f} MB", flush=True)

dummy = torch.randn(1, 3, 1080, 1920).cuda()
with torch.no_grad():
    for _ in range(3):
        _ = naf_small(dummy)
    torch.cuda.synchronize()

for h, w in [(540, 960), (1080, 1920)]:
    x = torch.randn(1, 3, h, w).cuda()
    lats = []
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = naf_small(x)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
    print(f"  {h}x{w}: {np.mean(lats):.1f} +/- {np.std(lats):.1f} ms "
          f"({1000/np.mean(lats):.1f} FPS)", flush=True)

del naf_small
torch.cuda.empty_cache()

# --- Frame Duplication (baseline FI, zero-cost) ---
print("\n=== Frame Duplication (baseline FI) ===", flush=True)
print("  Cost: 0 ms (simply repeat previous frame)", flush=True)
print("  Quality: low (stuttering on motion)", flush=True)

# --- Summary ---
print("\n=== Multi-Task Latency Summary (RTX 5080) ===", flush=True)
print(f"{'Task':<25} {'Model':<20} {'Params':>10} {'1080p ms':>10} {'FPS':>8}", flush=True)
print(f"{'SR (240p tier)':<25} {'NanoVSR-644k':<20} {'644,245':>10} {'12.3':>10} {'81.0':>8}", flush=True)
print(f"{'SR (480p tier)':<25} {'IMDN_low':<20} {'3,744':>10} {'0.69':>10} {'1448':>8}", flush=True)
print(f"{'Deblur':<25} {'NAFNet (width32)':<20} {naf_params:>10,} {'see above':>10} {'':>8}", flush=True)
print(f"{'Deblur (light)':<25} {'NAFNet-Small':<20} {small_params:>10,} {'see above':>10} {'':>8}", flush=True)
print(f"{'Frame Interpolation':<25} {'RIFE-lite':<20} {'~3M':>10} {'TBD':>10} {'TBD':>8}", flush=True)
print(f"{'FI (baseline)':<25} {'Frame Dup':<20} {'0':>10} {'0':>10} {'inf':>8}", flush=True)
print("\nNote: RIFE-lite requires train_log model download (not yet available).", flush=True)
print("DONE", flush=True)
