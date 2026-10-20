"""IMDN vs NanoVSR latency comparison at 480x270 input -> 1920x1080 output"""
import sys, os, time
import torch
import numpy as np

# --- IMDN ---
sys.path.insert(0, r'E:\dev\Code\Python\bones\enhance\IMDN')
from model import architecture
import utils

print("=== IMDN_low (x2, image SR) ===", flush=True)
imdn = architecture.IMDN_RTC(upscale=2, num_modules=3, nf=6)
ckpt = utils.load_state_dict(r'E:\dev\Code\Python\bones\enhance\IMDN\checkpoint\IMDN_low_DIV2K_x2.pth')
imdn.load_state_dict(ckpt, strict=True)
imdn = imdn.cuda().eval()
imdn_params = sum(p.numel() for p in imdn.parameters())
print(f"  Params: {imdn_params:,}", flush=True)
print(f"  GPU mem: {torch.cuda.memory_allocated()/1024/1024:.1f} MB", flush=True)

# Warmup IMDN (single frame, 480x270 -> 960x540, then we compare per-frame)
dummy_imdn = torch.randn(1, 3, 270, 480).cuda()
with torch.no_grad():
    for _ in range(5):
        _ = imdn(dummy_imdn)
    torch.cuda.synchronize()

# Measure IMDN (single frame)
imdn_lats = []
with torch.no_grad():
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = imdn(dummy_imdn)
        torch.cuda.synchronize()
        imdn_lats.append((time.perf_counter() - t0) * 1000)
print(f"  480x270->960x540: {np.mean(imdn_lats):.2f} +/- {np.std(imdn_lats):.2f} ms/frame "
      f"({1000/np.mean(imdn_lats):.0f} FPS)", flush=True)

# --- NanoVSR ---
print("\n=== NanoVSR-644k (x4, video SR, T=15) ===", flush=True)
sys.path.insert(0, r'E:\dev\Code\Python\bones\enhance\nanovsr')
from models.nanovsr import NanoVSR

torch.cuda.empty_cache()
nano = NanoVSR(num_feat=48, num_blocks=12, deploy=False)
ckpt = torch.load(r'E:\dev\Code\Python\bones\enhance\nanovsr\checkpoints\nanovsr_644k.pth',
                   map_location='cpu', weights_only=True)
if 'params' in ckpt: nano.load_state_dict(ckpt['params'])
else: nano.load_state_dict(ckpt)
nano.switch_to_deploy()
nano = nano.cuda().eval()
nano_params = sum(p.numel() for p in nano.parameters())
print(f"  Params: {nano_params:,}", flush=True)
print(f"  GPU mem: {torch.cuda.memory_allocated()/1024/1024:.1f} MB", flush=True)

# Warmup NanoVSR (15 frames, 480x270 -> 1920x1080)
dummy_nano = torch.randn(1, 15, 3, 270, 480).cuda()
with torch.no_grad():
    for _ in range(3):
        _ = nano(dummy_nano)
    torch.cuda.synchronize()

# Measure NanoVSR (15 frames batch)
nano_lats = []
with torch.no_grad():
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = nano(dummy_nano)
        torch.cuda.synchronize()
        nano_lats.append((time.perf_counter() - t0) * 1000)
print(f"  480x270->1920x1080 (15 frames): {np.mean(nano_lats):.1f} +/- {np.std(nano_lats):.1f} ms "
      f"({15/(np.mean(nano_lats)/1000):.1f} FPS)", flush=True)
print(f"  Per frame: {np.mean(nano_lats)/15:.1f} ms", flush=True)

# --- Summary ---
print("\n=== Summary ===", flush=True)
print(f"{'Model':<25} {'Params':>10} {'GPU MB':>8} {'ms/frame':>10} {'FPS':>8} {'Target':>12}", flush=True)
print(f"{'IMDN_low (x2 image)':<25} {imdn_params:>10,} {torch.cuda.memory_allocated()/1024/1024:>8.1f} "
      f"{np.mean(imdn_lats):>10.2f} {1000/np.mean(imdn_lats):>8.0f} {'960x540':>12}", flush=True)
print(f"{'NanoVSR-644k (x4 video)':<25} {nano_params:>10,} {'2.5':>8} "
      f"{np.mean(nano_lats)/15:>10.1f} {15/(np.mean(nano_lats)/1000):>8.1f} {'1920x1080':>12}", flush=True)
print("\nNote: IMDN outputs 960x540 (x2), NanoVSR outputs 1920x1080 (x4).", flush=True)
print("NanoVSR processes 15 frames with temporal context; IMDN processes single frames.", flush=True)
print("DONE", flush=True)
