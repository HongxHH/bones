"""Minimal NanoVSR test - load, warmup, measure, print"""
import sys, os, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.nanovsr import NanoVSR

print("Step 1: Building model...", flush=True)
model = NanoVSR(num_feat=48, num_blocks=12, deploy=False)
print(f"  Params: {sum(p.numel() for p in model.parameters()):,}", flush=True)

print("Step 2: Loading weights...", flush=True)
ckpt = torch.load('checkpoints/nanovsr_644k.pth', map_location='cpu', weights_only=True)
if 'params' in ckpt:
    model.load_state_dict(ckpt['params'])
elif 'state_dict' in ckpt:
    model.load_state_dict(ckpt['state_dict'])
else:
    model.load_state_dict(ckpt)
print("  Weights loaded", flush=True)

print("Step 3: Fusing to deploy mode...", flush=True)
model.switch_to_deploy()
model = model.cuda().eval()
print(f"  GPU memory: {torch.cuda.memory_allocated()/1024/1024:.1f} MB", flush=True)

print("Step 4: Warmup...", flush=True)
dummy = torch.randn(1, 15, 3, 270, 480).cuda()
with torch.no_grad():
    for i in range(3):
        _ = model(dummy)
        torch.cuda.synchronize()
print("  Warmup done", flush=True)

print("Step 5: Latency measurement...", flush=True)
for res_h, res_w in [(270, 480), (360, 640)]:
    x = torch.randn(1, 15, 3, res_h, res_w).cuda()
    lats = []
    with torch.no_grad():
        for _ in range(5):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            lats.append((time.perf_counter() - t0) * 1000)
    print(f"  {res_h}x{res_w}: {np.mean(lats):.1f} +/- {np.std(lats):.1f} ms "
          f"({15/(np.mean(lats)/1000):.1f} FPS)", flush=True)

print("Step 6: Single frame enhancement test...", flush=True)
frame = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
t = torch.from_numpy(frame).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).cuda() / 255.0
with torch.no_grad():
    out = model(t)
out = (out.clamp(0, 1) * 255).byte().squeeze().permute(1, 2, 0).cpu().numpy()
print(f"  Input: {frame.shape} -> Output: {out.shape}", flush=True)
print("ALL DONE", flush=True)
