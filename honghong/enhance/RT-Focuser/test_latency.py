import sys, time, torch, torch.nn.functional as F, numpy as np
sys.path.insert(0, '.')
from model.rt_focuser_model import RT_Focuser_Standard

def pad_to_multiple(x, multiple=32):
    """Pad input to multiple of `multiple`, return padded tensor and original size"""
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, h, w

print('=== RT-Focuser (Deblur, padded to 32x) ===', flush=True)
model = RT_Focuser_Standard()
params = sum(p.numel() for p in model.parameters())
print(f'Params: {params:,}', flush=True)

ckpt = torch.load('Pretrained_Weights/GoPro_RT_Focuser_Standard_256.pth', map_location='cpu', weights_only=True)
model.load_state_dict(ckpt, strict=True)
model = model.cuda().eval()
print(f'GPU mem: {torch.cuda.memory_allocated()/1024/1024:.1f} MB', flush=True)

for precision in ['fp32', 'fp16']:
    m = model.half() if precision == 'fp16' else model.float()
    for h, w in [(540,960), (720,1280), (1080,1920)]:
        x = torch.randn(1,3,h,w).cuda()
        if precision == 'fp16': x = x.half()
        x_pad, orig_h, orig_w = pad_to_multiple(x, 32)
        with torch.no_grad():
            for _ in range(3):
                _ = m(x_pad); torch.cuda.synchronize()
            lats = []
            for _ in range(5):
                torch.cuda.synchronize(); t0=time.perf_counter()
                out = m(x_pad)
                out = out[:, :, :orig_h, :orig_w]  # crop back
                torch.cuda.synchronize()
                lats.append((time.perf_counter()-t0)*1000)
        print(f'  {precision} {h}x{w}: {np.mean(lats):.1f}ms ({1000/np.mean(lats):.1f} FPS)', flush=True)

print()
print('=== Deblur Model Comparison (1080p, RTX 5080) ===', flush=True)
print('Model                    Params        FP32            FP16', flush=True)
print('NAFNet (17M)             17,111,907   157ms/6.4FPS    ~85ms/12FPS', flush=True)
print('NAFNet-Small (2.7M)       2,683,059   56.8ms/17.6FPS  30.8ms/32.4FPS', flush=True)
print(f'RT-Focuser (5.86M)        {params:>10,}   see above       see above', flush=True)
print()
print('Note: RT-Focuser requires input padding to 32-pixel multiples.', flush=True)
print('DONE', flush=True)
