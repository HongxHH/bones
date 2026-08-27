import argparse
import torch
import os
import numpy as np
import utils
# import skimage.color as sc
import cv2
from archs import Carn_arch
import imageio
import time
from archs import architecture



parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="CARN")
parser.add_argument("--checkpoint", type=str, default=r"E:\dev\Code\Python\bones\honghong\checkpoints\Carn_video.pth",
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--scale", type=int, default=4,
                    help='upscaling factor')
opt = parser.parse_args()

print(opt)

def read_frames(video_file, scale):
    frames = []
    cap = cv2.VideoCapture(video_file)
    count = 500
    # while True:
    while count > 0:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1920 // scale, 1080 // scale), interpolation=cv2.INTER_CUBIC) # 将视频帧缩放到目标分辨率
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # 将视频帧转换为RGB格式
        frames.append(frame)
        count -= 1
    return frames



cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

if opt.model == "IMDN_ultra":
    model = architecture.IMDN(upscale=opt.scale)
elif opt.model == "IMDN_high":
    model = architecture.IMDN(upscale=opt.scale, nf=32)
elif opt.model == "IMDN_medium":
    model = architecture.IMDN_RTC(upscale=opt.scale)
elif opt.model == "IMDN_low":
    model = architecture.IMDN_RTC(upscale=opt.scale, num_modules=3, nf=6)
elif opt.model == "CARN":
    model = Carn_arch.CARN(upscale=opt.scale)
else:
    raise NotImplementedError

if opt.checkpoint != "":
    model_dict = utils.load_state_dict(opt.checkpoint)
    model.load_state_dict(model_dict['params'], strict=True)
model = model.to(device)

scale2res = {4:"240p", 3:"360p", 2:"480p"}
res = scale2res[opt.scale]

# frames = read_frames('../../data/video_mp4box/stream{}/segment_2.mp4'.format(2), opt.scale)
frames = read_frames(r'E:\dev\Code\Python\bones\data\video_mp4box\output_1080p.mp4', opt.scale)
writer = imageio.get_writer("imdn_{}.mp4".format(res), format='FFMPEG', mode='I', fps=30,
                                codec='libx264', quality=10, pixelformat='yuv444p', macro_block_size=12)

for frame_idx in range(len(frames)):
    frame = frames[frame_idx]
    frame = frame / 255.0
    frame = np.transpose(frame, (2, 0, 1))
    frame = frame[np.newaxis, ...]
    frame = torch.from_numpy(frame).float()
    frames[frame_idx] = frame

start_time = None
outputs = []

for frame_idx in range(len(frames)):
    if frame_idx == 5:
        start_time = time.time()

    with torch.no_grad():
        im_input = frames[frame_idx].to(device)
        out = model(im_input)
        output = out.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        outputs.append(output)

torch.cuda.synchronize()
end_time = time.time()
print("Average time", (end_time - start_time) / (len(frames) - 5))
print("Average FPS", (len(frames) - 5) / (end_time - start_time))

if not opt.checkpoint:
    exit()

# for output in outputs:
#     if output.ndim == 3:
#         output = np.transpose(output, (1, 2, 0))
#     output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
#     writer.append_data(output)
writer.close()
