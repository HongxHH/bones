import cv2, numpy as np, sys
sys.path.insert(0, r"E:\dev\Code\Python\bones\workspace\pacnet\hard_survey")
sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
sys.path.insert(0, ".")
import measure_hard_survey as H
try:
    cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); print("DIS ok")
except Exception as e:
    print("DIS missing", e)
frames = H.read_frames(H.DEG["426_crf34"], [1000,1001,1002,1003,1004,1005,1006])
print("frames", len(frames), frames[0].shape)
import measure_warp_oracle as W
c = frames[3]; gc=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY)
def yd(a,b):
    ya=cv2.cvtColor(a,cv2.COLOR_BGR2YUV)[:,:,0].astype(float)
    yb=cv2.cvtColor(b,cv2.COLOR_BGR2YUV)[:,:,0].astype(float)
    return float(np.abs(ya-yb).mean())
for j in [0,2,4,6]:
    gn=cv2.cvtColor(frames[j],cv2.COLOR_BGR2GRAY)
    fl=W.flow_center_to_neighbor(gc,gn)
    w=W.warp_to_center(frames[j],fl)
    print(f"off={j-3:+d} mag={np.sqrt((fl**2).sum(-1)).mean():.2f} rawdiff={yd(frames[j],c):.2f} warpdiff={yd(w,c):.2f}")
