import torch, sys, os
sys.path.insert(0, "."); sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
from istream_player.modules.enhancer.imdn_model import IMDN_RTC
from cond_model import CondIMDNCat, WARM_L2, count_params
torch.manual_seed(0)
base = IMDN_RTC(upscale=4); base.load_state_dict(torch.load(WARM_L2, map_location="cpu")); base.eval()
cat = CondIMDNCat(upscale=4); mu = cat.load_warm(); cat.eval()
print("missing(all should be model.0.*):", mu.missing_keys)
print("unexpected:", mu.unexpected_keys)
print("params base/cat:", count_params(base), count_params(cat))
x = torch.randn(2,3,48,48)
with torch.no_grad():
    a = base(x); b = cat(x, None); c = cat(x, torch.zeros(2,7,48,48)); d = cat(x, torch.randn(2,7,48,48))
print("zero-cond equals warm backbone maxdiff:", float((a-b).abs().max()))
print("explicit zeros equals:", float((a-c).abs().max()))
print("random cond changes output maxdiff:", float((a-d).abs().max()))
