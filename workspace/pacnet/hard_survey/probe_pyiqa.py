# -*- coding: utf-8 -*-
import io, traceback
out = io.open(r"E:\dev\Code\Python\bones\workspace\pacnet\hard_survey\probe_out.txt", "w", encoding="utf-8")
try:
    import pyiqa
    out.write("pyiqa ver %s\n" % getattr(pyiqa, "__version__", "?"))
    names = pyiqa.list_models()
    want = [x for x in names if x.lower() in
            ("lpips", "dists", "clipiqa", "ms-ssim", "cw_ssim", "erqa", "niqe", "maniqa", "musiq", "vif")]
    out.write("want metrics: %s\n" % want)
    out.write("total metrics %d\n" % len(names))
except Exception:
    out.write("ERROR\n")
    out.write(traceback.format_exc())
out.close()
print("done")
