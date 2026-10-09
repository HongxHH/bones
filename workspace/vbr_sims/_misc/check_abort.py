import json

with open(r"E:\dev\Code\Python\bones\player\output\data\mt_bones_multitask_fast-True_video-bbb_trace-trace0000_20260829-131812.json") as f:
    d = json.load(f)

print("abort_reason_count:", d.get("abort_reason_count", "N/A"))
print("abort_wasted_frames_total:", d.get("abort_wasted_frames_total", "N/A"))
print("abort_wasted_frames_by_level:", d.get("abort_wasted_frames_by_level", "N/A"))
print("enhancer_abort_total:", d.get("enhancer_abort_total", "N/A"))

segments = d.get("segments", [])
abort_segs = [s for s in segments if s.get("abort_reason") and s.get("abort_reason") != ""]
print("有abort_reason的段数:", len(abort_segs))
for s in abort_segs[:10]:
    print("  seg%s: abort=%s, wasted=%s, enhanced=%s, is_enhance=%s" % (
        s.get("index", "?"), s.get("abort_reason"),
        s.get("wasted_enhanced_cnt", 0), s.get("enhanced_cnt", 0),
        s.get("is_enhance")))
