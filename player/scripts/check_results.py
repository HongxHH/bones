import json
import sys
import glob
import os

# Find the latest result files
data_dir = "output/data"
files = glob.glob(os.path.join(data_dir, "*.json"))
files.sort(key=os.path.getmtime, reverse=True)

print(f"Found {len(files)} result files\n")

for f in files[:6]:
    print(f"{'='*70}")
    print(f"File: {os.path.basename(f)}")
    print(f"{'='*70}")
    try:
        d = json.load(open(f))
        qoe = d.get('QOE', d.get('qoe', 'N/A'))
        print(f"  QoE:               {qoe}")
        print(f"  avg_vmaf:          {d.get('avg_vmaf', 'N/A')}")
        print(f"  num_stall:         {d.get('num_stall', 'N/A')}")
        print(f"  dur_stall(s):      {d.get('dur_stall', 'N/A')}")
        print(f"  quality_switches:  {d.get('num_quality_switches', 'N/A')}")
        print(f"  avg_bitrate:       {d.get('avg_bitrate', 'N/A')}")
        print(f"  enhancer_aborts:   {d.get('enhancer_abort_total', 'N/A')}")
        if 'enhance_level_count' in d:
            print(f"  enhance_levels:    {d['enhance_level_count']}")
        if 'enhance_level_completed_count' in d:
            print(f"  completed_levels:  {d['enhance_level_completed_count']}")
        if 'gpu_metrics' in d:
            gm = d['gpu_metrics']
            print(f"  gpu_avg_power(W):  {gm.get('avg_power_w', 'N/A')}")
            print(f"  gpu_peak_mem(MB):  {gm.get('peak_memory_mb', 'N/A')}")
        if 'startup_metrics' in d:
            sm = d['startup_metrics']
            print(f"  startup_delay(s):  {sm.get('first_segment_download_s', 'N/A')}")
    except Exception as e:
        print(f"  Error reading: {e}")
    print()
