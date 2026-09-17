@echo off
cd /d E:\dev\Code\Python\bones\data\tears_of_steel
if not exist stream0 mkdir stream0
if not exist stream1 mkdir stream1
if not exist stream2 mkdir stream2
if not exist stream3 mkdir stream3
if not exist stream4 mkdir stream4
ffmpeg -y -nostdin -i "E:\dev\Data\tears_of_steel_1080p.mp4" ^
 -filter_complex "[0:v]split=5[s0][s1][s2][s3][s4];[s0]scale=426:176:flags=bicubic[v0];[s1]scale=640:268:flags=bicubic[v1];[s2]scale=854:356:flags=bicubic[v2];[s3]scale=1280:532:flags=bicubic[v3];[s4]scale=1920:800:flags=bicubic[v4]" ^
 -map "[v0]" -map "[v1]" -map "[v2]" -map "[v3]" -map "[v4]" ^
 -c:v libx264 -preset fast -g 96 -keyint_min 96 -sc_threshold 0 -an ^
 -b:v:0 400k -maxrate:v:0 800k -bufsize:v:0 1600k ^
 -b:v:1 800k -maxrate:v:1 1600k -bufsize:v:1 3200k ^
 -b:v:2 1200k -maxrate:v:2 2400k -bufsize:v:2 4800k ^
 -b:v:3 2400k -maxrate:v:3 4800k -bufsize:v:3 9600k ^
 -b:v:4 4800k -maxrate:v:4 9600k -bufsize:v:4 19200k ^
 -f dash -seg_duration 4 -use_template 1 -use_timeline 0 ^
 -init_seg_name "stream$RepresentationID$/segment_.mp4" ^
 -media_seg_name "stream$RepresentationID$/segment_$Number$.m4s" ^
 -adaptation_sets "id=0,streams=v" ^
 tos.mpd
echo DASH_EXIT=%ERRORLEVEL%
