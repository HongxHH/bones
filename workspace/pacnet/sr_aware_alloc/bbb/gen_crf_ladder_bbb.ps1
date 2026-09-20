# SR aware allocation cross-content replication: BBB 426-wide CRF ladder
# BBB 30fps, 4s segment = 120 frames; force IDR every 120 frames, scenecut off
$src = "E:\dev\Code\Python\bones\data\video_mp4box\output_1080p.mp4"
$out = "E:\dev\Code\Python\bones\workspace\pacnet\sr_aware_alloc\bbb\ladder"
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($c in @(22,26,30,34,38,42)) {
  $f = Join-Path $out "lr426_crf${c}.mp4"
  if (Test-Path $f) { Write-Output "EXISTS $f"; continue }
  Write-Output "ENC crf=$c ..."
  ffmpeg -y -v error -i $src -vf "scale=426:-2:flags=bicubic" -an -c:v libx264 -crf $c -pix_fmt yuv420p -x264-params "keyint=120:min-keyint=120:scenecut=0" $f
  if ($LASTEXITCODE -ne 0) { Write-Output "FFMPEG_FAIL crf=$c"; exit 1 }
  Write-Output "DONE crf=$c size=$((Get-Item $f).Length)"
}
Write-Output "CRF_LADDER_BBB_ALL_DONE"
