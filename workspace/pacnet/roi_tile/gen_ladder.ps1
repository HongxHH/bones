# 生成均匀码率阶梯（bicubic 下采样 + libx264），用于 ROI 方案的等码率对手前沿
$src = "E:\dev\Data\tears_of_steel_1080p.mp4"
$out = "E:\dev\Code\Python\bones\workspace\pacnet\roi_tile\ladder"
New-Item -ItemType Directory -Force -Path $out | Out-Null
# width height crf
$jobs = @(
  @(640,266,34), @(640,266,30), @(640,266,26), @(640,266,22),
  @(854,356,34), @(854,356,30), @(854,356,26)
)
foreach ($j in $jobs) {
  $w=$j[0]; $h=$j[1]; $c=$j[2]
  $f = Join-Path $out "lr_${w}_crf${c}.mp4"
  if (Test-Path $f) { Write-Output "EXISTS $f"; continue }
  Write-Output "ENC w=$w crf=$c ..."
  ffmpeg -y -v error -i $src -vf "scale=${w}:${h}:flags=bicubic" -an -c:v libx264 -crf $c -pix_fmt yuv420p -g 48 $f
  Write-Output "DONE w=$w crf=$c size=$((Get-Item $f).Length)"
}
Write-Output "LADDER_ALL_DONE"
