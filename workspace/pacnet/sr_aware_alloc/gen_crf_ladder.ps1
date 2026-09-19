# SR 感知率失真分配实验：TOS 426 宽 CRF 阶梯编码
# 强制每 96 帧（4s@24fps）一个 IDR、关闭 scenecut，保证按 96 帧切片等价于逐段独立编码
$src = "E:\dev\Data\tears_of_steel_1080p.mp4"
$out = "E:\dev\Code\Python\bones\workspace\pacnet\sr_aware_alloc\ladder"
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($c in @(22,26,30,34,38,42)) {
  $f = Join-Path $out "lr426_crf${c}.mp4"
  if (Test-Path $f) { Write-Output "EXISTS $f"; continue }
  Write-Output "ENC crf=$c ..."
  ffmpeg -y -v error -i $src -vf "scale=426:-2:flags=bicubic" -an -c:v libx264 -crf $c -pix_fmt yuv420p `
    -x264-params "keyint=96:min-keyint=96:scenecut=0" $f
  Write-Output "DONE crf=$c size=$((Get-Item $f).Length)"
}
Write-Output "CRF_LADDER_ALL_DONE"
