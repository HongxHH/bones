# 现代分辨率超分探针：从原生 4K60 BBB 生成 1080p CRF 阶梯
# 用法：powershell -ExecutionPolicy Bypass -File gen_1080p_ladder.ps1
# 注意：ffmpeg 命令必须单行（后台 -File 模式下反引号续行会被吞）。
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$ladder = Join-Path $here "ladder"
New-Item -ItemType Directory -Force -Path $ladder | Out-Null
$src = "E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4"
foreach ($crf in 18,22,26,30,34) {
    $out = Join-Path $ladder "lr1080_crf$crf.mp4"
    if (Test-Path $out) { Write-Output "EXISTS crf$crf"; continue }
    ffmpeg -y -v error -i $src -vf "scale=1920:1080:flags=bicubic" -c:v libx264 -preset veryfast -crf $crf -g 240 -keyint_min 240 -sc_threshold 0 -pix_fmt yuv420p -an $out
    if ($LASTEXITCODE -ne 0) { Write-Output "FAIL crf$crf rc=$LASTEXITCODE"; exit 1 }
    Write-Output "DONE crf$crf"
}
Write-Output "ALL LADDER DONE"
