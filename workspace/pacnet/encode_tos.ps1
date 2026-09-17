$ErrorActionPreference = "Continue"
$work = "E:\dev\Code\Python\bones\data\tears_of_steel"
$input = "E:\dev\Data\tears_of_steel_1080p.mp4"

$jobs = @(
    @{name="360p"; w=640; h=268; br=800},
    @{name="480p"; w=854; h=356; br=1200},
    @{name="720p"; w=1280; h=532; br=2400},
    @{name="1080p"; w=1920; h=800; br=4800}
)

foreach ($j in $jobs) {
    $out = "$work\output_$($j.name).mp4"
    if (Test-Path $out) {
        Write-Output "SKIP $($j.name) (already exists)"
        continue
    }
    Write-Output "Encoding $($j.name) ($($j.w)x$($j.h)) @ $($j.br)k ..."
    $start = Get-Date
    & ffmpeg -y -i $input -an -c:v libx264 -preset fast -b:v "$($j.br)k" -maxrate "$($j.br*2)k" -bufsize "$($j.br*4)k" -vf "scale=$($j.w):$($j.h):flags=bicubic" $out 2>&1 | Select-Object -Last 2
    $elapsed = (Get-Date) - $start
    if (Test-Path $out) {
        $mb = [math]::Round((Get-Item $out).Length/1MB, 1)
        Write-Output "DONE $($j.name): ${mb}MB in $([math]::Round($elapsed.TotalSeconds,0))s"
    } else {
        Write-Output "FAILED $($j.name)"
    }
}
Write-Output "ALL ENCODING COMPLETE"
