# halt 前瞻安全系数扫描：K=1.5 乘法负载、TOS、headless，halt 前瞻（无 FORCE_EXIT）。
# 两条 trace：0921（最差网，卡顿短板）+ 0914（居中）；E3/E2 统一安全系数扫 1.1/1.2/1.3。
# 系数 1.0 基线已有数据（0921=121806、0914=103522），不重跑。
# 用法：powershell -ExecutionPolicy Bypass -File run_sweep_thresh.ps1
$ErrorActionPreference = "Continue"
$py = "E:\dev\env\miniconda3\envs\SR\python.exe"
Set-Location "E:\dev\Code\Python\bones\player"
$out = "E:\dev\Code\Python\bones\workspace\pacnet\sweep_thresh"
New-Item -ItemType Directory -Force $out | Out-Null
$progress = Join-Path $out "progress.txt"
"START $(Get-Date -Format o)" | Out-File $progress

$traces = @(
    @{ tag = "0921"; name = "report.2010-09-21_1001CEST" },
    @{ tag = "0914"; name = "report.2010-09-14_1038CEST" }
)
$coefs = @("1.1", "1.2", "1.3")

function Kill-Python {
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
}
function Latest-Json($nes) {
    $f = Get-ChildItem "output\data\bones_$nes`*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($f) { return $f.Name } else { return "MISSING" }
}

Kill-Python
foreach ($t in $traces) {
    $tag = $t.tag; $name = $t.name
    foreach ($c in $coefs) {
        $cs = $c.Replace(".", "")   # 1.1 -> 11
        $strategy = "halt_s$cs"
        $log = Join-Path $out "${tag}_${strategy}.log"
        $err = Join-Path $out "${tag}_${strategy}.err"
        Kill-Python
        $env:PACNET_BURN_K = "1.5"
        Remove-Item Env:\PACNET_FORCE_EXIT -ErrorAction SilentlyContinue
        $env:PACNET_E3_THRESH = $c
        $env:PACNET_E2_THRESH = $c
        "RUN  $tag $strategy (E3=E2=$c) start $(Get-Date -Format o)" | Out-File $progress -Append
        & $py play.py --config tos.yaml --input data/tears_of_steel/tos.mpd `
            --mod_enhancer pacnet --trace "data/3Glogs/$name.json" 1> $log 2> $err
        $code = $LASTEXITCODE
        $j = Latest-Json "pacnet"
        "DONE $tag $strategy exit=$code json=$j $(Get-Date -Format o)" | Out-File $progress -Append
    }
}
"ALL_DONE $(Get-Date -Format o)" | Out-File $progress -Append
