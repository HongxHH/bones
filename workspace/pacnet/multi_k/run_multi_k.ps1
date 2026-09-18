# 阶段二多轨迹验证：K=1.5 下每条 trace 跑 halt前瞻 / force e3 / IMDN 三组，串行。
# 用法：powershell -ExecutionPolicy Bypass -File run_multi_k.ps1
$ErrorActionPreference = "Continue"
$py = "E:\dev\env\miniconda3\envs\SR\python.exe"
Set-Location "E:\dev\Code\Python\bones\player"
$out = "E:\dev\Code\Python\bones\workspace\pacnet\multi_k"
New-Item -ItemType Directory -Force $out | Out-Null
$progress = Join-Path $out "progress.txt"
"START $(Get-Date -Format o)" | Out-File $progress

$traces = @(
    @{ tag = "0913"; name = "report.2010-09-13_1003CEST" },
    @{ tag = "0921"; name = "report.2010-09-21_1001CEST" }
)

function Kill-Python {
    Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3
}
function Latest-Json($nes) {
    $f = Get-ChildItem "output\data\bones_$nes`*" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
    if ($f) { return $f.Name } else { return "MISSING" }
}
function Run-One($tag, $strategy, $mod, $trace) {
    $log = Join-Path $out "${tag}_${strategy}.log"
    $err = Join-Path $out "${tag}_${strategy}.err"
    "RUN  $tag $strategy start $(Get-Date -Format o)" | Out-File $progress -Append
    & $py play.py --config tos.yaml --input data/tears_of_steel/tos.mpd `
        --mod_enhancer $mod --trace "data/3Glogs/$trace.json" 1> $log 2> $err
    $code = $LASTEXITCODE
    $nes = if ($mod -eq "pacnet") { "pacnet" } else { "imdn_always" }
    $j = Latest-Json $nes
    "DONE $tag $strategy exit=$code json=$j $(Get-Date -Format o)" | Out-File $progress -Append
}

foreach ($t in $traces) {
    $tag = $t.tag; $name = $t.name

    # 1) halt 前瞻（默认阈值 1.0/1.0，无 FORCE_EXIT）
    Kill-Python
    $env:PACNET_BURN_K = "1.5"
    Remove-Item Env:\PACNET_FORCE_EXIT -ErrorAction SilentlyContinue
    Remove-Item Env:\PACNET_E3_THRESH -ErrorAction SilentlyContinue
    Remove-Item Env:\PACNET_E2_THRESH -ErrorAction SilentlyContinue
    Run-One $tag "halt" "pacnet" $name

    # 2) 强制 e3
    Kill-Python
    $env:PACNET_BURN_K = "1.5"
    $env:PACNET_FORCE_EXIT = "2"
    Run-One $tag "force_e3" "pacnet" $name

    # 3) IMDN 基线（务必清掉 FORCE_EXIT）
    Kill-Python
    $env:PACNET_BURN_K = "1.5"
    Remove-Item Env:\PACNET_FORCE_EXIT -ErrorAction SilentlyContinue
    Run-One $tag "imdn" "imdn_always" $name
}

"ALL_DONE $(Get-Date -Format o)" | Out-File $progress -Append
