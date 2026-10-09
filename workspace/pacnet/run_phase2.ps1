# 阶段二：TOS 算力紧张(k≈1.5)下三组真实播放器对比
# 组1=PAC全e3死磕 / 组2=IMDN独立l2基线 / 组3=调好阈值的halt
$ErrorActionPreference = "Continue"
$py = "E:\dev\env\miniconda3\envs\SR\python.exe"
Set-Location "E:\dev\Code\Python\bones\player"
$trace = "data/3Glogs/report.2010-09-14_1038CEST.json"

# 公共：每帧额外17ms GPU负载（模拟外部占用，k≈1.5）
$env:PACNET_EXTRA_MS = "17"

Write-Host "=== [1/3] PAC 全e3 (PACNET_FORCE_EXIT=2) ==="
Remove-Item Env:PACNET_FORCE_EXIT -ErrorAction SilentlyContinue
Remove-Item Env:PACNET_E3_THRESH -ErrorAction SilentlyContinue
Remove-Item Env:PACNET_E2_THRESH -ErrorAction SilentlyContinue
$env:PACNET_FORCE_EXIT = "2"
& $py play.py --config tos.yaml --input data/tears_of_steel/tos.mpd --mod_enhancer pacnet --trace $trace *> "..\workspace\pacnet\p2_force_e3.log"
Write-Host "=== [1/3] done, exit=$LASTEXITCODE ==="

Write-Host "=== [2/3] IMDN 独立基线 ==="
Remove-Item Env:PACNET_FORCE_EXIT -ErrorAction SilentlyContinue
& $py play.py --config tos.yaml --input data/tears_of_steel/tos.mpd --mod_enhancer imdn_always --trace $trace *> "..\workspace\pacnet\p2_imdn_baseline.log"
Write-Host "=== [2/3] done, exit=$LASTEXITCODE ==="

Write-Host "=== [3/3] PAC 调好阈值 halt (e3=1.2, e2=1.6) ==="
$env:PACNET_E3_THRESH = "1.2"
$env:PACNET_E2_THRESH = "1.6"
& $py play.py --config tos.yaml --input data/tears_of_steel/tos.mpd --mod_enhancer pacnet --trace $trace *> "..\workspace\pacnet\p2_halt_tuned.log"
Write-Host "=== [3/3] done, exit=$LASTEXITCODE ==="

# 清理
Remove-Item Env:PACNET_EXTRA_MS, Env:PACNET_E3_THRESH, Env:PACNET_E2_THRESH -ErrorAction SilentlyContinue
Write-Host "=== ALL DONE ==="
