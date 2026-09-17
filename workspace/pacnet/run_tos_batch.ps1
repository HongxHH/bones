# TOS 多轨迹串行实验：09-14 / 09-21，各跑 pacnet 与 imdn_always
$ErrorActionPreference = "Continue"
Set-Location "E:\dev\Code\Python\bones\player"
$py = "E:\dev\env\miniconda3\envs\SR\python.exe"
$ws = "E:\dev\Code\Python\bones\workspace\pacnet"
$traces = @(
    "report.2010-09-14_1038CEST",
    "report.2010-09-21_1001CEST"
)
$methods = @("pacnet", "imdn_always")
$batchLog = "$ws\tos_batch.log"
"批次启动 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $batchLog -Encoding UTF8

foreach ($tr in $traces) {
    foreach ($m in $methods) {
        $tag = "${m}_${tr}"
        $out = "$ws\tos_$tag.log"
        $errf = "$ws\tos_$tag.err"
        "[$(Get-Date -Format HH:mm:ss)] START $tag" | Out-File $batchLog -Append -Encoding UTF8
        $args = @("play.py","--config","tos.yaml","--input","data/tears_of_steel/tos.mpd",
                  "--mod_enhancer",$m,"--trace","data/3Glogs/$tr.json")
        $p = Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory "E:\dev\Code\Python\bones\player" `
              -RedirectStandardOutput $out -RedirectStandardError $errf -WindowStyle Hidden -PassThru
        $p.WaitForExit()
        "[$(Get-Date -Format HH:mm:ss)] DONE  $tag exit=$($p.ExitCode)" | Out-File $batchLog -Append -Encoding UTF8
    }
}
"批次全部完成 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $batchLog -Append -Encoding UTF8
