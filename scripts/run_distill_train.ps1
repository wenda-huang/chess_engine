# Distillation training: fine-tune a self-play checkpoint on the lc0-labeled data from
# run_distill_label.ps1, restarting on crash/hang from the last epoch snapshot. Usage (from repo root):
#   powershell -File scripts\run_distill_train.ps1 [-Init models\best_small_it197.pt] [-Epochs 8]
param(
  [string]$Init = "models\best_small_it197.pt",
  [string]$DataDir = "data_distill",
  [string]$Out = "distill_small.pt",
  [int]$Epochs = 8,
  [double]$Lr = 5e-4,
  [double]$ResumeLr = 2.5e-4,
  [int]$BatchSize = 1024,
  [int]$Patience = 2,
  [int]$StallSec = 900,
  [int]$MaxRestarts = 15
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CHESSAI_DEVICE = "cuda"; $env:TORCH_BLAS_PREFER_HIPBLASLT = "0"
$env:OMP_NUM_THREADS = "2"; $env:MKL_NUM_THREADS = "2"
$env:CHESSAI_DATA = $DataDir
$py = ".\.venv\Scripts\python.exe"; $log = "logs\distill_pipeline.out"; $jsonl = "logs\supervised.jsonl"
$ckpt = "models\" + [IO.Path]::GetFileNameWithoutExtension($Out) + "_ckpt.pt"
function Say($m) { $line = "$(Get-Date -Format s) $m"; $line | Out-File $log -Append -Encoding utf8; Write-Host $line -ForegroundColor Magenta }
function Kill-Tree($id) {
  Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $id } | ForEach-Object { Kill-Tree $_.ProcessId }
  Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}
function Epochs-Done() {  # epochs finished by this run, from the crash-recovery snapshot
  if (-not (Test-Path $ckpt)) { return 0 }
  $e = & $py -c "import torch;print(torch.load(r'$ckpt',map_location='cpu',weights_only=False)['meta']['epoch'])"
  return [int]$e - $script:initEpoch
}
# Epoch counter stored in the init checkpoint (meta epochs are cumulative across runs).
$initEpoch = [int](& $py -c "import torch;print(torch.load(r'$Init',map_location='cpu',weights_only=False).get('meta',{}).get('epoch',-1))")

for ($try = 0; $try -lt $MaxRestarts; $try++) {
  $done = Epochs-Done
  if ($done -ge $Epochs) { break }
  if ($done -le 0) { $a = @("--resume", $Init, "--epochs", "$Epochs", "--lr", "$Lr") }
  else { $a = @("--resume", $ckpt, "--epochs", "$($Epochs - $done)", "--lr", "$ResumeLr"); Say "resuming from $ckpt, $done epochs done" }
  $argList = @("-m", "cli", "supervised") + $a + @("--batch-size", "$BatchSize", "--patience", "$Patience", "--out", $Out)
  Say "distill train attempt $try : $($a -join ' ')"
  $p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru `
       -RedirectStandardOutput "logs\distill_train.out" -RedirectStandardError "logs\distill_train.err"
  $null = $p.Handle; $start = Get-Date; $lastShown = ""
  while (-not $p.WaitForExit(30000)) {
    $tail = Get-Content $jsonl -Tail 1 -ErrorAction SilentlyContinue
    if ($tail -and $tail -ne $lastShown) { Write-Host "  $tail" -ForegroundColor Cyan; $lastShown = $tail }
    $idle = ((Get-Date) - (Get-Item $jsonl).LastWriteTime).TotalSeconds
    if ($idle -gt $StallSec -and ((Get-Date) - $start).TotalSeconds -gt $StallSec) {
      Say "silent for $([int]$idle) s; killing (hang?)"; Kill-Tree $p.Id; break
    }
  }
  $code = if ($p.HasExited) { $p.ExitCode } else { -1 }
  Say "distill train attempt $try exit code $code"
  # Early stopping and normal completion both exit 0; a crash leaves a snapshot to resume from.
  if ($code -eq 0) { Say "DISTILL TRAINING DONE -> models\$Out"; exit 0 }
  Start-Sleep -Seconds 60
}
Say "DISTILL TRAINING GAVE UP"; exit 1
