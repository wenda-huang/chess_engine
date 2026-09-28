# Distillation data: dump positions from GPU self-play, then label them with lc0 (GPU),
# restarting on crash/hang. Both steps resume where they left off. Usage (from repo root):
#   powershell -File scripts\run_distill_label.ps1 [-Checkpoint models\best_small_it197.pt] [-Samples 3500000]
param(
  [string]$Checkpoint = "models\best_small_it197.pt",
  [string]$DataDir = "data_distill",
  [int]$Fens = 1300000,          # unique self-play start positions
  [int]$Samples = 3500000,       # labeled samples to stop at (start position + lc0 continuation)
  [int]$ChainLen = 3,
  [int]$Nodes = 400,
  [int]$Workers = 6,
  [int]$Sims = 128,
  [int]$MaxRestarts = 15
)
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CHESSAI_DEVICE = "cuda"; $env:TORCH_BLAS_PREFER_HIPBLASLT = "0"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$env:CHESSAI_DATA = $DataDir
$py = ".\.venv\Scripts\python.exe"; $log = "logs\distill_pipeline.out"; $fensPath = "$DataDir\fens.txt"
New-Item -ItemType Directory -Force $DataDir | Out-Null
function Say($m) { $line = "$(Get-Date -Format s) $m"; $line | Out-File $log -Append -Encoding utf8; Write-Host $line -ForegroundColor Magenta }
function Kill-Tree($id) {
  Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $id } | ForEach-Object { Kill-Tree $_.ProcessId }
  Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}
function Run-Step($name, $argList, $outFile, $watchFile, $stallMin) {
  for ($try = 0; $try -lt $MaxRestarts; $try++) {
    Say "$name attempt $try"
    $p = Start-Process -FilePath $py -ArgumentList $argList -NoNewWindow -PassThru -RedirectStandardOutput $outFile -RedirectStandardError "$outFile.err"
    $null = $p.Handle
    $start = Get-Date; $shown = 0
    while (-not $p.WaitForExit(30000)) {
      if (Test-Path $outFile) {  # echo new progress lines to this window
        $lines = @(Get-Content $outFile -ErrorAction SilentlyContinue)
        if ($lines.Count -gt $shown) { $lines[$shown..($lines.Count - 1)] | Select-Object -Last 2 | ForEach-Object { Write-Host "  $_" -ForegroundColor Cyan }; $shown = $lines.Count }
      }
      $w = if (Test-Path $watchFile) { $watchFile } else { $outFile }
      $idle = ((Get-Date) - (Get-Item $w -ErrorAction SilentlyContinue).LastWriteTime).TotalMinutes
      if ($idle -gt $stallMin -and ((Get-Date) - $start).TotalMinutes -gt $stallMin) {
        Say "$name silent for $([int]$idle) min; killing (hang?)"; Kill-Tree $p.Id; break
      }
    }
    Get-Process lc0 -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    $code = if ($p.HasExited) { $p.ExitCode } else { -1 }
    Say "$name attempt $try exit code $code"
    if ($code -eq 0) { return $true }
    Start-Sleep -Seconds 60  # give a crashed GPU driver time to recover
  }
  return $false
}

# ---- 1. self-play positions (GPU search) ----
$a1 = @("scripts\selfplay_fens.py", "--checkpoint", $Checkpoint, "--out", $fensPath, "--n", "$Fens", "--sims", "$Sims")
$have = 0; if (Test-Path $fensPath) { $have = ([System.IO.File]::ReadLines((Resolve-Path $fensPath)) | Measure-Object).Count }
if ($have -ge $Fens) { Say "fens: $have already present, skipping" }
elseif (-not (Run-Step "fens" $a1 "logs\distill_fens.out" $fensPath 30)) { Say "FEN GENERATION FAILED"; exit 1 }

# ---- 2. lc0 labels (GPU teacher) ----
$a2 = @("-m", "cli", "label", "--fens", $fensPath, "--teacher", "lc0", "--nodes", "$Nodes", "--chain-len", "$ChainLen",
        "--workers", "$Workers", "--target-samples", "$Samples")
if (-not (Run-Step "label" $a2 "logs\distill_label.out" "logs\label.jsonl" 20)) { Say "LABELING FAILED"; exit 1 }
Say "DISTILL DATA DONE"
