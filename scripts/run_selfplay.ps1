# Self-play with crash/hang recovery. Restarts from the champion checkpoint and
# continues at the next unfinished iteration. Usage (from repo root):
#   powershell -File scripts\run_selfplay.ps1 [-Init models\small_10x128.pt] [-Iterations 40] [-Workers 10]
param(
  [string]$Init = "models\small_10x128.pt",   # only used for the very first launch
  [string]$Out = "selfplay_small.pt",
  [string]$Best = "best_small.pt",            # champion; resume point after a crash
  [int]$Iterations = 40,
  [int]$GamesPerIter = 256,                   # played concurrently on the GPU
  [int]$Sims = 400,
  [int]$ArenaSims = 0,                        # 0 = same as -Sims
  [int]$LrDecayFrom = 0,                      # iteration where the cosine LR decay begins
  [int]$LrDecayIters = 0,                     # decay length in iterations (0 = constant LR)
  [int]$TemperatureMoves = 18,                # plies sampled at temperature 1.0 (CLI default 25)
  [int]$ArenaGames = 400,                     # arena size (mirrored pairs; one fresh opening per pair)
  [double]$GateThreshold = 0.43,              # ~-50 Elo margin (was 0.5 = must prove strictly better)
  [switch]$NoGateSignificance,                # promote on score alone (no significance test)
  [string]$DataDir = "data_improved",         # labeled shards mixed into training (--sup-fraction)
  [int]$CpuMask = 0xF,                        # logical cores the run may use (0xF = 4 of 12); one runs the launcher thread
  [int]$StallSec = 1800,                      # no log activity for this long => hung
  [int]$MaxRestarts = 50,
  [string[]]$Extra = @()
)
$Extra = @($Extra | ForEach-Object { $_ -split "," } | Where-Object { $_ })  # -File passes comma lists as one string
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:CHESSAI_DATA = $DataDir; $env:CHESSAI_DEVICE = "cuda"; $env:TORCH_BLAS_PREFER_HIPBLASLT = "0"
$env:OMP_NUM_THREADS = "1"; $env:MKL_NUM_THREADS = "1"
$py = ".\.venv\Scripts\python.exe"; $jsonl = "logs\selfplay.jsonl"; $bestPath = "models\$Best"
$log = "logs\selfplay_pipeline.out"
function Say($m) { "$(Get-Date -Format s) $m" | Out-File $log -Append -Encoding utf8 }
function Kill-Tree($id) {
  Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $id } | ForEach-Object { Kill-Tree $_.ProcessId }
  Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}
function Next-Iter {
  if (-not (Test-Path $jsonl)) { return 0 }
  $m = Select-String -Path $jsonl -Pattern '"event": "iter_done", "iter": (\d+)' | ForEach-Object { [int]$_.Matches[0].Groups[1].Value } | Measure-Object -Maximum
  if ($m.Count -eq 0) { return 0 } else { return [int]$m.Maximum + 1 }
}

# Tail selfplay.jsonl in the background and turn iteration/promotion events into console
# lines, so progress is visible in this window (not just the log files).
$jsonlFull = Join-Path (Get-Location).Path $jsonl
$tailJob = Start-Job -ScriptBlock {
  param($path)
  while (-not (Test-Path $path)) { Start-Sleep -Seconds 1 }
  Get-Content -Path $path -Wait -Tail 0 | ForEach-Object {
    try { $evt = $_ | ConvertFrom-Json } catch { return }
    switch ($evt.event) {
      "start"         { "=== selfplay: iter $($evt.start_iter)-$($evt.iterations), games/iter=$($evt.games_per_iter), gate>=$($evt.gate_threshold) ===" }
      "iter_done"     { "[iter $($evt.iter)] iteration done ($($evt.seconds)s)" }
      "selfplay_done" { "[iter $($evt.iter)] selfplay: $($evt.seconds)s  W=$($evt.white) B=$($evt.black) D=$($evt.draw)  buffer=$($evt.buffer)" }
      "promote"       { "[iter $($evt.iter)] *** PROMOTED to champ v$($evt.champ_version)  score=$($evt.arena_score)  $($evt.wins)W-$($evt.draws)D-$($evt.losses)L ***" }
      "arena_reject"  { "[iter $($evt.iter)] arena: not promoted -- $($evt.gate_reason)  score=$($evt.arena_score)" }
      "done"          { "=== SELFPLAY RUN COMPLETE: champ v$($evt.champ_version), checkpoint=$($evt.checkpoint) ===" }
      default         { }
    }
  }
} -ArgumentList $jsonlFull
function Flush-Tail {
  Receive-Job -Job $tailJob -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_ -match '\*\*\*') { Write-Host $_ -ForegroundColor Green }
    elseif ($_ -match 'not promoted') { Write-Host $_ -ForegroundColor Yellow }
    elseif ($_ -match '^===') { Write-Host $_ -ForegroundColor Magenta }
    else { Write-Host $_ -ForegroundColor Cyan }
  }
}

$exitCode = 1
try {
for ($try = 0; $try -lt $MaxRestarts; $try++) {
  $start = if ($try -eq 0 -and -not (Test-Path $jsonl)) { 0 } else { Next-Iter }
  if ($start -ge $Iterations) { Say "SELFPLAY DONE"; $exitCode = 0; break }
  $init = if (Test-Path $bestPath) { $bestPath } else { $Init }
  $a = @("-m","cli","selfplay","--engine","gpu","--init",$init,"--out",$Out,"--best-out",$Best,"--iterations","$Iterations","--start-iter","$start",
         "--games-per-iter","$GamesPerIter","--sims","$Sims","--temperature-moves","$TemperatureMoves","--sup-fraction","0.5",
         "--arena-every","3","--arena-games","$ArenaGames","--gate-min-games","$ArenaGames",
         "--gate-threshold","$GateThreshold") + $Extra
  if ($NoGateSignificance) { $a += "--no-gate-significance" }
  if ($ArenaSims -gt 0) { $a += @("--arena-sims","$ArenaSims") }
  if ($LrDecayIters -gt 0) { $a += @("--lr-decay-from","$LrDecayFrom","--lr-decay-iters","$LrDecayIters") }
  Say "attempt $try : init=$init start_iter=$start"
  $p = Start-Process -FilePath $py -ArgumentList $a -NoNewWindow -PassThru -RedirectStandardOutput logs\selfplay.out -RedirectStandardError logs\selfplay.err
  $p.PriorityClass = "BelowNormal"; $p.ProcessorAffinity = $CpuMask  # set before the worker pool spawns so children inherit
  Start-Sleep 3  # the venv launcher spawns the real python; limit it too, before it starts the worker pool
  Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $p.Id } | ForEach-Object { $c = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue; if ($c) { $c.PriorityClass = "BelowNormal"; $c.ProcessorAffinity = $CpuMask } }
  $null = $p.Handle  # cache the handle so ExitCode is readable after exit
  $stalled = $false; $finished = $false
  while (-not $p.WaitForExit(30000)) {
    Flush-Tail
    if (Test-Path $jsonl) { $idle = ((Get-Date) - (Get-Item $jsonl).LastWriteTime).TotalSeconds } else { $idle = 0 }
    # Finished but the process lingers (exit hang): treat as done after a grace period.
    if ($idle -gt 90 -and (Select-String -Path $jsonl -Pattern '"event": "done"' -Quiet)) { Say "run finished but process lingers; killing"; Kill-Tree $p.Id; $p.WaitForExit(); $finished = $true; break }
    if ($idle -gt $StallSec) { Say "no log activity for $StallSec s; killing"; Kill-Tree $p.Id; $stalled = $true; break }
  }
  Flush-Tail
  $code = if ($finished) { 0 } elseif ($stalled) { -1 } else { $p.ExitCode }
  Say "attempt $try exit code $code"
  if ($code -eq 0) { Say "SELFPLAY DONE"; $exitCode = 0; break }
  Start-Sleep 30
}
if ($exitCode -ne 0) { Say "SELFPLAY GAVE UP" }
} finally {
  Flush-Tail
  Stop-Job -Job $tailJob -ErrorAction SilentlyContinue | Out-Null
  Remove-Job -Job $tailJob -Force -ErrorAction SilentlyContinue | Out-Null
}
exit $exitCode
