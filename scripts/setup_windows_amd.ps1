<#
.SYNOPSIS
  One-time setup for training on an AMD Radeon RX 9070 XT (RDNA4, gfx1201) on Windows.

.DESCRIPTION
  Creates .venv, installs a ROCm build of PyTorch for Windows (AMD "TheRock" gfx120X
  wheels), installs requirements.txt, fetches Stockfish, runs a GPU sanity check, and
  writes .env.ps1.  ROCm PyTorch exposes the GPU through the torch.cuda API, so the
  project's device name is "cuda".

  Usage (from the repo root, in PowerShell):
    powershell -ExecutionPolicy Bypass -File scripts\setup_windows_amd.ps1
    . .\.env.ps1
    .\.venv\Scripts\Activate.ps1

  Prerequisites: Windows 11, a recent AMD Adrenalin driver (25.9.1 or newer), and
  64-bit Python 3.11-3.13 on PATH.

.PARAMETER IndexUrl
  Override the PyTorch wheel index (default: AMD gfx120X-all, covers 9070 / 9070 XT).

.PARAMETER SkipStockfish
  Do not download Stockfish (set STOCKFISH_PATH yourself).

.PARAMETER SkipLc0
  Do not download lc0 + a network into lc0\ (needed only for --teacher lc0 labeling).
#>
[CmdletBinding()]
param(
  [string]$IndexUrl = "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
  [switch]$SkipStockfish,
  [switch]$SkipLc0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
Write-Host "==> Project root: $Root"

function Invoke-Checked {
  # Run a native command and fail loudly on a non-zero exit code.
  param([string]$Exe, [string[]]$CmdArgs)
  & $Exe @CmdArgs
  if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Exe $($CmdArgs -join ' ')" }
}

# --- 1. GPU / driver check ---
$gpus = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" }
if ($gpus) {
  foreach ($g in $gpus) { Write-Host "==> GPU: $($g.Name)  driver $($g.DriverVersion)" }
  if (-not ($gpus | Where-Object { $_.Name -match "9070|9060" })) {
    Write-Warning "No RX 9070/9060 found; the gfx120X wheels only support RDNA4 cards."
  }
} else {
  Write-Warning "No AMD GPU detected by Windows. Continuing anyway."
}

# --- 2. Python ---
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "Python not found on PATH. Install 64-bit Python 3.11-3.13." }
$ver = (& python -c "import sys;print('%d.%d'%sys.version_info[:2])").Trim()
if ($ver -notin @("3.11", "3.12", "3.13")) {
  throw "Python $ver is not supported by the ROCm Windows wheels (need 3.11-3.13)."
}
Write-Host "==> Python $ver"

# --- 3. venv ---
$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  Write-Host "==> Creating .venv..."
  Invoke-Checked python @("-m", "venv", ".venv")
}
Invoke-Checked $venvPy @("-m", "pip", "install", "--upgrade", "pip")

# --- 4. PyTorch (ROCm) ---
# Native stderr becomes a terminating error under "Stop" in PS 5.1, so relax it for probes.
$ErrorActionPreference = "Continue"
$null = & $venvPy -c "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)" 2>&1
$hasRocm = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"

if ($hasRocm) {
  Write-Host "==> ROCm torch already installed"
} else {
  Write-Host "==> Installing ROCm PyTorch from $IndexUrl (large download)..."
  $ErrorActionPreference = "Continue"
  $null = & $venvPy -m pip uninstall -y torch torchvision torchaudio 2>&1
  $ErrorActionPreference = "Stop"
  # Chess needs only torch. --pre because these are AMD nightly/dev builds.
  Invoke-Checked $venvPy @("-m", "pip", "install", "--pre", "torch", "--index-url", $IndexUrl)
}

# --- 5. Project requirements ---
Invoke-Checked $venvPy @("-m", "pip", "install", "-r", "requirements.txt")

# --- 6. Stockfish ---
$sfDir = Join-Path $Root "stockfish"
$stockfish = $env:STOCKFISH_PATH
if (-not $stockfish -or -not (Test-Path $stockfish)) {
  $found = Get-Command stockfish -ErrorAction SilentlyContinue
  if ($found) { $stockfish = $found.Source }
  else { $stockfish = (Get-ChildItem $sfDir -Recurse -Filter "stockfish*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1).FullName }
}
if (-not $stockfish -and -not $SkipStockfish) {
  Write-Host "==> Downloading Stockfish..."
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest" -Headers @{ "User-Agent" = "chess-setup" }
    $asset = $rel.assets | Where-Object { $_.name -match "windows-x86-64-avx2\.(zip|tar)$" } | Select-Object -First 1
    if (-not $asset) { throw "no windows-x86-64-avx2 asset in release $($rel.tag_name)" }
    $archive = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archive -UseBasicParsing
    New-Item -ItemType Directory -Force $sfDir | Out-Null
    if ($archive.EndsWith(".zip")) { Expand-Archive -Path $archive -DestinationPath $sfDir -Force }
    else { Invoke-Checked tar @("-xf", $archive, "-C", $sfDir) }
    $stockfish = (Get-ChildItem $sfDir -Recurse -Filter "stockfish*.exe" | Select-Object -First 1).FullName
  } catch {
    Write-Warning "Stockfish download failed ($_). Install it manually and set STOCKFISH_PATH."
  }
}
if ($stockfish) { Write-Host "==> Stockfish: $stockfish" }
else { Write-Warning "Stockfish not set up; needed for labeling and Elo evaluation." }

# --- 6b. lc0 (GPU teacher for labeling; DirectML build runs on the AMD GPU) ---
$lc0Dir = Join-Path $Root "lc0"
$lc0Exe = Join-Path $lc0Dir "lc0.exe"
$lc0Net = Join-Path $lc0Dir "net.pb.gz"
if (-not $SkipLc0) {
  try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    New-Item -ItemType Directory -Force $lc0Dir | Out-Null
    if (-not (Test-Path $lc0Exe)) {
      Write-Host "==> Downloading lc0 (DirectML build)..."
      $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/LeelaChessZero/lc0/releases/latest" -Headers @{ "User-Agent" = "chess-setup" }
      $asset = $rel.assets | Where-Object { $_.name -match "windows-onnx-dml\.zip$" } | Select-Object -First 1
      if (-not $asset) { throw "no windows-onnx-dml asset in release $($rel.tag_name)" }
      $zip = Join-Path $env:TEMP $asset.name
      Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -UseBasicParsing
      Expand-Archive -Path $zip -DestinationPath $lc0Dir -Force
    }
    if (-not (Test-Path $lc0Net)) {
      # Distilled 256x10 net: strong, and ~2x faster than 512x15 on DirectML (see README).
      Write-Host "==> Downloading lc0 network (37 MB)..."
      Invoke-WebRequest -Uri "https://storage.lczero.org/files/networks-contrib/t1-256x10-distilled-swa-2432500.pb.gz" -OutFile $lc0Net -UseBasicParsing
    }
    Write-Host "==> lc0: $lc0Exe"
  } catch {
    Write-Warning "lc0 setup failed ($_). Labeling with --teacher lc0 will not work until it is installed."
  }
}

# --- 7. GPU sanity check ---
Write-Host "==> Verifying GPU..."
$check = @'
import time, torch
print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
if not torch.cuda.is_available():
    raise SystemExit("FAILED: torch.cuda.is_available() is False (no ROCm GPU visible)")
print("device", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20): y = x @ x
torch.cuda.synchronize()
print("fp16 matmul ok, %.1f TFLOPS" % (20 * 2 * 2048**3 / (time.time() - t) / 1e12))
c = torch.nn.Conv2d(64, 64, 3, padding=1).cuda()
c(torch.randn(8, 64, 8, 8, device="cuda")).sum().backward()
print("conv2d ok")
'@
# Passing a multi-line -c script breaks on PS 5.1 (quotes get stripped), so use a file.
$checkFile = Join-Path $env:TEMP "gpu_check.py"
Set-Content -Path $checkFile -Value $check -Encoding ascii
& $venvPy $checkFile
if ($LASTEXITCODE -ne 0) {
  throw "GPU check failed. Update the AMD Adrenalin driver, then re-run. If the wheel index moved, pass -IndexUrl."
}

# --- 8. Env file ---
$envFile = Join-Path $Root ".env.ps1"
$lines = @(
  "# Dot-source before running:  . .\.env.ps1",
  "`$env:CHESSAI_DEVICE = 'cuda'   # ROCm torch exposes the AMD GPU as 'cuda'",
  "`$env:CHESSAI_DATA = '$Root\data_d16'",
  "`$env:LC0_PATH = '$lc0Exe'; `$env:LC0_WEIGHTS = '$lc0Net'",
  "`$env:CHESSAI_COMPILE = '0'     # torch.compile/inductor is unreliable on Windows ROCm",
  "`$env:OPENBLAS_NUM_THREADS = '1'; `$env:OMP_NUM_THREADS = '1'; `$env:MKL_NUM_THREADS = '1'"
)
if ($stockfish) { $lines += "`$env:STOCKFISH_PATH = '$stockfish'" }
$lines | Set-Content -Path $envFile -Encoding utf8
Write-Host "==> Wrote $envFile"

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  . .\.env.ps1"
Write-Host "  python -m cli supervised --epochs 24 --blocks 20 --channels 256 --out supervised_big.pt"
Write-Host "  python -m cli selfplay --init models\supervised_big.pt --sims 400 --workers 8 --selfplay-device cpu"
