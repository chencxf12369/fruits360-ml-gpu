<# 
.SYNOPSIS
  Cross-platform friendly Windows setup for fruits360 project.
  Creates .venv, installs dependencies, and optionally TensorFlow GPU.

.PARAMETER Python
  Preferred Python major.minor (e.g., "3.11"). Uses py launcher when possible.

.PARAMETER Gpu
  Attempt to install a GPU-enabled TensorFlow via `pip install "tensorflow[and-cuda]"`.
  If that fails, falls back to CPU-only `tensorflow`.

.PARAMETER Reinstall
  If specified, removes existing .venv before creating a new one.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Python 3.11 -Gpu
#>

[CmdletBinding()]
param(
  [string]$Python = "3.11",
  [switch]$Gpu,
  [switch]$Reinstall
)

$ErrorActionPreference = "Stop"

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-OK($msg)    { Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# --- repo-relative paths ---
$RepoRoot   = (Resolve-Path -Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath   = Join-Path $RepoRoot ".venv"
$ActivatePs = Join-Path $VenvPath "Scripts\Activate.ps1"
$ReqFile    = Join-Path $RepoRoot "requirements.txt"

# --- choose python launcher / executable (relative friendly) ---
function Resolve-Python() {
  # Prefer 'py -3.11' style; fall back to py, then python in PATH
  $candidates = @(
    "py -$($Python)",
    "py",
    "python"
  )
  foreach ($c in $candidates) {
    try {
      $version = & $c -c "import sys; print(sys.version.split()[0])" 2>$null
      if ($LASTEXITCODE -eq 0 -and $version) {
        Write-Info "Using Python via: $c (version $version)"
        return $c
      }
    } catch { }
  }
  throw "No suitable Python interpreter found. Install Python or the 'py' launcher."
}

$PY = Resolve-Python

# --- optional clean reinstall ---
if ($Reinstall -and (Test-Path $VenvPath)) {
  Write-Info "Removing existing venv at .venv (Reinstall requested)..."
  Remove-Item -Recurse -Force $VenvPath
}

# --- create venv if missing ---
if (-not (Test-Path $VenvPath)) {
  Write-Info "Creating virtual environment at .venv ..."
  & $PY -m venv "$VenvPath"
} else {
  Write-Info "Virtual environment already exists (.venv)."
}

# --- activate venv for the rest of this script ---
$activateCmd = ". `"$ActivatePs`""
Write-Info "Activating venv: $activateCmd"
Invoke-Expression $activateCmd

# --- upgrade pip toolchain ---
Write-Info "Upgrading pip / setuptools / wheel ..."
pip install -q --upgrade pip setuptools wheel
Write-OK "pip tools upgraded."

# --- install TensorFlow (GPU optional) ---
function Install-TensorFlow() {
  if ($Gpu) {
    Write-Info "Attempting GPU-enabled TensorFlow: pip install \"tensorflow[and-cuda]\" ..."
    try {
      pip install -q "tensorflow[and-cuda]"
      Write-OK "Installed tensorflow[and-cuda]."
      return
    } catch {
      Write-Warn "GPU TF install failed; falling back to CPU tensorflow."
    }
  }
  Write-Info "Installing CPU TensorFlow ..."
  pip install -q tensorflow
  Write-OK "Installed CPU tensorflow."
}

# If requirements.txt exists and already pins TensorFlow, respect it and skip explicit TF install
$requirementsPinsTF = $false
if (Test-Path $ReqFile) {
  try {
    $reqText = Get-Content -Raw -Encoding UTF8 $ReqFile
    if ($reqText -match '^\s*tensorflow' -or $reqText -match '^\s*tensorflow-macos' -or $reqText -match 'tensorflow\[and-cuda\]') {
      $requirementsPinsTF = $true
    }
  } catch { }
}

if (-not $requirementsPinsTF) {
  Install-TensorFlow
} else {
  Write-Info "requirements.txt appears to pin TensorFlow; will install from requirements."
}

# --- install from requirements.txt if present ---
if (Test-Path $ReqFile) {
  Write-Info "Installing dependencies from requirements.txt ..."
  pip install -q -r "$ReqFile"
  Write-OK "requirements.txt installed."
} else {
  Write-Warn "No requirements.txt found; continuing."
}

# --- install project in editable mode ---
Write-Info "Installing project in editable mode (pip install -e .) ..."
pip install -q -e "$RepoRoot"
Write-OK "Project installed."

# --- create artifacts directories (relative) ---
$Artifacts = Join-Path $RepoRoot "artifacts"
$TbLogs    = Join-Path $Artifacts "tb_logs"
$Ckpts     = Join-Path $Artifacts "checkpoints"
New-Item -ItemType Directory -Force -Path $Artifacts, $TbLogs, $Ckpts | Out-Null

# --- detect or prompt for fruits360 dataset --- 
function Resolve-Fruits360Dataset() {
  $defaultPath = Join-Path $env:USERPROFILE "data\Fruit-Images-Dataset"
  $datasetPath = $null

  Write-Info "Checking for fruits360 dataset..."

  if (Test-Path $defaultPath) {
    Write-OK "Found dataset at default path: $defaultPath"
    return $defaultPath
  }

  Write-Warn "Dataset not found at default path: $defaultPath"
  $response = Read-Host "Would you like to download the fruits360 dataset to this location? (y/n)"
  
  if ($response -eq "y") {
    $datasetUrl = "https://github.com/Horea94/Fruit-Images-Dataset/archive/master.zip"
    $zipPath = Join-Path $env:TEMP "fruits360.zip"
    Write-Info "Downloading dataset from GitHub..."
    Invoke-WebRequest -Uri $datasetUrl -OutFile $zipPath
    Write-Info "Extracting dataset..."
    Expand-Archive -Path $zipPath -DestinationPath (Split-Path $defaultPath -Parent) -Force
    Rename-Item -Path (Join-Path (Split-Path $defaultPath -Parent) "Fruit-Images-Dataset-master") -NewName "fruits360"
    Remove-Item $zipPath
    Write-OK "Dataset downloaded and extracted to: $defaultPath"
    return $defaultPath
  } else {
    $customPath = Read-Host "Enter custom path to fruits360 dataset"
    if (Test-Path $customPath) {
      Write-OK "Found dataset at: $customPath"
      return $customPath
    } else {
      Write-Err "Custom path not found. Please download the dataset manually."
      return $null
    }
  }
}

$Fruits360Path = Resolve-Fruits360Dataset


# --- quick sanity check: import package & print a summary if available ---
try {
  Write-Info "Verifying package import and config summary ..."
  $code = @"
from fruits360 import config
print(config.summary())
"@
  & $PY -c $code
  Write-OK "Package import check passed."
} catch {
  Write-Warn "Could not import fruits360 or print summary (this may be okay if paths differ)."
}

Write-OK "Setup complete. To activate the environment in this shell:"
Write-Host "  . $ActivatePs" -ForegroundColor Yellow
