[CmdletBinding()]
param(
  [switch]$SkipMobile,
  [switch]$SkipDocker,
  [switch]$SkipNgrok,
  [int]$NgrokPort = 3001
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$paths = @{
  WebApp        = Join-Path $repoRoot 'ASL-Web'
  WebServer     = Join-Path $repoRoot 'ASL-Web\server'
  CallAppServer = Join-Path $repoRoot 'ASL-CallApp\server'
  MobileApp     = Join-Path $repoRoot 'ASL-MobileAPP'
}

$envFiles = @(
  @{
    Example = Join-Path $paths.WebServer '.env.example'
    Target  = Join-Path $paths.WebServer '.env'
  },
  @{
    Example = Join-Path $paths.CallAppServer '.env.example'
    Target  = Join-Path $paths.CallAppServer '.env'
  }
)

function Write-Step {
  param([string]$Message)
  Write-Host "==> $Message" -ForegroundColor Cyan
}

function Test-CommandExists {
  param([string]$Name)
  return $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Add-PathIfMissing {
  param([string]$Candidate)

  if ([string]::IsNullOrWhiteSpace($Candidate)) {
    return
  }

  if (-not (Test-Path -LiteralPath $Candidate)) {
    return
  }

  $pathEntries = ($env:PATH -split ';').Where({ $_ -ne '' })
  if ($pathEntries -notcontains $Candidate) {
    $env:PATH = "$Candidate;$env:PATH"
  }
}

function Update-SessionPathFromNpm {
  try {
    $npmPrefix = (npm config get prefix 2>$null).Trim()
    if (-not [string]::IsNullOrWhiteSpace($npmPrefix)) {
      Add-PathIfMissing -Candidate $npmPrefix
      Add-PathIfMissing -Candidate (Join-Path $npmPrefix 'bin')
    }
  }
  catch {
    Write-Host 'No se pudo actualizar PATH desde npm; se continuara con la verificacion actual.' -ForegroundColor Yellow
  }
}

function Install-Ngrok {
  Write-Step 'ngrok no esta disponible en PATH. Intentando instalarlo automaticamente'

  if (Test-CommandExists 'winget') {
    Write-Host 'Instalando ngrok con winget...' -ForegroundColor Yellow
    winget install --exact --id Ngrok.Ngrok --silent --accept-package-agreements --accept-source-agreements
    return
  }

  if (Test-CommandExists 'npm') {
    Write-Host 'Instalando ngrok con npm -g...' -ForegroundColor Yellow
    npm install --global ngrok
    Update-SessionPathFromNpm
    return
  }

  throw 'No se encontro un instalador compatible para ngrok. Instala winget o npm e intenta de nuevo.'
}

function Ensure-NgrokAvailable {
  if (Test-CommandExists 'ngrok') {
    return
  }

  Install-Ngrok

  if (-not (Test-CommandExists 'ngrok')) {
    Update-SessionPathFromNpm
  }

  if (-not (Test-CommandExists 'ngrok')) {
    throw 'ngrok no pudo instalarse o no quedo disponible en PATH.'
  }

  Write-Host 'ngrok instalado y disponible.' -ForegroundColor Green
}

function Ensure-PathExists {
  param([string]$Path, [string]$Label)
  if (-not (Test-Path -LiteralPath $Path)) {
    throw "$Label no existe: $Path"
  }
}

function Ensure-EnvFiles {
  foreach ($pair in $envFiles) {
    if (-not (Test-Path -LiteralPath $pair.Target)) {
      if (-not (Test-Path -LiteralPath $pair.Example)) {
        throw "Falta archivo de ejemplo para crear $($pair.Target)"
      }

      Copy-Item -LiteralPath $pair.Example -Destination $pair.Target
      Write-Host "Creado $($pair.Target) a partir de .env.example" -ForegroundColor Yellow
    }
  }
}

function Start-DevProcess {
  param(
    [string]$Name,
    [string]$WorkingDirectory,
    [string]$Command
  )

  $wrappedCommand = "`$Host.UI.RawUI.WindowTitle = '$Name'; Set-Location -LiteralPath '$WorkingDirectory'; Write-Host '[$Name] $Command' -ForegroundColor Green; $Command"

  Write-Step "Abriendo ventana para $Name"
  Start-Process -FilePath 'powershell' -ArgumentList @(
    '-NoExit',
    '-Command',
    $wrappedCommand
  ) | Out-Null
}

foreach ($entry in $paths.GetEnumerator()) {
  Ensure-PathExists -Path $entry.Value -Label $entry.Key
}

if (-not (Test-CommandExists 'npm')) {
  throw 'npm no esta disponible en PATH.'
}

if (-not $SkipDocker) {
  if (-not (Test-CommandExists 'docker')) {
    throw 'docker no esta disponible en PATH.'
  }
}

if (-not $SkipNgrok) {
  Ensure-NgrokAvailable
}

Ensure-EnvFiles

if (-not $SkipDocker) {
  Write-Step 'Levantando MongoDB con Docker Compose'
  docker compose -f (Join-Path $paths.WebServer 'compose.yaml') up -d mongodb
}

Start-DevProcess -Name 'ASL-Web Server' -WorkingDirectory $paths.WebServer -Command 'npm run dev'
Start-DevProcess -Name 'ASL-CallApp Server' -WorkingDirectory $paths.CallAppServer -Command 'npm run dev'
Start-DevProcess -Name 'ASL-Web App' -WorkingDirectory $paths.WebApp -Command 'npm run dev'

if (-not $SkipMobile) {
  Start-DevProcess -Name 'ASL-MobileAPP' -WorkingDirectory $paths.MobileApp -Command 'npm start'
}

if (-not $SkipNgrok) {
  Start-DevProcess -Name 'ngrok' -WorkingDirectory $repoRoot -Command "ngrok http $NgrokPort"
}

Write-Host ''
Write-Host 'Servicios solicitados iniciados.' -ForegroundColor Green
Write-Host 'Opciones utiles:' -ForegroundColor Cyan
Write-Host '  .\run.ps1 -SkipMobile      # omite Expo'
Write-Host '  .\run.ps1 -SkipDocker      # omite Mongo en Docker'
Write-Host '  .\run.ps1 -SkipNgrok       # omite el tunel ngrok'
Write-Host '  .\run.ps1 -NgrokPort 3101  # expone otro puerto con ngrok'
