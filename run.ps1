[CmdletBinding()]
param(
  [switch]$SkipMobile,
  [switch]$SkipDocker,
  [switch]$SkipNgrok,
  [switch]$ShowWindows,
  [switch]$UseNginxGateway = $true,
  [int]$GatewayPort = 8080,
  [int]$NgrokPort = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($NgrokPort -eq 0) {
  $NgrokPort = if ($UseNginxGateway) { $GatewayPort } else { 3001 }
}
if (-not $SkipNgrok -and $UseNginxGateway -and $NgrokPort -ne $GatewayPort) {
  throw 'NgrokPort debe coincidir con GatewayPort para habilitar inferencia ASL.'
}

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$paths = @{
  WebApp        = Join-Path $repoRoot 'ASL-Web'
  WebServer     = Join-Path $repoRoot 'ASL-Web\server'
  CallAppServer = Join-Path $repoRoot 'ASL-CallAPP\server'
  CallApp = Join-Path $repoRoot 'ASL-CallAPP\app'
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
  },
  @{
    Example = Join-Path $paths.CallApp '.env.example'
    Target  = Join-Path $paths.CallApp '.env'
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
  if (Resolve-NgrokExecutable) {
    return
  }

  Install-Ngrok

  if (-not (Resolve-NgrokExecutable)) {
    Update-SessionPathFromNpm
  }

  if (-not (Resolve-NgrokExecutable)) {
    throw 'No se encontro ngrok.exe. Un shim ngrok.ps1/ngrok.cmd no garantiza un binario Windows valido.'
  }

  Write-Host 'ngrok instalado y disponible.' -ForegroundColor Green
}

function Resolve-NgrokExecutable {
  $native = Get-Command 'ngrok.exe' -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($native) { return $native.Source }
  # The npm shim can invoke an extensionless file instead of the Windows binary.
  foreach ($shim in @(Get-Command 'ngrok' -All -ErrorAction SilentlyContinue)) {
    if (-not $shim.Source) { continue }
    $candidate = Join-Path (Split-Path -Parent $shim.Source) 'node_modules\ngrok\bin\ngrok.exe'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
  }
  return $null
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

function Ensure-MongoDB {
  param([string]$ComposePath)

  $existing = docker ps -a --filter 'name=^/asl-mongodb$' --format '{{.ID}}'
  if ($LASTEXITCODE -ne 0) { throw 'No se pudo consultar Docker. Verifica Docker Desktop.' }
  if ($existing) {
    $raw = docker inspect $existing
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo inspeccionar asl-mongodb.' }
    $container = @($raw | ConvertFrom-Json)[0]
    $labels = $container.Config.Labels
    $ownerFile = $labels.'com.docker.compose.project.config_files'
    $dataMount = @($container.Mounts | Where-Object { $_.Destination -eq '/data/db' -and $_.RW })
    $port = @($container.HostConfig.PortBindings.'27017/tcp' | Where-Object { $_.HostPort -eq '27017' })
    if ($labels.'com.docker.compose.service' -ne 'mongodb' -or
        $ownerFile -ne $ComposePath -or $container.Config.Image -ne 'mongo:7' -or
        $dataMount.Count -ne 1 -or $port.Count -eq 0) {
      throw 'asl-mongodb existe con otra configuracion. No se modifico ni borro; revisa su propietario, volumen y puerto.'
    }
    Write-Step 'Reutilizando asl-mongodb y su volumen existente'
    if (-not $container.State.Running) {
      docker start $existing | Out-Null
      if ($LASTEXITCODE -ne 0) { throw 'No se pudo iniciar el MongoDB existente.' }
    }
  } else {
    docker compose -f $ComposePath up -d mongodb
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo iniciar MongoDB.' }
  }
  for ($attempt = 0; $attempt -lt 15; $attempt++) {
    docker exec asl-mongodb mongosh --quiet --eval 'quit(db.adminCommand({ping:1}).ok === 1 ? 0 : 1)' 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { return }
    Start-Sleep -Seconds 1
  }
  throw 'MongoDB no respondio al ping; no se iniciaran los demas servicios.'
}

function Wait-DevService {
  param([string]$Name, [string]$Url, [string]$ExpectedText = '', [int]$TimeoutSeconds = 30)
  $timer = [Diagnostics.Stopwatch]::StartNew()
  do {
    try {
      $response = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 2 -ErrorAction Stop
      $content = $response.Content
      if ($content -is [byte[]]) { $content = [Text.Encoding]::UTF8.GetString($content) }
      if ($response.StatusCode -eq 200 -and (!$ExpectedText -or $content -match $ExpectedText)) {
        Write-Host "$Name disponible: $Url" -ForegroundColor Green
        return
      }
    } catch { }
    Start-Sleep -Milliseconds 500
  } while ($timer.Elapsed.TotalSeconds -lt $TimeoutSeconds)
  throw "$Name no respondio correctamente en $Url. Revisa su log en .dev-logs y el proceso que ocupa el puerto."
}

function Start-DevProcess {
  param(
    [string]$Name,
    [string]$WorkingDirectory,
    [string]$Command,
    [int]$Port = 0,
    [string]$ReadyUrl = '',
    [string]$ExpectedText = ''
  )

  if ($Port -gt 0) {
    $listener = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    if ($listener.Count -gt 0) {
      Write-Host "$Name : puerto $Port ocupado (PID $($listener[0].OwningProcess)); no se inicia otra copia. Verifica el servicio existente." -ForegroundColor Yellow
      if ($ReadyUrl) { Wait-DevService -Name $Name -Url $ReadyUrl -ExpectedText $ExpectedText }
      return
    }
  }
  $logDirectory = Join-Path $repoRoot '.dev-logs'
  New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
  $logPath = Join-Path $logDirectory (('{0}-{1}.log' -f ($Name -replace '[^a-zA-Z0-9-]', '_'), (Get-Date -Format 'yyyyMMdd-HHmmss-fff')))
  $escapedDirectory = $WorkingDirectory.Replace("'", "''")
  $escapedLog = $logPath.Replace("'", "''")
  $wrappedCommand = @"
`$ErrorActionPreference = 'Stop'
`$code = 1
try {
  Start-Transcript -LiteralPath '$escapedLog' -Force | Out-Null
  Set-Location -LiteralPath '$escapedDirectory'
  `$global:LASTEXITCODE = 0
  # Native stderr is output, not proof of failure; honor the process exit code.
  `$ErrorActionPreference = 'Continue'
  & { $Command } *>&1 | ForEach-Object { Write-Host `$_ }
  `$code = `$LASTEXITCODE
  Write-Host "Proceso finalizado. Codigo: `$code"
} catch {
  Write-Host (`$_ | Out-String) -ForegroundColor Red
} finally {
  Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
exit `$code
"@
  $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($wrappedCommand))
  $style = if ($ShowWindows) { 'Normal' } else { 'Hidden' }
  Write-Step "Iniciando $Name. Log: $logPath"
  $process = Start-Process -FilePath 'powershell' -WindowStyle $style -ArgumentList @(
    '-NoProfile', '-EncodedCommand', $encoded
  ) -PassThru
  if ($process.WaitForExit(1500)) {
    throw "$Name termino durante el arranque (codigo $($process.ExitCode)). Revisa $logPath"
  }
  if ($ReadyUrl) { Wait-DevService -Name $Name -Url $ReadyUrl -ExpectedText $ExpectedText }
  Write-Host "$Name iniciado (PID $($process.Id))."
}

function Start-NgrokTunnel {
  param([int]$Port)
  $existing = $null
  try { $existing = Invoke-RestMethod 'http://127.0.0.1:4040/api/tunnels' -TimeoutSec 3 } catch { }
  if ($null -ne $existing) {
    $matching = @($existing.tunnels | Where-Object {
      $address = [string]$_.config.addr
      $address -in @("http://localhost:$Port", "http://127.0.0.1:$Port", "localhost:$Port", "127.0.0.1:$Port", "$Port")
    })
    if ($matching.Count -gt 0) {
      Write-Step "Reutilizando ngrok activo: $($matching[0].public_url) -> $Port"
      return
    }
    throw "Ngrok ya esta activo pero no apunta a $Port. No se inicio una segunda sesion ni se cerro el tunel existente."
  }
  $executable = Resolve-NgrokExecutable
  if (-not $executable) { throw 'No se encontro ngrok.exe; revisa su instalacion.' }
  $escapedExecutable = $executable.Replace("'", "''")
  Start-DevProcess -Name 'ngrok' -WorkingDirectory $repoRoot -Command "& '$escapedExecutable' http $Port --log stdout"
  for ($attempt = 0; $attempt -lt 15; $attempt++) {
    try {
      $ready = Invoke-RestMethod 'http://127.0.0.1:4040/api/tunnels' -TimeoutSec 2
    } catch { $ready = $null }
    if ($null -ne $ready -and @($ready.tunnels).Count -gt 0) {
      $matching = @($ready.tunnels | Where-Object {
        [string]$_.config.addr -in @("http://localhost:$Port", "http://127.0.0.1:$Port", "localhost:$Port", "127.0.0.1:$Port", "$Port")
      })
      if ($matching.Count -gt 0) {
        Write-Step "Ngrok listo: $($matching[0].public_url) -> $Port"
        return
      }
      throw "Ngrok inicio pero el tunel no apunta a $Port. Revisa .dev-logs/ngrok-*.log."
    }
    Start-Sleep -Seconds 1
  }
  throw 'Ngrok no confirmo el tunel. Revisa .dev-logs/ngrok-*.log para el error; no se anuncia como iniciado.'
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

if ($UseNginxGateway -and $SkipDocker) {
  throw 'No puedes combinar -UseNginxGateway con -SkipDocker en este runbook. El gateway Nginx se levanta con docker compose; quita -SkipDocker o arranca el gateway manualmente sin usar esta opcion.'
}

if (-not $SkipNgrok) {
  Ensure-NgrokAvailable
}

Ensure-EnvFiles

if (-not $SkipDocker) {
  Write-Step 'Levantando MongoDB con Docker Compose'
  Ensure-MongoDB -ComposePath (Join-Path $paths.WebServer 'compose.yaml')

  if ($UseNginxGateway) {
    Write-Step "Levantando gateway Nginx en el puerto $GatewayPort"
    $env:ASL_GATEWAY_PORT = [string]$GatewayPort
    docker compose -f (Join-Path $repoRoot 'docker-compose.nginx.yml') up -d
    if ($LASTEXITCODE -ne 0) { throw 'No se pudo iniciar el gateway y el modelo.' }
  }
}

Start-DevProcess -Name 'ASL-Web Server' -WorkingDirectory $paths.WebServer -Command 'npm.cmd run dev' -Port 3001 -ReadyUrl 'http://localhost:3001/api/health' -ExpectedText '"status"\s*:\s*"ok"'
Start-DevProcess -Name 'ASL-CallApp Server' -WorkingDirectory $paths.CallAppServer -Command 'npm.cmd run dev' -Port 3101 -ReadyUrl 'http://localhost:3101/api/health' -ExpectedText '"mongoReadyState"\s*:\s*1'
Start-DevProcess -Name 'ASL-Web App' -WorkingDirectory $paths.WebApp -Command 'npm.cmd run dev -- --port 5173 --strictPort' -Port 5173 -ReadyUrl 'http://localhost:5173' -ExpectedText '/src/main.tsx'
Start-DevProcess -Name 'ASL-CallApp App' -WorkingDirectory $paths.CallApp -Command 'npm.cmd run dev -- --port 5174 --strictPort' -Port 5174 -ReadyUrl 'http://localhost:5174' -ExpectedText '/src/main.tsx'

if (-not $SkipMobile) {
  Start-DevProcess -Name 'ASL-MobileAPP' -WorkingDirectory $paths.MobileApp -Command 'npm.cmd start -- --port 8081' -Port 8081 -ReadyUrl 'http://localhost:8081/status' -ExpectedText 'packager-status:running'
}

if ($UseNginxGateway) {
  Wait-DevService -Name 'Gateway' -Url "http://localhost:$GatewayPort/api/health" -ExpectedText '"status"\s*:\s*"ok"'
}

if (-not $SkipNgrok) {
  $recommendedNgrokPort = if ($UseNginxGateway) { $GatewayPort } else { 3001 }

  if ($NgrokPort -ne $recommendedNgrokPort) {
    if ($UseNginxGateway) {
      Write-Host "Advertencia: con -UseNginxGateway, el tunel recomendado debe apuntar al gateway Nginx en el puerto $GatewayPort." -ForegroundColor Yellow
    } else {
      Write-Host 'Advertencia: sin -UseNginxGateway, el tunel seguira saliendo por el backend web actual en 3001. Ese modo queda como transicion; la recomendacion primaria es usar el gateway Nginx.' -ForegroundColor Yellow
    }
  }

  Start-NgrokTunnel -Port $NgrokPort
}

Write-Host ''
Write-Host 'Servicios HTTP solicitados verificados; revisa las URL y logs anteriores.' -ForegroundColor Green
Write-Host 'Tunel recomendado:' -ForegroundColor Cyan
if ($UseNginxGateway) {
  Write-Host "  Publica el gateway Nginx en $GatewayPort; /api/asl va al modelo, /calls y /api/interpreter a ASL-CallAPP y el resto a ASL-Web."
} else {
  Write-Host '  Modo transicion: publica ASL-Web/server en 3001; el proxy Node actual reenvia /calls y /api/interpreter hacia ASL-CallAPP/server.'
}
Write-Host 'Opciones utiles:' -ForegroundColor Cyan
Write-Host '  .\run.ps1 -ShowWindows    # muestra las terminales de los procesos nuevos'
Write-Host '  .\run.ps1 -SkipMobile      # omite Expo'
Write-Host '  .\run.ps1 -SkipDocker -UseNginxGateway:$false  # modo legado sin inferencia'
Write-Host '  .\run.ps1 -SkipNgrok       # omite el tunel ngrok'
Write-Host '  .\run.ps1                 # gateway y modelo por defecto, ngrok en 8080'
Write-Host '  .\run.ps1 -GatewayPort 8080 -UseNginxGateway'
Write-Host '  .\run.ps1 -UseNginxGateway:$false  # modo legado sin inferencia, puerto 3001'
