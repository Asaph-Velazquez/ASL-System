$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $root 'run.ps1'), [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw ($errors | Out-String) }
$ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq 'Ensure-MongoDB'}, $false) |
  ForEach-Object { . ([scriptblock]::Create($_.Extent.Text)) }
function Write-Step { param($Message) }
function Start-Sleep { param($Seconds) }

$compose = Join-Path $root 'ASL-Web\server\compose.yaml'
function Reset-Fixture {
  $script:calls = [System.Collections.Generic.List[string]]::new()
  $script:exists = $true
  $script:daemonFailure = $false
  $script:composeFailure = $false
  $script:pingFailure = $false
  $script:fixture = @{
    Config = @{ Image = 'mongo:7'; Labels = @{
      'com.docker.compose.service' = 'mongodb'
      'com.docker.compose.project.config_files' = $compose
    } }
    Mounts = @(@{Destination='/data/db'; RW=$true})
    HostConfig = @{PortBindings=@{'27017/tcp'=@(@{HostPort='27017'})}}
    State = @{Running=$true}
  }
}
function docker {
  $script:calls.Add(($args -join ' '))
  $script:LASTEXITCODE = 0
  switch ($args[0]) {
    'ps' { if ($script:daemonFailure) { $script:LASTEXITCODE=1 } elseif ($script:exists) { 'test-container' } }
    'inspect' { ConvertTo-Json -InputObject @($script:fixture) -Depth 8 -Compress }
    'start' { }
    'compose' { if ($script:composeFailure) { $script:LASTEXITCODE=1 } }
    'exec' { if ($script:pingFailure) { $script:LASTEXITCODE=1 } }
    default { throw "Unexpected Docker operation: $args" }
  }
}
function Assert-True($condition, $message) { if (-not $condition) { throw $message } }
function Assert-Rejected($action, $pattern) {
  try { & $action } catch { if ($_.Exception.Message -match $pattern) { return }; throw }
  throw "Expected rejection: $pattern"
}

Reset-Fixture
Ensure-MongoDB $compose
Assert-True ($calls.Count -eq 3 -and $calls[2] -like 'exec *') 'Running Mongo must be inspected and pinged, not recreated.'
Reset-Fixture
$fixture.State.Running = $false
Ensure-MongoDB $compose
Assert-True ($calls[2] -eq 'start test-container') 'Stopped Mongo must reuse the existing container.'
Reset-Fixture
$fixture.Config.Labels.'com.docker.compose.project.config_files' = 'another-project'
Assert-Rejected { Ensure-MongoDB $compose } 'otra configuracion'
Assert-True ($calls.Count -eq 2) 'A foreign container must not be mutated.'
Reset-Fixture
$exists = $false
Ensure-MongoDB $compose
Assert-True ($calls[1] -like 'compose * up -d mongodb') 'Missing container must use Compose.'
Reset-Fixture
$daemonFailure = $true
Assert-Rejected { Ensure-MongoDB $compose } 'Docker Desktop'
Reset-Fixture
$exists = $false; $composeFailure = $true
Assert-Rejected { Ensure-MongoDB $compose } 'iniciar MongoDB'
Reset-Fixture
$pingFailure = $true
Assert-Rejected { Ensure-MongoDB $compose } 'no respondio'

# Evaluate only parameter/default validation, never service-launch statements.
$prefix = $ast.Extent.Text.Split([string[]]@('$repoRoot ='), [System.StringSplitOptions]::None)[0]
$resolve = [scriptblock]::Create($prefix + "`n" + '[pscustomobject]@{Gateway=[bool]$UseNginxGateway; Port=$NgrokPort}')
$config = & $resolve
Assert-True ($config.Gateway -and $config.Port -eq 8080) 'Default must route to the gateway.'
Assert-True ((& $resolve -GatewayPort 8181).Port -eq 8181) 'Custom gateway must propagate to ngrok.'
Assert-True ((& $resolve -UseNginxGateway:$false).Port -eq 3001) 'Explicit legacy mode must remain available.'
Assert-Rejected { & $resolve -NgrokPort 3001 } 'debe coincidir'
Write-Output 'PASS: 7 Mongo startup scenarios and 4 gateway port scenarios; no real services launched.'

$ast.FindAll({param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -in @('Start-DevProcess', 'Start-NgrokTunnel', 'Resolve-NgrokExecutable', 'Wait-DevService')}, $false) |
  ForEach-Object { . ([scriptblock]::Create($_.Extent.Text)) }
$repoRoot = $root
$ShowWindows = $false
$script:launches = 0
$script:occupied = $false
$script:exited = $false
$script:tunnelAddress = 'http://localhost:8080'
$script:inspectorMode = 'online'
$script:inspectorCalls = 0
function Get-NetTCPConnection { if ($script:occupied) { [pscustomobject]@{OwningProcess=123} } }
function Invoke-RestMethod {
  $script:inspectorCalls++
  if ($script:inspectorMode -eq 'offline' -or ($script:inspectorMode -eq 'start' -and $script:inspectorCalls -eq 1)) {
    throw 'Inspector not ready'
  }
  [pscustomobject]@{tunnels=@([pscustomobject]@{public_url='https://example.invalid'; config=@{addr=$script:tunnelAddress}})}
}
function Start-Process {
  param($FilePath, $WindowStyle, $ArgumentList, [switch]$PassThru)
  $script:launches++
  Assert-True ($ArgumentList[0] -eq '-NoProfile') 'Child shell must not run user profiles.'
  Assert-True ($WindowStyle -eq $(if ($ShowWindows) {'Normal'} else {'Hidden'})) 'Window style must honor ShowWindows.'
  $command = [Text.Encoding]::Unicode.GetString([Convert]::FromBase64String($ArgumentList[-1]))
  $script:lastCommand = $command
  $parseTokens=$null; $parseErrors=$null
  [void][System.Management.Automation.Language.Parser]::ParseInput($command, [ref]$parseTokens, [ref]$parseErrors)
  Assert-True ($parseErrors.Count -eq 0) 'Child script must parse even with spaces/apostrophes in paths.'
  Assert-True ($command.Contains('Start-Transcript') -and $command.Contains('exit $code')) 'Child must log and propagate exit status.'
  $process = [pscustomobject]@{Id=999; ExitCode=1}
  $process | Add-Member -MemberType ScriptMethod -Name WaitForExit -Value {param($milliseconds) return $script:exited}
  return $process
}
Start-NgrokTunnel 8080
Assert-True ($launches -eq 0) 'Existing matching tunnel must not launch ngrok again.'
$tunnelAddress = 'http://localhost:3001'
Assert-Rejected { Start-NgrokTunnel 8080 } 'no apunta'
Assert-True ($launches -eq 0) 'Wrong tunnel must not start a duplicate.'
$occupied = $true
Start-DevProcess 'test' $root 'npm.cmd run dev' 3001
Assert-True ($launches -eq 0) 'Occupied port must not launch a duplicate.'
$occupied = $false
Start-DevProcess 'test' "C:\test space\user's app" 'npm.cmd run dev' 3001
$ShowWindows = $true
Start-DevProcess 'test' $root 'npm.cmd run dev' 3001
$exited = $true
Assert-Rejected { Start-DevProcess 'test' $root 'npm.cmd run dev' 3001 } 'termino durante'
Write-Output 'PASS: 6 process/tunnel scenarios; no real services launched.'

$script:nativeBinary = 'C:\Tools\ngrok.exe'
$script:npmBinaryExists = $false
function Get-Command {
  param($Name, $CommandType, [switch]$All, $ErrorAction)
  if ($Name -eq 'ngrok.exe') {
    if ($script:nativeBinary) { [pscustomobject]@{Source=$script:nativeBinary} }
  } else { [pscustomobject]@{Source='C:\npm\ngrok.ps1'} }
}
function Test-Path { param($LiteralPath, $PathType) return $script:npmBinaryExists }
Assert-True ((Resolve-NgrokExecutable) -eq 'C:\Tools\ngrok.exe') 'Native executable must take precedence.'
$nativeBinary = $null; $npmBinaryExists = $true
Assert-True ((Resolve-NgrokExecutable) -eq 'C:\npm\node_modules\ngrok\bin\ngrok.exe') 'npm shim must resolve to its .exe, never extensionless ngrok.'
$npmBinaryExists = $false
Assert-True ($null -eq (Resolve-NgrokExecutable)) 'Missing binary must not resolve to the shim.'

$nativeBinary = 'C:\Tools\ngrok.exe'; $exited = $false; $ShowWindows = $false
$inspectorMode = 'start'; $inspectorCalls = 0; $tunnelAddress = 'http://localhost:8080'
$before = $launches
Start-NgrokTunnel 8080
Assert-True ($launches -eq $before + 1 -and $lastCommand.Contains("& 'C:\Tools\ngrok.exe' http 8080")) 'Cold startup must launch the executable exactly once.'
$inspectorCalls = 0; $tunnelAddress = 'http://localhost:3001'
Assert-Rejected { Start-NgrokTunnel 8080 } 'no apunta'
$inspectorMode = 'offline'
$before = $launches
Assert-Rejected { Start-NgrokTunnel 8080 } 'no confirmo'
Assert-True ($launches -eq $before + 1) 'Timeout must not recursively create more ngrok processes.'

$script:httpBody = 'packager-status:running'
function Invoke-WebRequest { [pscustomobject]@{StatusCode=200; Content=[Text.Encoding]::UTF8.GetBytes($script:httpBody)} }
Wait-DevService -Name 'Expo test' -Url 'http://localhost:8081/status' -ExpectedText 'packager-status:running' -TimeoutSeconds 0
$httpBody = 'unrelated server'
Assert-Rejected { Wait-DevService -Name 'Expo test' -Url 'http://localhost:8081/status' -ExpectedText 'packager-status:running' -TimeoutSeconds 0 } 'no respondio'
Write-Output 'PASS: 3 executable resolution, 3 cold tunnel and 2 HTTP readiness scenarios.'
