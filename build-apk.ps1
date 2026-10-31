[CmdletBinding()]
param(
    [string]$ProjectPath = '.release-worktrees/mobile',
    [ValidateSet('Debug', 'Release')][string]$Variant = 'Release',
    [uri]$PublicBaseUrl,
    [string]$OutputDirectory,
    [switch]$SkipApiCheck,
    [switch]$Check
)
$ErrorActionPreference = 'Stop'
if (-not [IO.Path]::IsPathRooted($ProjectPath)) { $ProjectPath = Join-Path $PSScriptRoot $ProjectPath }
$taskProject = (Resolve-Path -LiteralPath $ProjectPath).Path
if (-not (Test-Path -LiteralPath (Join-Path $taskProject 'package.json'))) { throw 'ProjectPath debe ser el proyecto móvil.' }
if ($Variant -eq 'Release') {
    if (-not $PublicBaseUrl -or $PublicBaseUrl.Scheme -ne 'https' -or $PublicBaseUrl.IsLoopback -or $PublicBaseUrl.Host -in @('0.0.0.0', '[::]', '::') -or $PublicBaseUrl.AbsolutePath -ne '/' -or $PublicBaseUrl.UserInfo -or $PublicBaseUrl.Query -or $PublicBaseUrl.Fragment) {
        throw 'Release requiere -PublicBaseUrl con el origen HTTPS público, sin ruta ni credenciales.'
    }
    $taskBranch = (& git -C $taskProject branch --show-current).Trim()
    if ($LASTEXITCODE -ne 0 -or $taskBranch -ne 'master') { throw 'Release debe compilarse desde master.' }
    if (-not $SkipApiCheck) {
        Add-Type -AssemblyName System.Net.Http
        $taskHttp = [Net.Http.HttpClient]::new()
        $taskHttp.Timeout = [TimeSpan]::FromSeconds(12)
        $taskBody = [Net.Http.StringContent]::new('{}', [Text.Encoding]::UTF8, 'application/json')
        try {
            $taskResponse = $taskHttp.PostAsync(($PublicBaseUrl.AbsoluteUri.TrimEnd('/') + '/api/consents'), $taskBody).GetAwaiter().GetResult()
            try {
                $taskReceipt = $taskResponse.Content.ReadAsStringAsync().GetAwaiter().GetResult()
                if ([int]$taskResponse.StatusCode -ne 400 -or $taskReceipt -notmatch 'Invalid consent') { throw 'El backend no ofrece el contrato de consentimiento actualizado. Desplegar ASL-Web/server antes de distribuir el APK.' }
            } finally { $taskResponse.Dispose() }
        } finally { $taskBody.Dispose(); $taskHttp.Dispose() }
    } else { Write-Warning 'Se omite la comprobación API de consentimiento; verificar el despliegue antes de distribuir.' }
}
$taskDockerCommand = Get-Command docker -ErrorAction SilentlyContinue
if ($taskDockerCommand) { $taskDocker = $taskDockerCommand.Source } else { $taskDocker = Join-Path $env:LOCALAPPDATA 'Programs/DockerDesktop/resources/bin/docker.exe' }
if (-not (Test-Path -LiteralPath $taskDocker)) { throw 'Falta Docker Desktop. Iniciarlo antes de compilar.' }
& $taskDocker info --format '{{.ServerVersion}}'
if ($LASTEXITCODE -ne 0) { throw 'Docker Desktop debe estar activo y usar contenedores Linux.' }
$taskImage = 'asl-android-builder:node22-jdk17-sdk36'
& $taskDocker build --tag $taskImage (Join-Path $PSScriptRoot 'infra/android')
if ($LASTEXITCODE -ne 0) { throw 'No se pudo preparar la imagen Android.' }
if ($Check) {
    & $taskDocker run --rm --mount "type=bind,source=$taskProject,target=/source,readonly" --mount 'type=volume,source=asl-android-gradle,target=/root/.gradle' --entrypoint bash $taskImage -lc 'java -version && node --version && test -f "$ANDROID_HOME/platforms/android-36/android.jar" && test -x "$ANDROID_HOME/ndk/27.1.12297006/toolchains/llvm/prebuilt/linux-x86_64/bin/clang" && mkdir -p /tmp/gradle-check && cp -R /source/android/gradle /tmp/gradle-check/ && tr -d "\r" < /source/android/gradlew > /tmp/gradle-check/gradlew && cd /tmp/gradle-check && bash gradlew --version'
    if ($LASTEXITCODE -ne 0) { throw 'La imagen no supera la comprobación Java/SDK/NDK/Gradle; Check necesita el wrapper Android generado.' }
    Write-Host 'Imagen Android preparada; Java, SDK y NDK se ejecutan dentro de Docker.'
    return
}
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $PSScriptRoot ('.dev-logs/android-' + $Variant.ToLowerInvariant()) }
if (-not [IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory = Join-Path $PSScriptRoot $OutputDirectory }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$taskOutput = (Resolve-Path -LiteralPath $OutputDirectory).Path
$taskHash = [BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($taskProject))).Replace('-', '').Substring(0, 12).ToLowerInvariant()
$taskArguments = @('run', '--rm', '--name', "asl-android-build-$taskHash", '--mount', "type=bind,source=$taskProject,target=/source,readonly", '--mount', "type=bind,source=$taskOutput,target=/output", '--mount', "type=volume,source=asl-android-work-$taskHash,target=/workspace", '--mount', "type=volume,source=asl-android-node-$taskHash,target=/workspace/node_modules", '--mount', 'type=volume,source=asl-android-gradle,target=/root/.gradle', '--env', "BUILD_VARIANT=$Variant")
if ($PublicBaseUrl) { $taskArguments += @('--env', "PUBLIC_BASE_URL=$($PublicBaseUrl.AbsoluteUri.TrimEnd('/'))") }
if ($Variant -eq 'Debug' -and $env:EXPO_PUBLIC_LOCAL_GATEWAY_URL) { $taskArguments += @('--env', "EXPO_PUBLIC_LOCAL_GATEWAY_URL=$env:EXPO_PUBLIC_LOCAL_GATEWAY_URL") }
$taskArguments += $taskImage
& $taskDocker @taskArguments
if ($LASTEXITCODE -ne 0) { throw 'Falló el build Android en Docker; revisar el error anterior.' }
$taskApk = Join-Path $taskOutput ('app-' + $Variant.ToLowerInvariant() + '.apk')
if (-not (Test-Path -LiteralPath $taskApk)) { throw 'Falta el APK esperado.' }
Get-FileHash -LiteralPath $taskApk -Algorithm SHA256
Write-Host "APK generado: $taskApk"
