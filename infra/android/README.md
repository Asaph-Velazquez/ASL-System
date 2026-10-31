# Compilación Android con Docker

Java 17, Node.js 22, SDK 36, NDK 27 y CMake se preparan en una imagen Linux.
El anfitrión necesita Docker Desktop activo en modo Linux; no necesita instalar
Java. `node_modules` y Gradle usan volúmenes Linux separados de Windows.
La fuente se monta solo para lectura; se omiten `.env` privados y Android se
genera en el volumen. El contexto de construcción es solo `infra/android`.

Desde la raíz, preparar y comprobar la imagen sin generar APK:

```powershell
.\build-apk.ps1 -Variant Debug -ProjectPath .\ASL-MobileAPP -Check
```

Generar el APK final desde `master`:

```powershell
.\build-apk.ps1 -ProjectPath .\.release-worktrees\mobile `
  -PublicBaseUrl https://asl-system-testing.mexicocentral.cloudapp.azure.com
```

Salida: `.dev-logs/android-release/app-release.apk`. Para otra carpeta, usar
`-OutputDirectory`. Se muestra SHA-256 sin sustituir los instaladores existentes.
El origen HTTPS/WSS se integra en release; no despliega el backend.
Antes de compilar release se comprueba que `/api/consents` rechaza un cuerpo
vacío con el contrato esperado, sin guardar registros. Si el backend no está
actualizado, el comando se detiene con una explicación. Para preparar un paquete
antes del despliegue puede usarse `-SkipApiCheck`, verificando la API antes de distribuir.
La firma conserva el keystore del Android local cuando existe; en otro caso usa
el certificado de pruebas generado por Expo.

Para desarrollo, con Metro y ADB en el anfitrión:

```powershell
.\build-apk.ps1 -Variant Debug -ProjectPath .\ASL-MobileAPP
```

Salida: `.dev-logs/android-debug/app-debug.apk`. El contenedor no accede al USB.
SDK/NDK/CMake corresponden a Expo 54; actualizar el Dockerfile si cambian los
requisitos nativos. Los volúmenes conservan dependencias y caché, sin borrar
archivos del proyecto anfitrión.
