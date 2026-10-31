# Entrega de producción e instaladores — 30 de octubre de 2026

## Alcance

La entrega lleva las correcciones de los catorce casos fallidos a las ramas de
producción: `master` de ASL-MobileAPP y `main` de ASL-Web y ASL-CallAPP. El destino
de los clientes compilados es
`https://asl-system-testing.mexicocentral.cloudapp.azure.com`, con WebSocket seguro
del mismo dominio. Esta tarea no publica código en Azure ni modifica sus datos.

## Integración de ramas

El commit `a9969151e` de `ASL-System/main` fija los tres submódulos de producción
y registra sus ramas. Las confirmaciones permanecen en los repositorios locales.

| Módulo | Rama | Commit de producción | Resultado |
|---|---|---|---|
| ASL-MobileAPP | `master` | `dcf69f8` | Nueve correcciones trasladadas desde `development` y adaptación de validación de sesión |
| ASL-Web | `main` | `9e0fe85` | Los cuatro commits correctivos ya pertenecían a la rama |
| ASL-CallAPP | `main` | `e5df9f6` | Los cuatro commits correctivos ya pertenecían a la rama |

La integración móvil parte de `99510cf` y conserva el contrato de reconocimiento
de producción. Los archivos del módulo nativo, `constants/network.ts`,
`services/signSequence.ts` y `services/aslRecognition.ts` permanecen idénticos a
esa base. La resolución de URLs sigue admitiendo la configuración HTTPS/WSS de
Azure. Los cambios experimentales del modelo mixto y de resolución local de
`development` no se incluyen en esta integración.

| Caso móvil | Commit de producción |
|---|---|
| CPM_MOV_042 | `2a73625` |
| CPM_MOV_001 | `543ac73` |
| CPM_MOV_002 | `77a654e` |
| CPM_MOV_011 | `9c0a6fd` |
| CPM_MOV_004 | `9597d51` |
| CPM_CNX_003 | `247153e` |
| CPM_MOV_058 | `2409a3e` |
| CPM_MOV_016 | `e669543` |
| CPM_MOV_017 | `274ed48` |

`dcf69f8` completa la dependencia de autenticación requerida por CPM_MOV_011:
las respuestas 429, 5xx, los errores de transporte y las respuestas inválidas del
servidor se consideran transitorios. Las respuestas 401 y 403 siguen invalidando
la sesión. La revisión independiente detectó y verificó la corrección del caso de
respuesta 200 malformada, que ahora se clasifica como error transitorio 502.

En `.gitmodules` de producción se explicitan `branch = master` para el móvil y
`branch = main` para Web y CallAPP. Así, una actualización remota de submódulos
consulta las ramas de producción indicadas. El checkout ordinario continúa
respetando los commits fijados por el repositorio raíz.

## Verificación disponible

| Frontera | Evidencia |
|---|---|
| Móvil | 101 pruebas finales aprobadas y repetidas por el orquestador; tipos correctos y lint sin errores, con 18 advertencias existentes |
| Revisión móvil independiente | 29 pruebas de autenticación, privacidad, bienvenida, revocación y recuperación aprobadas; `git diff --check` correcto |
| Consola del intérprete | 30 pruebas aprobadas; compilación web y `desktop:frontend` correctas |
| Servidor de llamadas | 11 pruebas aprobadas, ninguna omitida, con MongoDB y bases temporales aisladas |
| Destino del intérprete | Bundle de escritorio verificado con HTTPS de Azure y `wss://asl-system-testing.mexicocentral.cloudapp.azure.com/calls` |
| Staff | Lint, compilación web y NSIS correctos; 7 pruebas de utilidades y 73 del backend aprobadas, ninguna omitida |
| Destino del staff | Bundle de escritorio verificado con HTTPS de Azure y WebSocket seguro `/ws/hotel` del mismo dominio |

Las pruebas de llamadas cubren sesión emitida por el hotel, señalización,
identidad central, cancelación sin reporte cuando no hubo aceptación y entrega
de reportes al hotel. Usan bases de prueba que se eliminan al terminar.

## Estado de los artefactos

La compilación del APK terminó correctamente. Los instaladores del intérprete y del
staff también se generaron y verificaron. Sus copias en `instaladores` se cotejan con
las huellas siguientes. Los artefactos anteriores no acreditan esta entrega.

| Artefacto verificado | Tamaño | SHA-256 |
|---|---:|---|
| `ASL Interpreter_0.1.0_x64-setup.exe` | 1 989 778 bytes | `8B1D6579CE08B9608303BE6528EFE54C53ABDBC47F083AB1F75DE728359D83EE` |
| `ASL Staff_0.1.0_x64-setup.exe` | 2 060 442 bytes | `134E74EADBF5FBA7854556CC821AA594CF914A746A61A89D6693387236699352` |
| `ASL-MobileAPP-1.0.0-testers.apk` | 392 476 562 bytes | `32B796A0CD77D360A8A09B20B00B708472E1F113C78736BB65D69E31D7E30992` |

El APK release se generó desde `ASL-MobileAPP/master` (`dcf69f8`) con 842 tareas
de Gradle exitosas. Declara `com.asaphvelazquez.ASLMobileAPP`, versión `1.0.0`,
SDK mínimo 24 y destino 36; contiene `arm64-v8a` y `armeabi-v7a`, no es
debuggable y conserva el certificado local del APK anterior. El bundle Hermes
incluye las URL HTTPS/WSS de Azure, `POST /api/consents` y la pantalla de
privacidad/cámara. El archivo sustituyó al APK anterior en `instaladores`; el
original quedó respaldado en `.dev-logs/installer-backup-2026-10-30/`.

El nuevo instalador del intérprete se generó en
`ASL-CallAPP/app/src-tauri/target/release/bundle/nsis/`. El ejecutable incluido
es PE AMD64 y tiene 8 707 584 bytes. Su SHA-256 es
`64FA5BBFF305A0D74A98075F248569BA518BB75F85CD9D3A37220AA0982109D5`.
El lanzador NSIS es PE de 32 bits y empaqueta la aplicación de 64 bits, conforme
al resultado `Target: x64` del empaquetador. La compilación nativa terminó en
cinco minutos y cinco segundos. El ejecutable se inició en una comprobación de
proceso, respondió con el título **ASL Interpreter Console** y se cerró mediante
su propia ventana. No se instaló el paquete ni se inició una llamada.

El instalador del staff se generó en
`ASL-Web/src-tauri/target/release/bundle/nsis/`. Su ejecutable PE AMD64 tiene
8 779 776 bytes y SHA-256
`EE2E978ECD1856523401740684BF861D11997455A3AE19EE63B0BE6E4DF0288E`.
La compilación nativa terminó en cuatro minutos y cincuenta y siete segundos.
El proceso respondió con el título **ASL Staff Control Panel** y se cerró por su
propia ventana. No se ejecutó el instalador ni se modificó una instalación previa.

El entorno cuenta con Rust 1.92.0 aislado en `.release-tools`, y las dependencias
fijadas de Cargo se descargaron correctamente. Windows devolvió «operación
cancelada» al solicitar elevación, aunque el usuario no vio el aviso UAC.
Se prepararon herramientas portables oficiales:
MSVC 14.51 y Windows SDK 10.0.26100. Los paquetes de Microsoft se extraen dentro
de `.release-tools` y se usan mediante variables del proceso, sin registro ni
instalación global. Las dos compilaciones nativas y sus paquetes NSIS terminaron
correctamente. Los paquetes de escritorio no tienen firma Authenticode.

Los instaladores de escritorio incluyen la interfaz; los backends y MongoDB se
ejecutan por separado. La actualización del servidor de Azure debe coordinarse
antes de usar los nuevos contratos, incluido `POST /api/consents`.

La comprobación de disponibilidad de Azure desde este equipo agotó los tiempos
de espera de TCP, ocho segundos, y HTTP, veinte segundos. Esto limita la
verificación remota de esta entrega; no determina que el servidor esté caído.
Las URL públicas se comprobaron en los paquetes compilados. No se ejecutaron
operaciones autenticadas ni modificaciones en el despliegue de Azure.

La comprobación física de cámara, micrófono, audio y video entre equipos, y la
impresión física de QR siguen aplazadas por decisión del usuario. Compilar los
clientes no constituye evidencia de esas pruebas.
