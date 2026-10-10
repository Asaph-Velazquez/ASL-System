# 📖 Directorio del Proyecto

Este repositorio concentra los modulos principales del sistema ASL para recepcion, gestion operativa, interpretacion remota y procesamiento de lenguaje de senas.

## 💾 Clonar Repositorio

Para clonar el repositorio junto con sus submodulos de desarrollo:

```bash
git clone --recurse-submodules https://github.com/Asaph-Velazquez/ASL-System.git
```

Despues puedes entrar al modulo que quieras revisar o ejecutar:

```bash
cd "Nombre del modulo de desarrollo"
```

## 🗃️ Estructura del Proyecto

Durante el desarrollo del sistema se trabaja con los siguientes modulos:

- `ASL-MobileAPP`: aplicacion movil principal del sistema, desarrollada con Expo + TypeScript.
- `ASL-Web`: panel web operativo para personal del hotel, desarrollado con React + Vite + TypeScript. Recibe, visualiza y administra solicitudes en tiempo real.
- `ASL-CallApp`: dominio de llamadas e interpretacion remota. Incluye un servidor Node.js + WebSocket para sesion de llamada, presencia de interpretes y envio de reportes a `ASL-Web`, ademas de una consola web para el interprete.
- `ASL-ModelServer`: servicio independiente de inferencia ASL, desarrollado con FastAPI. Aloja el modelo ONNX y su manifiesto, recibe landmarks de la mano (no imagenes ni video), valida la sesion del huesped y devuelve la glosa reconocida con su confianza. Se publica unicamente a traves de `/api/asl/*` en el gateway Nginx; mantiene la inferencia separada del backend operativo de `ASL-Web`.

## 🔗 Relacion Entre Modulos

El flujo general del sistema se distribuye asi:

1. `ASL-MobileAPP` captura la interaccion del huesped.
2. Para reconocimiento de senas, la app obtiene landmarks y los envia a `ASL-ModelServer`, que ejecuta el modelo ONNX y responde con una prediccion.
3. `ASL-Web` permite al personal atender solicitudes y visualizar seguimiento operativo.
4. `ASL-CallApp` gestiona llamadas en tiempo real entre huesped e interprete, y reinyecta reportes de interpretacion hacia `ASL-Web` cuando se requiere seguimiento.

`ASL-IA` ya no forma parte de la arquitectura activa ni del flujo de ejecucion. El servicio vigente para inferencia de señas es `ASL-ModelServer`.

## 🚪 Gateway Nginx Base

El repo incluye una base versionable de gateway en [infra/nginx/nginx.conf](C:/Users/samur/Downloads/TT/ASL-System/infra/nginx/nginx.conf) y [docker-compose.nginx.yml](C:/Users/samur/Downloads/TT/ASL-System/docker-compose.nginx.yml) para centralizar la entrada publica sin inspeccionar eventos de aplicacion. Esta es la **recomendacion primaria** para exposicion publica con `ngrok`.

Rutas base del gateway:

- `/api/interpreter/*` -> `ASL-CallAPP/server` en `3101`
- `/calls` -> `ASL-CallAPP/server` en `3101` con soporte `WebSocket upgrade`
- `/api/asl/*` -> `ASL-ModelServer` en la red privada de Docker, sin pasar por ASL-Web
- resto de rutas HTTP -> `ASL-Web/server` en `3001`

Arranque local del gateway:

```powershell
docker compose -f .\docker-compose.nginx.yml up -d
```

O desde el runbook raiz:

```powershell
.\run.ps1 -UseNginxGateway -NgrokPort 8080
```

El gateway y el modelo se inician por defecto con `./run.ps1` o `./run.sh`.
El puerto de ngrok se deriva de `GatewayPort` / `--gateway-port` (8080 por defecto);
un puerto explicito distinto se rechaza antes de iniciar servicios.
Para el modo legado sin reconocimiento ASL usa `-UseNginxGateway:$false` en
PowerShell o `--no-nginx-gateway` en Bash. `-SkipDocker` / `--skip-docker`
requiere ese modo legado. Los procesos de desarrollo de PowerShell se inician
en segundo plano sin abrir ventanas. Usa `./run.ps1 -ShowWindows` para mostrar
las terminales nuevas. Los logs se guardan en `.dev-logs/` (no versionados);
`Get-Content .dev-logs/ngrok-*.log -Tail 30` permite consultar errores del tunel.
El script reutiliza ngrok si su inspector local confirma el puerto esperado;
si apunta a otro puerto, se detiene sin cerrar el tunel. Los puertos ocupados
se informan con PID y no generan otra copia del servicio. Antes de terminar,
el script comprueba HTTP y el contenido esperado de las APIs, interfaces Vite,
Metro y gateway; esto es una comprobacion de disponibilidad, no una prueba E2E.
Web usa 5173 y CallApp 5174
con `strictPort`, evitando que Vite se desplace silenciosamente a otro puerto.

En Windows se ejecuta `ngrok.exe` directamente: primero se busca en PATH y,
si ngrok proviene de npm, junto al paquete que referencia su shim. No se ejecuta
el archivo sin extension que algunos wrappers npm llaman por error. La salida
nativa y los errores se incorporan al log, conservando el codigo de terminacion.
Los logs pueden contener datos de desarrollo; redactalos antes de compartirlos.

Si la camara muestra HTTP 413, verifica primero el destino del tunel:
`ngrok http 8080`, no `ngrok http 3001`. Una secuencia de 60 frames puede ocupar
unos 80 KB: el backend del hotel limita JSON a 10 KB, mientras que la ruta
autenticada de inferencia admite 256 KiB. No aumentes el limite global del hotel
para resolver un error de enrutamiento.

En Windows, `run.ps1` reutiliza `asl-mongodb` si pertenece al mismo archivo
Compose, usa `mongo:7`, conserva un montaje escribible en `/data/db` y publica
27017. Si esta detenido lo inicia; si no existe lo crea con Compose. Verifica
un ping antes de continuar. Si encuentra otro propietario/configuracion, se
detiene sin eliminar contenedores ni volumenes. Pruebas del arranque:
`./infra/tests/run-startup.tests.ps1` (simuladas; no inician servicios reales).

Con esa variante:

- `ngrok` debe publicar `http://localhost:8080`
- `ASL-Web/server` sigue en `3001`
- `ASL-CallAPP/server` sigue en `3101`
- `ASL-ModelServer` se construye y arranca con el gateway; utiliza el mismo `JWT_SECRET` que `ASL-Web/server/.env`
- antes del arranque, reemplaza el `JWT_SECRET` de ejemplo por un secreto real; el servidor del modelo rechaza el valor de ejemplo
- la app móvil debe usar la URL del gateway en `EXPO_PUBLIC_API_URL` y `EXPO_PUBLIC_WS_URL` (o usar solo `EXPO_PUBLIC_PUBLIC_BASE_URL` sin las dos variables explícitas)
- el gateway usa `host.docker.internal` para alcanzar Web y CallAPP, y una red privada para el servidor del modelo

Nota de entorno:

- `host.docker.internal` y `host-gateway` funcionan bien como base de desarrollo en Docker Desktop sobre Windows y macOS
- fuera de ese entorno pueden requerir ajuste manual del upstream o una estrategia distinta de red en Docker/Linux

## 🌍 Exposicion Publica Recomendada

Para desarrollo remoto con `ngrok`, la recomendacion del repositorio es:

1. levantar `ASL-Web/server` en `3001`
2. levantar `ASL-CallAPP/server` en `3101`
3. levantar el gateway Nginx en `8080`
4. publicar solo `http://localhost:8080`

Ese flujo evita depender de dos dominios publicos distintos y deja una entrada unica escalable para futuros servicios.

Modo de transicion:

- si todavia no usas Nginx, puedes publicar `ASL-Web/server` en `3001`
- en ese caso el proxy Node actual del backend web reenvia `/calls` y `/api/interpreter/*` hacia `ASL-CallAPP/server`
- ese camino se documenta como compatibilidad transitoria, no como recomendacion primaria

## 📝 Notas

Para pruebas en una sola instancia EC2, usar `docker-compose.ec2.yml` y seguir
[la guia EC2](infra/ec2/README.md). Incluye los tres backends, MongoDB persistente,
gateway HTTP/WebSocket y una opcion HTTPS; las aplicaciones Tauri se distribuyen
por separado. No requiere los procesos locales de `run.ps1` ni `host.docker.internal`.

- Cada modulo mantiene su propio entorno, dependencias y README local.
- Se recomienda revisar el README de cada carpeta antes de instalar o ejecutar servicios.
- Si el flujo remoto incluye videollamada, la referencia publica recomendada debe salir del gateway Nginx; no mezcles un dominio publico para `3001` con otro tunel improvisado para `3101` dentro del mismo runbook base.
