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

- `ASL-IA`: desarrollo del modelo de Machine Learning para procesamiento de lenguaje de senas, principalmente en Python.
- `ASL-MobileAPP`: aplicacion movil principal del sistema, desarrollada con Expo + TypeScript.
- `ASL-Web`: panel web operativo para personal del hotel, desarrollado con React + Vite + TypeScript. Recibe, visualiza y administra solicitudes en tiempo real.
- `ASL-CallApp`: dominio de llamadas e interpretacion remota. Incluye un servidor Node.js + WebSocket para sesion de llamada, presencia de interpretes y envio de reportes a `ASL-Web`, ademas de una consola web para el interprete.

## 🔗 Relacion Entre Modulos

El flujo general del sistema se distribuye asi:

1. `ASL-MobileAPP` captura la interaccion del huesped.
2. `ASL-IA` procesa o apoya el entendimiento del lenguaje de senas.
3. `ASL-Web` permite al personal atender solicitudes y visualizar seguimiento operativo.
4. `ASL-CallApp` gestiona llamadas en tiempo real entre huesped e interprete, y reinyecta reportes de interpretacion hacia `ASL-Web` cuando se requiere seguimiento.

## 🚪 Gateway Nginx Base

El repo incluye una base versionable de gateway en [infra/nginx/nginx.conf](C:/Users/samur/Downloads/TT/ASL-System/infra/nginx/nginx.conf) y [docker-compose.nginx.yml](C:/Users/samur/Downloads/TT/ASL-System/docker-compose.nginx.yml) para centralizar la entrada publica sin inspeccionar eventos de aplicacion. Esta es la **recomendacion primaria** para exposicion publica con `ngrok`.

Rutas base del gateway:

- `/api/interpreter/*` -> `ASL-CallAPP/server` en `3101`
- `/calls` -> `ASL-CallAPP/server` en `3101` con soporte `WebSocket upgrade`
- resto de rutas HTTP -> `ASL-Web/server` en `3001`

Arranque local del gateway:

```powershell
docker compose -f .\docker-compose.nginx.yml up -d
```

O desde el runbook raiz:

```powershell
.\run.ps1 -UseNginxGateway -NgrokPort 8080
```

Con esa variante:

- `ngrok` debe publicar `http://localhost:8080`
- `ASL-Web/server` sigue en `3001`
- `ASL-CallAPP/server` sigue en `3101`
- el gateway usa `host.docker.internal` para alcanzar ambos servicios desde el contenedor

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

- Cada modulo mantiene su propio entorno, dependencias y README local.
- Se recomienda revisar el README de cada carpeta antes de instalar o ejecutar servicios.
- Si el flujo remoto incluye videollamada, la referencia publica recomendada debe salir del gateway Nginx; no mezcles un dominio publico para `3001` con otro tunel improvisado para `3101` dentro del mismo runbook base.
