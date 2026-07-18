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

## 📝 Notas

- Cada modulo mantiene su propio entorno, dependencias y README local.
- Se recomienda revisar el README de cada carpeta antes de instalar o ejecutar servicios.
