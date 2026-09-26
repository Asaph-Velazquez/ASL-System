# AGENTS.md - Mapa operativo para agentes de IA

Este archivo define el contexto minimo para que un agente trabaje en `ASL-System`
sin asumir un stack incorrecto. Es un monorepo con varios modulos; antes de
editar, identifica exactamente cual vas a tocar.

## 1. Arranque obligatorio

1. Lee `.agents/feature_list.json` y toma una sola tarea `pending` o la tarea
   explicita indicada por el usuario.
2. Antes de implementar, cambia el `status` de esa feature a `in_progress` en
   `.agents/feature_list.json`.
3. Revisa el arbol del repo y confirma el modulo afectado:
   - `ASL-Web`
   - `ASL-MobileAPP`
   - `ASL-CallAPP`
   - `ASL-IA`
4. Lee el `README.md` raiz y el `README.md` del modulo que vayas a modificar.
5. Lee las reglas especificas en `.agents/rules/<modulo>.md`.
6. Antes de cambiar codigo, revisa si hay cambios locales en el modulo con
   `git status --short`.

## 2. Mapa del repositorio

| Ruta | Rol |
|---|---|
| `README.md` | Vista general del sistema y relacion entre modulos |
| `run.ps1` | Arranque local coordinado de Web, Mobile, CallApp, Mongo y ngrok |
| `.agents/feature_list.json` | Backlog base para agentes |
| `.agents/agents/*.md` | Plantillas de rol para lider, implementador y revisor |
| `.agents/rules/` | Reglas por modulo |
| `ASL-Web/` | Panel web operativo en React + Vite |
| `ASL-Web/server/` | Backend Express + WebSocket + Mongo del panel |
| `ASL-MobileAPP/` | App movil Expo Router + React Native |
| `ASL-CallAPP/app/` | Consola web del interprete |
| `ASL-CallAPP/server/` | Backend de llamadas e interpretacion |
| `ASL-IA/` | Scripts Python para reconocimiento de senas |

## 3. Reglas duras

- Una sola tarea a la vez.
- No inventes rutas, servicios o tecnologias que no existan en el repo.
- Si tocas un modulo JS/TS, valida al menos con sus scripts reales (`lint`,
  `build`, `typecheck` segun aplique).
- Si tocas `ASL-IA`, valida con ejecucion dirigida del script afectado o con una
  verificacion estatica razonable si el flujo completo depende de camara/dataset.
- No declares una tarea terminada si no registraste que comandos de verificacion
  corriste y cual fue el resultado.
- No sobrescribas cambios locales ajenos del usuario.
- No dejes `console.log`, `print`, secretos, archivos temporales ni comentarios
  de relleno.
- No dejes la feature con `status` incorrecto: usa `in_progress` al comenzar,
  `done` solo si cumple deliverables y verificacion, y `blocked` si existe un
  bloqueo real documentado.

## 4. Convenciones del proyecto

### Monorepo

- `ASL-Web` y `ASL-CallAPP/app` usan React + Vite + TypeScript.
- `ASL-Web/server` y `ASL-CallAPP/server` usan Node.js + Express + MongoDB.
- `ASL-MobileAPP` usa Expo Router + React Native + TypeScript.
- `ASL-IA` usa Python con `opencv-python`, `mediapipe`, `numpy` y `pandas`.

### Comandos utiles

```powershell
.\run.ps1
git status --short
```

### Verificacion por modulo

- `ASL-Web`
  - `npm run lint`
  - `npm run build`
- `ASL-Web/server`
  - `npm start` o `npm run dev` para humo local
- `ASL-MobileAPP`
  - `npm run lint`
  - `npm run typecheck`
- `ASL-CallAPP/app`
  - `npm run build`
- `ASL-CallAPP/server`
  - `npm start` o `npm run dev` para humo local
- `ASL-IA`
  - `python -m py_compile <archivo>.py`
  - ejecucion dirigida del script si el cambio lo requiere

## 5. Flujo recomendado

1. Identificar modulo y alcance.
2. Leer reglas del modulo.
3. Implementar el cambio minimo suficiente.
4. Ejecutar verificaciones del modulo.
5. Actualizar el `status` de la feature en `.agents/feature_list.json` a
   `done` o `blocked` segun el resultado real.
6. Revisar diff final para evitar ruido.

## 6. Si te bloqueas

- Busca primero en el modulo afectado patrones existentes.
- Si el bloqueo depende de infraestructura externa, documenta exactamente que
  comando fallo y por que.
- Si la tarea cruza modulos, divide el trabajo por frontera tecnica antes de
  editar.
