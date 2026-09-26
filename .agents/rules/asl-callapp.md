# Reglas del modulo: ASL-CallAPP

- Trata `app/` y `server/` como submodulos distintos.
- `app/`:
  - React + Vite + TypeScript
  - valida con `npm run build`
- `server/`:
  - Express + WebSocket + MongoDB
  - revisa autenticacion, presencia de interpretes y cierre de llamadas
- Si cambias el reporte reenviado a `ASL-Web`, valida nombres de campos,
  endpoint destino y dependencias cruzadas.
- Revisa `.env.example` o documenta variables nuevas si el cambio las requiere.
