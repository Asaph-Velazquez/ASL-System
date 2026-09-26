# Reglas del modulo: ASL-Web

- Separa mentalmente `ASL-Web/src` y `ASL-Web/server`; no mezcles frontend y
  backend en la misma decision tecnica.
- Frontend:
  - React + Vite + TypeScript
  - valida con `npm run lint` y `npm run build`
  - no introduzcas tipos `any` si puedes modelar el dato
- Backend:
  - Express + MongoDB + WebSocket
  - conserva consistencia entre rutas, modelos y servicios
  - si cambias payloads que consume el frontend, actualiza ambos lados
- Revisa `.env.example` si agregas variables nuevas.
- Si tocas seguridad o autenticacion, revisa `middleware/`.
