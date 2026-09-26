# Reglas del modulo: ASL-MobileAPP

- Stack: Expo Router + React Native + TypeScript.
- Respeta la estructura por rutas en `app/` y los componentes compartidos en
  `components/`.
- Si cambias servicios (`services/`), verifica contratos con `ASL-Web` o
  `ASL-CallAPP`.
- Valida con:
  - `npm run lint`
  - `npm run typecheck`
- Evita romper imports por alias o rutas relativas.
- Si cambias assets o flujos ASL/Text, confirma que el comportamiento siga
  separado por modo.
