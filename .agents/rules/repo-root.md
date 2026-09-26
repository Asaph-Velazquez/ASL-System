# Reglas del modulo: repo-root

- Este modulo cubre solo infraestructura compartida, scripts de raiz,
  automatizacion local, documentacion y configuracion transversal del monorepo.
- No mezcles aqui cambios de logica de aplicacion que pertenezcan a
  `ASL-Web`, `ASL-MobileAPP`, `ASL-CallAPP` o `ASL-IA`.
- Si cambias scripts de arranque como `run.ps1`:
  - valida que las rutas existan realmente
  - no anuncies flujos que el script no pueda levantar
  - deja mensajes de consola consistentes con el runbook actual
- Si agregas infraestructura compartida como `docker-compose`, `nginx` o
  gateways:
  - enruta por path, host o protocolo, no por payload de aplicacion
  - documenta claramente dependencias locales como Docker, `host.docker.internal`
    o puertos esperados
  - evita fijar dominios, secretos o valores dependientes de una sola maquina
- Si cambias documentacion:
  - alinea `README.md` raiz con los README de modulos afectados
  - distingue entre recomendacion primaria, compatibilidad transitoria y
    limitaciones conocidas
- Valida con:
  - revision manual del diff y del flujo documentado
  - parseo o ejecucion dirigida del script/config cambiado cuando aplique
- Si introduces nuevas variables de entorno o artefactos operativos, deja claro
  donde viven y que modulo los consume.
