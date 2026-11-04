# Comparación local de FP32 e INT8

El comparador envía las mismas 32 frases sintéticas a dos contenedores ya
iniciados. No crea cuentas, no utiliza datos de huéspedes y no escribe en MongoDB.
El corpus se redactó para esta comprobación; no utiliza CSV del entrenamiento.
No tiene referencias humanas y no permite calcular precisión lingüística.

Desde la raíz del repositorio, con Python y Docker disponibles:

```powershell
python ASL-GlossTranslator/benchmarks/compare.py --docker docker
```

Si el ejecutable no está en PATH, pasa su ruta completa mediante `--docker`.
Los valores iniciales son FP32 en `127.0.0.1:8002`, contenedor
`asl-system-asl-gloss-translator-1`, e INT8 en `127.0.0.1:8003`, contenedor
`asl-system-asl-gloss-translator-int8-1`. Las opciones `--fp32-container`,
`--int8-container`, `--fp32-url` y `--int8-url` permiten seleccionar otros nombres
y puertos locales. Se rechazan destinos HTTP externos. Ambos contenedores deben
declarar `TRANSLATOR_THREADS=2` y devolver la variante correcta en `/health`.

El programa realiza tres calentamientos por variante y después tres repeticiones
de cada frase. Alterna el orden FP32/INT8 en cada caso y repetición, con una sola
petición activa en total. No es una prueba de carga concurrente. No mide el tiempo
de carga inicial del modelo: los dos servicios deben estar listos al comenzar.

Genera `comparison.json` y `COMPARACION.md` en
`.dev-logs/gloss-int8-comparison/`. Para conservar varias ejecuciones, selecciona
una carpeta diferente mediante `--output`. El JSON conserva cada respuesta,
latencia, error, calentamiento y muestra de memoria. Las tablas presentan media,
p50 y p95 de solicitudes exitosas; los fallos permanecen registrados y contados.

El uso de CPU es la diferencia de `usage_usec` en cgroup antes y después de la
carga, dividida entre peticiones. Incluye el pequeño coste de consultar recursos.
La memoria se toma de la secuencia de `docker stats`; el máximo muestreado no es
el máximo del kernel y puede omitir picos breves. El tamaño de imagen es lógico:
no equivale al espacio adicional físico después de compartir capas. Los tamaños
ONNX se registran por separado. Los dos modelos permanecen cargados en el mismo
host y cualquier otra carga local puede influir en los resultados.

La tabla de traducciones muestra la primera salida y registra estabilidad entre
repeticiones. Una coincidencia con FP32 solo indica igualdad textual. Una
diferencia puede mejorar, conservar o empeorar el significado; revisar en
particular negaciones, cantidades, habitaciones, destinos y urgencia. Antes de
usar INT8 en Azure se necesita revisar mensajes representativos y medir también
en el hardware del servidor.

Pruebas de cálculos y manejo de errores, sin modelos ni contenedores:

```powershell
python -m unittest discover -s ASL-GlossTranslator/benchmarks -p test_compare.py -v
```
