# Servicio de glosas ASL a inglés

Servicio privado de ASL-Web para convertir frases en glosas a inglés antes de
guardar y distribuir una solicitud. Ejecuta el paquete original
`model_onnx/asl_translator_fp32/asl_translator_fp32`: FLAN-T5 ajustado al dominio,
encoder y decoder ONNX de precisión completa, tokenizer, corrector de contexto,
reglas de preprocesamiento y revisión de sustantivos. Los pesos y scripts del
paquete se conservan sin cambios.

El contenedor funciona en CPU y no descarga modelos durante su ejecución.
No necesita GPU, PyTorch ni Transformers. Verifica tamaños y SHA-256 de los
16 archivos del manifiesto al construir la imagen y antes de cargar el modelo.
La carga ocurre una sola vez por proceso; `/health` responde cuando termina.

## Contrato privado

`POST /translate`, con JSON:

```json
{"text":"I WATER NEED","source":"recognizer","is_question":false}
```

`text` es obligatorio. `source` admite `"recognizer"` o `null`; cuando se omite,
el paquete detecta si el texto es glosa o inglés natural. El inglés natural se
devuelve sin traducir. `is_question` admite un booleano o `null` y se puede omitir.
La respuesta conserva el contrato del paquete:

```json
{
  "gloss":"I WATER NEED",
  "input_type":"gloss",
  "corrected":"I WATER NEED",
  "changes":[],
  "preprocessed":"X-I NEED WATER",
  "english":"I need water.",
  "noun_fixes":[]
}
```

El ejemplo muestra la forma de la respuesta; el contenido exacto depende del
modelo. `GET /health` devuelve
`{"status":"ok","model":"asl_gloss_to_english","variant":"fp32"}`.
El servicio no autentica usuarios del hotel: solo debe ser accesible desde el
backend por una red privada de Docker o un puerto limitado a `127.0.0.1` en
desarrollo. No se publica en el gateway ni se llama directamente desde las apps.
ASL-Web conserva la autenticación y autorización existentes.

ASL-Web aplica autodetección a solicitudes nuevas tanto de modo texto como ASL.
Conserva el mensaje original en los metadatos; los reintentos no vuelven a
traducir. Una indisponibilidad del traductor impide guardar y difundir la petición.

Para producción, combina el Compose existente con
`docker-compose.gloss-production.yml`, manteniendo sus archivos de entorno y
otros overlays (por ejemplo HTTPS). Construye primero la imagen FP32 con el
paquete local verificado y ejecuta `up -d --no-deps --wait` solo para
`asl-gloss-translator asl-web-server`. El traductor queda sin puertos públicos.
Respalda la imagen web anterior antes de recrearla. Los pesos no se versionan:
distribuye el paquete verificado o la imagen construida por un canal privado.

## Límites y errores

- Cuerpo JSON: 8192 bytes; texto: entre 1 y 2000 caracteres, sin entradas vacías.
- Entrada efectiva: hasta 128 tokens, incluyendo el prefijo del modelo y el
  resultado de las reglas. Una entrada mayor se rechaza con `422`; no se entrega
  una traducción de un texto truncado silenciosamente.
- El modelo conserva su generación original: cuatro haces y un máximo de
  128 tokens de salida. Una traducción puede contener errores u omisiones;
  conservar la glosa original permite revisarla.
- Hay una sola inferencia simultánea y no hay cola de solicitudes. Otra
  petición concurrente recibe `503 translator_busy` y `Retry-After: 1`.
- El plazo de respuesta predeterminado es de 10 segundos (`504` al agotarse).
  La CPU nativa continúa hasta terminar y mantiene ocupado su cupo: cancelar
  una espera HTTP no cancela ONNX. ASL-Web debe conservar el error como fallo
  de traducción, sin presentar la glosa original como una traducción exitosa.
- Datos inválidos: `422`; cuerpo demasiado grande: `413`; modelo no listo o
  fallo de inferencia: `503`. Los errores no incluyen el texto recibido.

## Construcción y arranque local

Desde la raíz del repositorio:

```powershell
docker compose -f docker-compose.gloss.yml up -d --build
```

El puerto local es `http://127.0.0.1:8002`. Para ASL-Web ejecutado en el host,
configura `GLOSS_TRANSLATOR_URL=http://127.0.0.1:8002`. Si ambos servicios están
en la misma red de Docker, usa `http://asl-gloss-translator:8000`.

También se puede construir de forma independiente:

```powershell
docker build -f ASL-GlossTranslator/Dockerfile -t asl-gloss-translator .
docker run --rm -p 127.0.0.1:8002:8000 asl-gloss-translator
```

El contexto de construcción usa una lista de archivos permitidos en
`Dockerfile.dockerignore`; no envía dependencias de otras apps, archivos `.env`
ni registros. El proceso del contenedor corre como usuario sin privilegios.

Variables del servicio:

| Variable | Valor inicial | Uso |
|---|---|---|
| `TRANSLATOR_THREADS` | `2` | Hilos de CPU de ONNX por sesión; admite 1 a 16. |
| `TRANSLATOR_TIMEOUT_SECONDS` | `10` | Plazo HTTP; admite más de 0 y hasta 120 segundos. Coordinarlo con el plazo de ASL-Web y el acuse de la app. |
| `MODEL_PACKAGE_DIR` | `/opt/asl-model` | Ubicación del paquete íntegro dentro de la imagen. |

Los dos ONNX ocupan 1 089 917 491 bytes en disco (aproximadamente 1,02 GiB).
El uso de RAM es mayor al tamaño de los pesos y depende de ONNX Runtime y la
longitud de las frases. Para pruebas, reservar al menos 3 GiB libres y medir
`docker stats` con frases reales antes de fijar límites de producción. No usar
varios workers sin presupuestar una copia adicional del modelo por proceso.

## Variante INT8 para comparación local

La construcción INT8 necesita primero la imagen local `asl-gloss-translator:local`.
Desde la raíz del repositorio, ejecuta en este orden:

```powershell
docker compose -f docker-compose.gloss.yml build asl-gloss-translator
docker compose -f docker-compose.gloss-int8.yml up -d --build --wait
```

El segundo comando genera un paquete independiente a partir de la imagen FP32 y
lo sirve en `http://127.0.0.1:8003`. Cuantiza los pesos de las multiplicaciones
matriciales del encoder y del decoder, incluidos sus subgrafos, mediante INT8
dinámico por canal. Conserva el preprocesamiento y la generación originales.
El paquete FP32 de origen permanece intacto.

La construcción utiliza dos etapas: la primera necesita los pesos FP32 y las
herramientas de cuantización; la imagen final copia solamente el paquete INT8 y
las dependencias de ejecución. No hereda las capas ni los pesos FP32 de la etapa
de construcción. Su `manifest.json` identifica `variant: int8-dynamic`, actualiza
los hashes y tamaños e incluye `provenance` con el hash del manifiesto fuente,
las opciones de cuantización y los operadores observados en ambos modelos.

Consulta [el procedimiento de comparación](benchmarks/README.md) para repetir las
mediciones con las mismas frases. En la evaluación local INT8 redujo recursos y
tiempos, pero introdujo cambios de significado en algunas traducciones. Por ello
no se adoptó como motor habitual: ASL-Web continúa usando FP32 y la variante INT8
queda disponible para evaluación aislada.

## Verificación

Con Python 3.11 y las dependencias de pruebas instaladas:

```powershell
python -m pip install -r ASL-GlossTranslator/requirements-test.txt
python -m unittest discover -s ASL-GlossTranslator/tests -v
```

Las pruebas de contrato sustituyen el traductor para comprobar validación,
errores, saturación y plazos sin cargar los pesos. La comprobación real del
modelo requiere construir el contenedor y enviar frases a `/translate`; las
pruebas con sustitutos no demuestran calidad lingüística del modelo.

Referencias de implementación:
[ciclo de vida de FastAPI](https://fastapi.tiangolo.com/advanced/events/),
[tokenizer del paquete](../model_onnx/asl_translator_fp32/asl_translator_fp32/README.md)
y [administración de hilos de ONNX Runtime](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
