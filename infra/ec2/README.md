# Pruebas en una instancia EC2

Este despliegue aloja solo ASL-Web/server, ASL-CallAPP/server y ASL-ModelServer.
Las aplicaciones Tauri se instalan en las computadoras; el movil se distribuye
por separado. Una MongoDB autenticada conserva `asl-hotel` y `asl-call` en un
volumen Docker. Los servicios se comunican por nombres DNS internos.

## Preparar la instancia

Usar Ubuntu x86_64 (por ejemplo 24.04), inicialmente 4 GiB RAM y disco EBS
gp3 de 30 GiB; ajustar con el consumo observado. Este archivo no crea recursos
AWS ni garantiza que los cargos esten cubiertos por tus creditos.
En el Security Group permitir SSH 22 solo desde tu IP, HTTP 80 y, para HTTPS,
443. Mantener cerrados 27017, 3001, 3101 y 8000.

Instalar Docker Engine y el plugin Compose segun
https://docs.docker.com/engine/install/ubuntu/ . Usar Compose 2.24.4 o superior
para la opcion HTTPS (`!override`). Comprobar `docker compose version`.
Los comandos siguientes asumen acceso a Docker; usar `sudo docker` si corresponde.

Desde EC2, clonar el repositorio y entrar a la raiz:

```bash
git clone --recurse-submodules https://github.com/Asaph-Velazquez/ASL-System.git
cd ASL-System
test -s ASL-ModelServer/model.onnx
test -s ASL-ModelServer/manifest.json
test -s ASL-Web/server/package-lock.json
test -s ASL-CallAPP/server/package-lock.json
```

Publicar o copiar estos nuevos archivos a EC2 antes de ejecutar Compose. Si el
modelo se almacena mediante Git LFS, descargarlo con Git LFS: un puntero no
es un modelo ONNX valido. Los Dockerfiles necesitan ambos package-lock.json;
si faltan en el checkout, generarlos con `npm install --package-lock-only`
en cada carpeta server antes de construir las imagenes.

## Configurar y arrancar

```bash
cp infra/ec2/.env.example infra/ec2/.env
chmod 600 infra/ec2/.env
openssl rand -hex 32
```

Ejecutar el ultimo comando cinco veces. Editar `infra/ec2/.env` y colocar un
valor distinto en MONGO_PASSWORD, JWT_SECRET, CALL_JWT_SECRET,
INTERPRETER_JWT_SECRET y CALL_INTERNAL_TOKEN. No subir ese archivo a Git.
Compose comparte automaticamente los secretos que deben coincidir.
El usuario `asl` es administrador de MongoDB: esta simplificacion es para
pruebas, con MongoDB sin puerto publico. Para produccion separar usuarios y permisos.

```bash
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml config --quiet
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml up -d --build --wait --wait-timeout 240
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml ps
curl --fail http://localhost/gateway-health
curl --fail http://localhost/api/health
```

No usar `config` sin `--quiet` para compartir resultados: muestra secretos.
HTTP permite comprobar arranque con `http://IP_PUBLICA`, pero para login remoto
y pruebas de los clientes usar HTTPS. No se crean cuentas automaticamente.
Para crear un administrador con contrasena propia, usar el modelo existente
(el hook del modelo genera el hash):

```bash
read -r -p 'Usuario administrador: ' ADMIN_USERNAME
read -r -s -p 'Contrasena: ' ADMIN_PASSWORD
printf '\n'
export ADMIN_USERNAME ADMIN_PASSWORD
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml exec -T -e ADMIN_USERNAME -e ADMIN_PASSWORD asl-web-server node --input-type=module -e 'import mongoose from "mongoose"; import {StaffUser} from "./models/index.js"; try { await mongoose.connect(process.env.MONGODB_URI); await StaffUser.create({username:process.env.ADMIN_USERNAME,password:process.env.ADMIN_PASSWORD,fullName:"Administrator",role:"admin"}); } finally { await mongoose.disconnect(); }'
unset ADMIN_USERNAME ADMIN_PASSWORD
```

Crear los interpretes desde Staff Management con rol Interpreter.

## HTTPS con dominio

Apuntar un registro DNS A a la IP publica de EC2 y completar PUBLIC_HOST
(sin `https://`) y ACME_EMAIL en `infra/ec2/.env`. Caddy obtiene y renueva el
certificado y redirige HTTP a HTTPS. Abrir 80 y 443 en el Security Group.
La opcion reemplaza la publicacion de Nginx por localhost:8080.

```bash
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml -f infra/ec2/docker-compose.https.yml config --quiet
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml -f infra/ec2/docker-compose.https.yml up -d --build --wait --wait-timeout 240
curl --fail https://api.TU-DOMINIO/gateway-health
curl --fail https://api.TU-DOMINIO/api/health
```

Para comandos posteriores con HTTPS, incluir ambos `-f`. Conservar los volumenes
de Caddy para mantener certificados. Si cambia la IP al reiniciar EC2, actualizar
DNS. HTTPS no configura TURN: audio/video entre algunas redes puede necesitar
un servidor TURN; este Compose solo incluye senalizacion WebSocket.

## Configurar los clientes

Antes de reconstruir los instaladores, editar los archivos `.env.desktop.local`:

ASL-Web:

```dotenv
VITE_API_URL=https://api.TU-DOMINIO
VITE_WS_URL=wss://api.TU-DOMINIO/ws/hotel
```

ASL-CallAPP/app:

```dotenv
VITE_CALL_API_URL=https://api.TU-DOMINIO
VITE_CALL_WS_URL=wss://api.TU-DOMINIO/calls
```

Ejecutar `npm run desktop:build` en cada frontend desde Windows. Las URLs quedan
integradas al compilar; los instaladores actuales que usan localhost no conectan
a EC2. Los origins Tauri ya estan permitidos. Para un frontend en navegador,
agregar su origin exacto a ALLOWED_ORIGINS.

En ASL-MobileAPP configurar `EXPO_PUBLIC_PUBLIC_BASE_URL=https://api.TU-DOMINIO`
y retirar overrides antiguos de EXPO_PUBLIC_API_URL/EXPO_PUBLIC_WS_URL.
Reconstruir el cliente movil cuando corresponda. No incluir secretos de servidor.

## Operacion y persistencia

```bash
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml logs --tail 100
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml exec nginx-gateway nginx -t
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml exec asl-model-server python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health').read().decode())"
```

Backup (agregar el segundo `-f` si se usa HTTPS):

```bash
docker compose --env-file infra/ec2/.env -f docker-compose.ec2.yml exec -T mongodb sh -c 'exec mongodump --username "$MONGO_INITDB_ROOT_USERNAME" --password "$MONGO_INITDB_ROOT_PASSWORD" --authenticationDatabase admin --archive --gzip' > mongodb-backup.archive.gz
```

El backup contiene datos y hashes: guardarlo fuera de EC2 de forma privada.
`docker compose ... down` conserva los volumenes; `down -v` borra las bases.
La contrasena inicial de MongoDB solo se aplica al crear un volumen vacio:
cambiar MONGO_PASSWORD en `.env` no rota la contrasena de una base existente.
No borrar el volumen para solucionar un error de autenticacion.
Los volumenes residen en el disco de EC2; perder ese disco pierde los datos.

Detener EC2 cuando no se use reduce consumo de computo; EBS y otros recursos
pueden seguir generando cargos. No hay ALB ni NAT Gateway en esta configuracion.
Verificar login de staff e interprete, sesion del huesped, solicitudes en vivo,
inferencia y una llamada con reporte desde dispositivos reales antes de compartir.
