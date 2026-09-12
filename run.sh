#!/usr/bin/env bash
# Lanzador de desarrollo para macOS de ASL-System.

set -Eeuo pipefail

skip_mobile=false
skip_docker=false
skip_ngrok=false
use_nginx_gateway=false
gateway_port=8080
ngrok_port=3001

usage() {
  cat <<'EOF'
Uso: ./run.sh [opciones]

  --skip-mobile                 Omite Expo
  --skip-docker                 Omite MongoDB en Docker (incompatible con --use-nginx-gateway)
  --skip-ngrok                  Omite el tunel ngrok
  --use-nginx-gateway           Inicia el gateway Nginx con Docker Compose
  --gateway-port PUERTO         Puerto del gateway Nginx (por defecto: 8080)
  --ngrok-port PUERTO           Puerto que publicara ngrok (por defecto: 3001)
  -h, --help                    Muestra esta ayuda
EOF
}

while (($#)); do
  case "$1" in
    --skip-mobile) skip_mobile=true ;;
    --skip-docker) skip_docker=true ;;
    --skip-ngrok) skip_ngrok=true ;;
    --use-nginx-gateway) use_nginx_gateway=true ;;
    --gateway-port)
      gateway_port="${2:?Falta el valor de --gateway-port}"
      shift
      ;;
    --ngrok-port)
      ngrok_port="${2:?Falta el valor de --ngrok-port}"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Opcion no reconocida: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

web_app="$repo_root/ASL-Web"
web_server="$web_app/server"
call_app_server="$repo_root/ASL-CallAPP/server"
call_app="$repo_root/ASL-CallAPP/app"
mobile_app="$repo_root/ASL-MobileAPP"

step() { printf '==> %s\n' "$1"; }

require_directory() {
  [[ -d "$1" ]] || { echo "$2 no existe: $1" >&2; exit 1; }
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || { echo "$1 no esta disponible en PATH." >&2; exit 1; }
}

update_path_from_npm() {
  local npm_prefix
  npm_prefix="$(npm config get prefix 2>/dev/null || true)"
  [[ -n "$npm_prefix" && "$npm_prefix" != "undefined" ]] || return
  [[ -d "$npm_prefix/bin" ]] && export PATH="$npm_prefix/bin:$PATH"
  [[ -d "$npm_prefix" ]] && export PATH="$npm_prefix:$PATH"
}

ensure_ngrok() {
  command -v ngrok >/dev/null 2>&1 && return

  step 'ngrok no esta disponible en PATH. Intentando instalarlo automaticamente'
  if command -v brew >/dev/null 2>&1; then
    printf 'Instalando ngrok con Homebrew...\n'
    brew install ngrok/ngrok/ngrok
  elif command -v npm >/dev/null 2>&1; then
    printf 'Instalando ngrok con npm -g...\n'
    npm install --global ngrok
    update_path_from_npm
  else
    echo 'No se encontro Homebrew ni npm para instalar ngrok. Instala uno e intenta de nuevo.' >&2
    exit 1
  fi

  command -v ngrok >/dev/null 2>&1 || { echo 'ngrok no pudo instalarse o no quedo disponible en PATH.' >&2; exit 1; }
  printf 'ngrok instalado y disponible.\n'
}

ensure_env_file() {
  local example="$1" target="$2"
  if [[ ! -f "$target" ]]; then
    [[ -f "$example" ]] || { echo "Falta archivo de ejemplo para crear $target" >&2; exit 1; }
    cp "$example" "$target"
    printf 'Creado %s a partir de .env.example\n' "$target"
  fi
}

start_dev_process() {
  local name="$1" working_directory="$2" process_command="$3"
  step "Abriendo Terminal para $name"
  osascript - "$name" "$working_directory" "$process_command" <<'APPLESCRIPT'
on run argv
  set serviceName to item 1 of argv
  set workingDirectory to item 2 of argv
  set processCommand to item 3 of argv
  set terminalCommand to "printf '\\033]0;" & serviceName & "\\007'; cd " & quoted form of workingDirectory & "; echo " & quoted form of ("[" & serviceName & "] " & processCommand) & "; exec " & processCommand
  tell application "Terminal"
    activate
    do script terminalCommand
  end tell
end run
APPLESCRIPT
}

require_directory "$web_app" 'ASL-Web'
require_directory "$web_server" 'ASL-Web/server'
require_directory "$call_app_server" 'ASL-CallAPP/server'
require_directory "$call_app" 'ASL-CallAPP/app'
require_directory "$mobile_app" 'ASL-MobileAPP'
require_command npm

if [[ "$skip_docker" == false ]]; then
  require_command docker
fi

if [[ "$use_nginx_gateway" == true && "$skip_docker" == true ]]; then
  echo 'No puedes combinar --use-nginx-gateway con --skip-docker. El gateway Nginx se levanta con Docker Compose.' >&2
  exit 2
fi

if [[ "$skip_ngrok" == false ]]; then
  ensure_ngrok
fi

ensure_env_file "$web_server/.env.example" "$web_server/.env"
ensure_env_file "$call_app_server/.env.example" "$call_app_server/.env"
ensure_env_file "$call_app/.env.example" "$call_app/.env"

if [[ "$skip_docker" == false ]]; then
  step 'Levantando MongoDB con Docker Compose'
  docker compose -f "$web_server/compose.yaml" up -d mongodb

  if [[ "$use_nginx_gateway" == true ]]; then
    step "Levantando gateway Nginx en el puerto $gateway_port"
    ASL_GATEWAY_PORT="$gateway_port" docker compose -f "$repo_root/docker-compose.nginx.yml" up -d
  fi
fi

start_dev_process 'ASL-Web Server' "$web_server" 'npm run dev'
start_dev_process 'ASL-CallApp Server' "$call_app_server" 'npm run dev'
start_dev_process 'ASL-Web App' "$web_app" 'npm run dev'
start_dev_process 'ASL-CallApp App' "$call_app" 'npm run dev'

if [[ "$skip_mobile" == false ]]; then
  start_dev_process 'ASL-MobileAPP' "$mobile_app" 'npm start'
fi

if [[ "$skip_ngrok" == false ]]; then
  recommended_ngrok_port=3001
  [[ "$use_nginx_gateway" == true ]] && recommended_ngrok_port="$gateway_port"
  if [[ "$ngrok_port" != "$recommended_ngrok_port" ]]; then
    if [[ "$use_nginx_gateway" == true ]]; then
      echo "Advertencia: con --use-nginx-gateway, el tunel recomendado apunta al gateway Nginx en el puerto $gateway_port." >&2
    else
      echo 'Advertencia: sin --use-nginx-gateway, el tunel usa el backend web actual en 3001.' >&2
    fi
  fi
  start_dev_process 'ngrok' "$repo_root" "ngrok http $ngrok_port"
fi

printf '\nServicios solicitados iniciados.\n'
if [[ "$use_nginx_gateway" == true ]]; then
  echo "Tunel recomendado: publica el gateway Nginx en $gateway_port."
else
  echo 'Tunel recomendado: modo transicion, publica ASL-Web/server en 3001.'
fi
