#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
RUNTIME_DIR="$PROJECT_ROOT/.run"
ROOT_REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_BIN="$VENV_DIR/bin/python3"
PIP_BIN="$VENV_DIR/bin/pip"

mkdir -p "$RUNTIME_DIR"

confirm_install() {
  local message="$1"
  if [[ ! -t 0 ]]; then
    echo "$message" >&2
    echo "Non-interactive shell detected, so installation cannot be confirmed automatically." >&2
    return 1
  fi

  while true; do
    read -r -p "$message [y/N]: " reply
    case "${reply:-n}" in
      [Yy]|[Yy][Ee][Ss]) return 0 ;;
      [Nn]|[Nn][Oo]|"") return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

ensure_python_venv() {
  if [[ -x "$PYTHON_BIN" ]]; then
    return 0
  fi

  echo "Project virtual environment not found at .venv."
  echo
  if ! confirm_install "Create .venv and install Python dependencies there?"; then
    echo "Cannot start backend without a project virtual environment." >&2
    exit 1
  fi

  (
    cd "$PROJECT_ROOT"
    python3 -m venv "$VENV_DIR"
  )

  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Failed to create virtual environment at $VENV_DIR" >&2
    exit 1
  fi
}

python_dependency_status() {
  "$PYTHON_BIN" - <<'PY'
import importlib
required = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pandas", "pandas"),
    ("joblib", "joblib"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
]
missing = []
for module_name, package_name in required:
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append(package_name)
if missing:
    print("\n".join(missing))
PY
}

ensure_python_dependencies() {
  local missing
  missing="$(python_dependency_status)"
  if [[ -z "$missing" ]]; then
    return 0
  fi

  echo "Missing Python dependencies detected for backend / machine learning:"
  echo "$missing" | sed 's/^/- /'
  echo
  if ! confirm_install "Install required Python dependencies using .venv/bin/pip install -r requirements.txt?"; then
    echo "Cannot start backend without required Python dependencies." >&2
    exit 1
  fi

  (
    cd "$PROJECT_ROOT"
    "$PIP_BIN" install -r "$ROOT_REQUIREMENTS"
  )

  missing="$(python_dependency_status)"
  if [[ -n "$missing" ]]; then
    echo "Python dependencies are still missing after installation:" >&2
    echo "$missing" | sed 's/^/- /' >&2
    exit 1
  fi
}

ensure_frontend_dependencies() {
  if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "Missing frontend dependencies detected: frontend/node_modules"
    echo
    if ! confirm_install "Install required frontend dependencies using npm install?"; then
      echo "Cannot start frontend without frontend dependencies." >&2
      exit 1
    fi
    (cd "$FRONTEND_DIR" && npm install)
  fi
}

is_port_in_use() {
  local port="$1"
  lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
}

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Freeing port $port..."
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

choose_ports() {
  local preferred_backend=7330
  local preferred_frontend=7331
  local fallback_backend=9519
  local fallback_frontend=9520

  if ! is_port_in_use "$preferred_backend" && ! is_port_in_use "$preferred_frontend"; then
    BACKEND_PORT="$preferred_backend"
    FRONTEND_PORT="$preferred_frontend"
    return
  fi

  if ! is_port_in_use "$fallback_backend" && ! is_port_in_use "$fallback_frontend"; then
    BACKEND_PORT="$fallback_backend"
    FRONTEND_PORT="$fallback_frontend"
    return
  fi

  kill_port "$preferred_backend"
  kill_port "$preferred_frontend"
  BACKEND_PORT="$preferred_backend"
  FRONTEND_PORT="$preferred_frontend"
}

wait_for_http() {
  local url="$1"
  local label="$2"
  for _ in $(seq 1 60); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "$label did not become ready: $url" >&2
  return 1
}

cleanup() {
  local exit_code=$?
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit "$exit_code"
}

trap cleanup INT TERM EXIT

choose_ports
ensure_python_venv
ensure_python_dependencies
ensure_frontend_dependencies

BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}"
FRONTEND_URL="http://127.0.0.1:${FRONTEND_PORT}"

BACKEND_LOG="$RUNTIME_DIR/backend.log"
FRONTEND_LOG="$RUNTIME_DIR/frontend.log"

: >"$BACKEND_LOG"
: >"$FRONTEND_LOG"

echo "Starting backend on $BACKEND_URL"
(
  cd "$PROJECT_ROOT"
  PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  BACKEND_HOST="127.0.0.1" \
  BACKEND_PORT="$BACKEND_PORT" \
  BACKEND_RELOAD="true" \
  "$PYTHON_BIN" backend/run.py
) >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!

wait_for_http "$BACKEND_URL/api/v1/health" "Backend"

echo "Starting frontend on $FRONTEND_URL"
(
  cd "$FRONTEND_DIR"
  VITE_PROXY_TARGET="$BACKEND_URL" \
  npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort
) >"$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!

wait_for_http "$FRONTEND_URL" "Frontend"

echo
echo "Backend live URL : $BACKEND_URL"
echo "Frontend live URL: $FRONTEND_URL"
echo
echo "Backend log : $BACKEND_LOG"
echo "Frontend log: $FRONTEND_LOG"
echo
echo "Press Ctrl+C to stop both services."

wait "$BACKEND_PID" "$FRONTEND_PID"
