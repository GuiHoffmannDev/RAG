#!/bin/bash
# Container entrypoint. SERVICE selects the process: api (default), ui, or all (both in one
# container, for platforms that run a single service). Any explicit command bypasses this logic.
set -euo pipefail

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

bootstrap() {
  if [ "${AUTO_INDEX:-true}" = "true" ]; then
    # A failure must not keep the API down: /ready reports the missing index.
    python -m src.ingestion.bootstrap || echo "WARNING: automatic indexing failed. See the log above." >&2
  fi
}

api() { exec uvicorn src.api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT_INTERNAL:-${PORT:-8000}}" --proxy-headers; }
ui() { exec streamlit run app/main.py --server.address=0.0.0.0 --server.port="${PORT:-8501}" --server.headless=true; }

case "${SERVICE:-api}" in
  api)
    bootstrap
    api
    ;;
  ui)
    ui
    ;;
  all)
    bootstrap
    export API_URL="${API_URL:-http://127.0.0.1:8000}"
    API_HOST=127.0.0.1 API_PORT_INTERNAL=8000 api &
    ui &
    trap 'kill $(jobs -p) 2>/dev/null || true' TERM INT
    # Exit as soon as either process dies, so the platform restarts the container.
    wait -n
    status=$?
    kill $(jobs -p) 2>/dev/null || true
    exit "$status"
    ;;
  *)
    echo "Unknown SERVICE '${SERVICE}'. Use api, ui or all." >&2
    exit 64
    ;;
esac
