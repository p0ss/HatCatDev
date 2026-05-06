#!/usr/bin/env bash
#
# Start Chorion inside the distrobox.
#
# Usage:
#   bash src/chorion/start.sh              # normal start
#   bash src/chorion/start.sh --reload     # dev mode with auto-reload
#
set -euo pipefail

DISTROBOX_NAME="chorion"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env.chorion"
EXTRA_ARGS="${*:-}"

# Check distrobox exists
if ! distrobox list 2>/dev/null | grep -q "$DISTROBOX_NAME"; then
    echo "Error: Distrobox '$DISTROBOX_NAME' not found. Run setup.sh first."
    exit 1
fi

# Check env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found. Run setup.sh first."
    exit 1
fi

# Load env vars
set -a
source "$ENV_FILE"
set +a

echo "Starting Chorion on ${CHORION_HOST:-127.0.0.1}:${CHORION_PORT:-8901}..."

# Run inside distrobox
distrobox enter "$DISTROBOX_NAME" -- bash -c "
    cd '$PROJECT_ROOT'

    # Export env vars
    export CHORION_SERVICE_SECRET='${CHORION_SERVICE_SECRET:-}'
    export MONGO_URI='${MONGO_URI:-mongodb://localhost:27017}'
    export CHORION_DB_NAME='${CHORION_DB_NAME:-LibreChat}'

    # Run via poetry — set PYTHONPATH so 'chorion' is importable
    export PYTHONPATH='$PROJECT_ROOT/src:\${PYTHONPATH:-}'
    poetry run python -m chorion.server \
        --host '${CHORION_HOST:-127.0.0.1}' \
        --port '${CHORION_PORT:-8901}' \
        $EXTRA_ARGS
"
