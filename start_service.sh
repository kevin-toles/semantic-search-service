#!/bin/bash
# Start semantic-search-service locally
# 
# Usage: ./start_service.sh [MODE]
#   MODE: docker|hybrid|native (default: auto-detect)
#   --fake: Use fake clients for testing
#
# Examples:
#   ./start_service.sh           # Auto-detect mode
#   ./start_service.sh hybrid    # Explicit hybrid mode
#   ./start_service.sh --fake    # Testing mode (no database)

cd "$(dirname "$0")"

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
elif [ -f venv/bin/activate ]; then
    source venv/bin/activate
else
    echo "ERROR: No virtual environment found. Run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# Load environment from .env file
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a
fi

# Parse arguments
USE_FAKE=false
for arg in "$@"; do
    case $arg in
        --fake)
            USE_FAKE=true
            ;;
        docker|hybrid|native)
            export INFRASTRUCTURE_MODE="$arg"
            ;;
    esac
done

# Configure client mode
if [ "$USE_FAKE" == "true" ]; then
    export USE_FAKE_CLIENTS=true
    echo "Starting with FAKE clients (testing mode - no database required)"
else
    unset USE_FAKE_CLIENTS
    echo "Starting with REAL database clients"
    echo "  Infrastructure mode: ${INFRASTRUCTURE_MODE:-auto-detect}"
fi

echo "Starting semantic-search-service on port ${SEMANTIC_SEARCH_PORT:-8081}..."

# Run with uvicorn
python -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port ${SEMANTIC_SEARCH_PORT:-8081} \
    --reload
