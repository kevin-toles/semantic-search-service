#!/bin/bash
# Start semantic-search-service locally
# Usage: ./start_service.sh [--real]

cd "$(dirname "$0")"

# Load environment
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check for --real flag to use real database clients
if [ "$1" == "--real" ]; then
    export USE_REAL_CLIENTS=true
    echo "Starting with REAL database clients"
    echo "  Neo4j: $NEO4J_URL"
    echo "  Qdrant: $QDRANT_URL"
else
    export USE_REAL_CLIENTS=false
    echo "Starting with FAKE clients (no database required)"
fi

echo "Starting semantic-search-service on port ${SEMANTIC_SEARCH_PORT:-8081}..."

# Run with uvicorn
python -m uvicorn src.main:app \
    --host 0.0.0.0 \
    --port ${SEMANTIC_SEARCH_PORT:-8081} \
    --reload
