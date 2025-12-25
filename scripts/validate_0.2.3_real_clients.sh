#!/bin/bash
# =============================================================================
# WBS 0.2.3 Validation Script: Real Database Clients
# Reference: END_TO_END_INTEGRATION_WBS.md
# =============================================================================

set -e

echo "=========================================="
echo "WBS 0.2.3 Validation: Real Database Clients"
echo "=========================================="

PASSED=0
FAILED=0

# Helper function for test results
check_result() {
    local test_name="$1"
    local condition="$2"
    if [ "$condition" == "true" ]; then
        echo "✅ PASS: $test_name"
        PASSED=$((PASSED + 1))
    else
        echo "❌ FAIL: $test_name"
        FAILED=$((FAILED + 1))
    fi
}

# Ensure infrastructure is running
echo ""
echo "Checking infrastructure..."
docker ps | grep -q "integration-qdrant" || (echo "✗ Start docker-compose.integration.yml first" && exit 1)
docker ps | grep -q "integration-neo4j" || (echo "✗ Start docker-compose.integration.yml first" && exit 1)
echo "✓ Infrastructure running"

# Kill any existing local uvicorn instance
echo ""
echo "Cleaning up existing instances..."
pkill -f "uvicorn src.main:app.*8091" 2>/dev/null || true
sleep 2

# Start with real clients
echo ""
echo "1. Starting semantic-search with REAL clients on port 8091..."
cd /Users/kevintoles/POC/semantic-search-service

export USE_REAL_CLIENTS=true
export QDRANT_URL=http://localhost:6333
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=devpassword
export EMBEDDING_MODEL=all-mpnet-base-v2

# Start uvicorn in background
START_TIME=$(date +%s)
uvicorn src.main:app --host 0.0.0.0 --port 8091 --log-level info &
APP_PID=$!

# Wait for service to start (model loading takes time)
echo "   Waiting for service to start (model loading)..."
for i in {1..30}; do
    if curl -s http://localhost:8091/health > /dev/null 2>&1; then
        END_TIME=$(date +%s)
        STARTUP_TIME=$((END_TIME - START_TIME))
        echo "   Service started in ${STARTUP_TIME}s"
        break
    fi
    sleep 1
done

# Test 1: App starts with real clients
echo ""
echo "Test 1: App starts with real clients (HTTP 200)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8091/health 2>/dev/null || echo "000")
check_result "Service returns HTTP 200" "$([ "$HTTP_CODE" == "200" ] && echo 'true' || echo 'false')"

# Test 2: Get health response
echo ""
echo "Test 2: Health endpoint returns valid response"
HEALTH=$(curl -s http://localhost:8091/health 2>/dev/null || echo '{}')
echo "   Health response: $HEALTH"
HAS_DEPS=$(echo "$HEALTH" | jq 'has("dependencies")' 2>/dev/null || echo 'false')
check_result "Response has dependencies field" "$HAS_DEPS"

# Test 3: Qdrant connected
echo ""
echo "Test 3: Qdrant connection status"
QDRANT_STATUS=$(echo "$HEALTH" | jq -r '.dependencies.qdrant // "unknown"' 2>/dev/null)
echo "   Qdrant status: $QDRANT_STATUS"
check_result "Qdrant connected" "$([ "$QDRANT_STATUS" == "connected" ] && echo 'true' || echo 'false')"

# Test 4: Neo4j connected
echo ""
echo "Test 4: Neo4j connection status"
NEO4J_STATUS=$(echo "$HEALTH" | jq -r '.dependencies.neo4j // "unknown"' 2>/dev/null)
echo "   Neo4j status: $NEO4J_STATUS"
check_result "Neo4j connected" "$([ "$NEO4J_STATUS" == "connected" ] && echo 'true' || echo 'false')"

# Test 5: Embedder loaded
echo ""
echo "Test 5: Embedder status"
EMBEDDER_STATUS=$(echo "$HEALTH" | jq -r '.dependencies.embedder // "unknown"' 2>/dev/null)
echo "   Embedder status: $EMBEDDER_STATUS"
check_result "Embedder loaded" "$([ "$EMBEDDER_STATUS" == "loaded" ] && echo 'true' || echo 'false')"

# Test 6: Startup time < 30s
echo ""
echo "Test 6: Startup time < 30s"
echo "   Startup time: ${STARTUP_TIME}s"
check_result "Startup under 30s" "$([ "$STARTUP_TIME" -lt 30 ] && echo 'true' || echo 'false')"

# Test 7: Real embedding works (768 dimensions)
echo ""
echo "Test 7: Real embedder produces 768-dim vectors"
EMBED_RESP=$(curl -s -X POST http://localhost:8091/v1/embed \
    -H "Content-Type: application/json" \
    -d '{"text": "integration test for real embedding"}' 2>/dev/null || echo '{}')
DIMS=$(echo "$EMBED_RESP" | jq '.dimensions // 0' 2>/dev/null)
echo "   Embedding dimensions: $DIMS"
check_result "Embedding has 768 dimensions" "$([ "$DIMS" == "768" ] && echo 'true' || echo 'false')"

# Cleanup
echo ""
echo "Cleaning up..."
kill $APP_PID 2>/dev/null || true

# Summary
echo ""
echo "=========================================="
echo "WBS 0.2.3 Validation Summary"
echo "=========================================="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All WBS 0.2.3 acceptance tests passed!"
    exit 0
else
    echo "❌ Some tests failed. Please review."
    exit 1
fi
