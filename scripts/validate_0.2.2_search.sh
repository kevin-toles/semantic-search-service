#!/bin/bash
# =============================================================================
# WBS 0.2.2 Validation Script: /v1/search Endpoint
# Reference: END_TO_END_INTEGRATION_WBS.md
# =============================================================================

set -e

SEMANTIC_SEARCH_URL="${SEMANTIC_SEARCH_URL:-http://localhost:8081}"
PASSED=0
FAILED=0

echo "=========================================="
echo "WBS 0.2.2 Validation: /v1/search Endpoint"
echo "=========================================="
echo "Semantic Search URL: $SEMANTIC_SEARCH_URL"
echo ""

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

# -----------------------------------------------------------------------------
# Test 1: Endpoint exists (HTTP 200)
# -----------------------------------------------------------------------------
echo ""
echo "Test 1: POST /v1/search returns HTTP 200"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$SEMANTIC_SEARCH_URL/v1/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "test search query"}')
check_result "Endpoint returns HTTP 200" "$([ "$HTTP_CODE" == "200" ] && echo 'true' || echo 'false')"

# -----------------------------------------------------------------------------
# Test 2: Returns results array
# -----------------------------------------------------------------------------
echo ""
echo "Test 2: Response contains results array"
RESPONSE=$(curl -s -X POST "$SEMANTIC_SEARCH_URL/v1/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "test"}')
HAS_RESULTS=$(echo "$RESPONSE" | jq 'has("results") and (.results | type == "array")')
check_result "Response has 'results' array" "$HAS_RESULTS"

# -----------------------------------------------------------------------------
# Test 3: Results have id and score fields
# -----------------------------------------------------------------------------
echo ""
echo "Test 3: Results have id and score (if any)"
# Note: Empty results is valid for empty collection
RESULTS_VALID=$(echo "$RESPONSE" | jq '
    .results == [] or 
    (.results | all(has("id") and has("score")))
')
check_result "Results have id/score fields" "$RESULTS_VALID"

# -----------------------------------------------------------------------------
# Test 4: Limit parameter works
# -----------------------------------------------------------------------------
echo ""
echo "Test 4: Limit parameter is respected"
RESPONSE=$(curl -s -X POST "$SEMANTIC_SEARCH_URL/v1/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "limit": 3}')
LIMIT_WORKS=$(echo "$RESPONSE" | jq '.results | length <= 3')
check_result "Limit parameter works" "$LIMIT_WORKS"

# -----------------------------------------------------------------------------
# Test 5: Score in range [0, 1]
# -----------------------------------------------------------------------------
echo ""
echo "Test 5: Score values are in [0, 1] range"
SCORE_VALID=$(echo "$RESPONSE" | jq '
    .results == [] or 
    (.results | all(.score >= 0 and .score <= 1))
')
check_result "Scores in valid range" "$SCORE_VALID"

# -----------------------------------------------------------------------------
# Test 6: Response contains required metadata
# -----------------------------------------------------------------------------
echo ""
echo "Test 6: Response contains required metadata"
RESPONSE=$(curl -s -X POST "$SEMANTIC_SEARCH_URL/v1/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "metadata test"}')
HAS_METADATA=$(echo "$RESPONSE" | jq 'has("total") and has("query") and has("latency_ms")')
check_result "Response has total/query/latency_ms" "$HAS_METADATA"

# -----------------------------------------------------------------------------
# Test 7: Validation - empty query rejected
# -----------------------------------------------------------------------------
echo ""
echo "Test 7: Empty query is rejected (422)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$SEMANTIC_SEARCH_URL/v1/search" \
    -H "Content-Type: application/json" \
    -d '{"query": ""}')
check_result "Empty query returns 422" "$([ "$HTTP_CODE" == "422" ] && echo 'true' || echo 'false')"

# -----------------------------------------------------------------------------
# Test 8: Validation - invalid limit rejected
# -----------------------------------------------------------------------------
echo ""
echo "Test 8: Invalid limit is rejected (422)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$SEMANTIC_SEARCH_URL/v1/search" \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "limit": 0}')
check_result "Zero limit returns 422" "$([ "$HTTP_CODE" == "422" ] && echo 'true' || echo 'false')"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "WBS 0.2.2 Validation Summary"
echo "=========================================="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ All WBS 0.2.2 acceptance tests passed!"
    exit 0
else
    echo "❌ Some tests failed. Please review and fix."
    exit 1
fi
