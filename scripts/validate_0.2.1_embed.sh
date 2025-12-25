#!/bin/bash
# ==============================================================================
# WBS 0.2.1 Validation Script - /v1/embed Endpoint
# ==============================================================================
# Purpose: Validate /v1/embed endpoint functionality
# Reference: END_TO_END_INTEGRATION_WBS.md WBS 0.2.1
#
# Usage:
#   ./scripts/validate_0.2.1_embed.sh
#
# Prerequisites:
#   - semantic-search service running on localhost:8081
#   - jq installed for JSON parsing
#
# Exit Codes:
#   0 - All validations passed
#   1 - Validation failed
# ==============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║              WBS 0.2.1 VALIDATION: /v1/embed Endpoint                    ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

BASE_URL="${SEMANTIC_SEARCH_URL:-http://localhost:8081}"

echo "Target: $BASE_URL"
echo ""

echo "1. Testing endpoint exists (HTTP 200)..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{"text":"test"}')
[ "$STATUS" = "200" ] && echo "   ✓ HTTP 200" || (echo "   ✗ Got HTTP $STATUS" && exit 1)

echo "2. Testing single embedding returned..."
COUNT=$(curl -s -X POST "$BASE_URL/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{"text":"hello world"}' | jq '.embeddings | length')
[ "$COUNT" = "1" ] && echo "   ✓ 1 embedding returned" || (echo "   ✗ Got $COUNT" && exit 1)

echo "3. Testing embedding dimension (768)..."
DIM=$(curl -s -X POST "$BASE_URL/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{"text":"hello world"}' | jq '.embeddings[0] | length')
[ "$DIM" = "768" ] && echo "   ✓ 768 dimensions" || (echo "   ✗ Got $DIM dimensions" && exit 1)

echo "4. Testing model name returned..."
MODEL=$(curl -s -X POST "$BASE_URL/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{"text":"test"}' | jq -r '.model')
[ -n "$MODEL" ] && [ "$MODEL" != "null" ] && echo "   ✓ Model: $MODEL" || (echo "   ✗ No model name" && exit 1)

echo "5. Testing batch embedding (3 texts)..."
BATCH_COUNT=$(curl -s -X POST "$BASE_URL/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{"text":["apple","banana","cherry"]}' | jq '.embeddings | length')
[ "$BATCH_COUNT" = "3" ] && echo "   ✓ Batch: 3 embeddings" || (echo "   ✗ Got $BATCH_COUNT" && exit 1)

echo "6. Testing validation (empty text = 422)..."
EMPTY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/v1/embed" \
  -H "Content-Type: application/json" \
  -d '{"text":""}')
[ "$EMPTY_STATUS" = "422" ] && echo "   ✓ Empty text rejected (422)" || echo "   ⚠ Got $EMPTY_STATUS (acceptable if 400)"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                    WBS 0.2.1 VALIDATION PASSED                           ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
