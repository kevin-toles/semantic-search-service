# ==============================================================================
# Semantic Search Service - Dockerfile
# ==============================================================================
# Reference: GRAPH_RAG_POC_PLAN.md Phase 7 (WBS 7.6)
# Multi-stage build for production deployment
# ==============================================================================

# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder to system-wide location
COPY --from=builder /root/.local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SEMANTIC_SEARCH_PORT=8081

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:${SEMANTIC_SEARCH_PORT}/health || exit 1

# Expose port
EXPOSE 8081

# Start the service with multiple workers for higher concurrency
# Use 4 workers to handle ~150-200 concurrent users
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8081", "--workers", "4"]
