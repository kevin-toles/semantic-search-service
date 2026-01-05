"""
Infrastructure-aware configuration for semantic-search-service.

This module provides dynamic URL resolution based on INFRASTRUCTURE_MODE.
NO HARDCODED VALUES - URLs are determined at runtime based on mode.

Modes:
  - docker: All services in Docker (uses Docker DNS names)
  - hybrid: Native Qdrant+Redis, Docker Neo4j (uses localhost)
  - native: All native (uses localhost)

Usage:
    from src.infrastructure_config import get_infrastructure_urls
    
    urls = get_infrastructure_urls()
    qdrant_url = urls["QDRANT_URL"]
    neo4j_uri = urls["NEO4J_URI"]
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Docker DNS names (used when running inside Docker containers)
DOCKER_HOSTNAMES = {
    "neo4j": "ai-platform-neo4j",
    "qdrant": "ai-platform-qdrant", 
    "redis": "ai-platform-redis",
}

# Default ports
DEFAULT_PORTS = {
    "neo4j_bolt": 7687,
    "neo4j_http": 7474,
    "qdrant": 6333,
    "redis": 6379,
}


def _detect_running_in_docker() -> bool:
    """Detect if we're running inside a Docker container.
    
    Checks for /.dockerenv file or cgroup hints.
    """
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            return "docker" in f.read()
    except FileNotFoundError:
        return False


def get_infrastructure_mode() -> str:
    """Get the current infrastructure mode.
    
    Priority:
    1. INFRASTRUCTURE_MODE environment variable (set by Platform Control)
    2. Auto-detect if running in Docker
    3. Default to 'hybrid' (most common development mode)
    
    Returns:
        One of: 'docker', 'hybrid', 'native'
    """
    # Check explicit env var first
    mode = os.environ.get("INFRASTRUCTURE_MODE", "").lower()
    if mode in ("docker", "hybrid", "native"):
        logger.info("Infrastructure mode from env: %s", mode)
        return mode
    
    # Auto-detect Docker container
    if _detect_running_in_docker():
        logger.info("Auto-detected: running in Docker container")
        return "docker"
    
    # Default to hybrid for local development
    logger.info("Using default infrastructure mode: hybrid")
    return "hybrid"


def get_infrastructure_urls(mode: str | None = None) -> dict[str, str]:
    """Get infrastructure URLs based on mode.
    
    Args:
        mode: Optional mode override. If None, auto-detects.
        
    Returns:
        Dict with keys:
          - QDRANT_URL: Full URL for Qdrant
          - NEO4J_URI: Bolt URI for Neo4j
          - NEO4J_HOST: Just the hostname for Neo4j
          - REDIS_URL: Full URL for Redis
    """
    current_mode = mode or get_infrastructure_mode()
    
    if current_mode == "docker":
        # All Docker: use Docker DNS names
        # This is for when semantic-search-service itself runs in Docker
        urls = {
            "QDRANT_URL": f"http://{DOCKER_HOSTNAMES['qdrant']}:{DEFAULT_PORTS['qdrant']}",
            "NEO4J_URI": f"bolt://{DOCKER_HOSTNAMES['neo4j']}:{DEFAULT_PORTS['neo4j_bolt']}",
            "NEO4J_HOST": DOCKER_HOSTNAMES["neo4j"],
            "REDIS_URL": f"redis://{DOCKER_HOSTNAMES['redis']}:{DEFAULT_PORTS['redis']}",
        }
    else:
        # Hybrid or Native: services run natively, connect via localhost
        # Even in hybrid mode (Docker Neo4j), ports are exposed to localhost
        urls = {
            "QDRANT_URL": f"http://localhost:{DEFAULT_PORTS['qdrant']}",
            "NEO4J_URI": f"bolt://localhost:{DEFAULT_PORTS['neo4j_bolt']}",
            "NEO4J_HOST": "localhost",
            "REDIS_URL": f"redis://localhost:{DEFAULT_PORTS['redis']}",
        }
    
    logger.info("Infrastructure URLs for mode '%s': %s", current_mode, urls)
    return urls


def get_environment_with_urls() -> dict[str, str]:
    """Get full environment dict with infrastructure URLs populated.
    
    This merges:
    1. Existing environment variables
    2. Dynamic infrastructure URLs (if not explicitly set)
    
    Returns:
        Dict of environment variable name -> value
    """
    urls = get_infrastructure_urls()
    
    # Start with what's explicitly set in environment
    result = {}
    
    # For each URL, use env var if set, otherwise use dynamic value
    for key, dynamic_value in urls.items():
        env_value = os.environ.get(key)
        if env_value:
            result[key] = env_value
            logger.debug("%s from env: %s", key, env_value)
        else:
            result[key] = dynamic_value
            logger.debug("%s from dynamic: %s", key, dynamic_value)
    
    return result


if __name__ == "__main__":
    # Test the configuration
    import sys
    logging.basicConfig(level=logging.INFO)
    
    mode = get_infrastructure_mode()
    print(f"Auto-detected mode: {mode}")
    print()
    
    for test_mode in ["docker", "hybrid", "native"]:
        urls = get_infrastructure_urls(test_mode)
        print(f"URLs for '{test_mode}':")
        for k, v in urls.items():
            print(f"  {k}: {v}")
        print()
