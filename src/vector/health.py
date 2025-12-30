"""
Qdrant health check and client utilities.

Provides connectivity verification and client initialization for Qdrant vector database.
"""

import time
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from src.core.config import Settings


def get_qdrant_client(settings: Settings) -> QdrantClient:
    """
    Create and return a Qdrant client instance.

    Args:
        settings: Application settings with Qdrant configuration

    Returns:
        Qdrant client instance

    Raises:
        ConnectionError: If URL is invalid or connection cannot be established
    """
    # SECURITY: Allows both http:// and https:// intentionally for local Docker
    # networking (e.g., http://qdrant:6333). Production uses HTTPS via env config.
    # Reviewed and marked SAFE in SonarCloud (S5332).
    if not settings.qdrant_url.startswith(("http://", "https://")):
        raise ConnectionError(f"Invalid Qdrant URL scheme: {settings.qdrant_url}")

    try:
        client = QdrantClient(url=settings.qdrant_url)
        return client
    except Exception as e:
        raise ConnectionError(f"Failed to create Qdrant client: {e}") from e


def check_qdrant_health(settings: Settings) -> bool:
    """
    Check if Qdrant is healthy and reachable.

    Args:
        settings: Application settings with Qdrant configuration

    Returns:
        True if Qdrant is healthy, False otherwise
    """
    try:
        client = get_qdrant_client(settings)
        # get_collections() is a lightweight operation to verify connectivity
        client.get_collections()
        return True
    except Exception:
        return False


def check_qdrant_health_detailed(settings: Settings) -> dict[str, Any]:
    """
    Check Qdrant health with detailed information.

    Args:
        settings: Application settings with Qdrant configuration

    Returns:
        Dictionary with status, url, and additional health details
    """
    start_time = time.time()
    try:
        client = get_qdrant_client(settings)
        collections = client.get_collections()
        latency_ms = (time.time() - start_time) * 1000

        return {
            "status": "healthy",
            "url": settings.qdrant_url,
            "collections_count": len(collections.collections),
            "latency_ms": round(latency_ms, 2),
        }
    except UnexpectedResponse as e:
        return {
            "status": "unhealthy",
            "url": settings.qdrant_url,
            "error": f"Unexpected response: {e}",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "url": settings.qdrant_url,
            "error": str(e),
        }
