"""
Neo4j health check and driver utilities.

Provides connectivity verification and driver initialization for Neo4j graph database.
"""

import time
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

from src.core.config import Settings


def get_neo4j_driver(settings: Settings) -> Any:
    """
    Create and return a Neo4j driver instance.

    Args:
        settings: Application settings with Neo4j configuration

    Returns:
        Neo4j driver instance

    Raises:
        ConnectionError: If URL is invalid or connection cannot be established
    """
    if not settings.neo4j_url.startswith(("bolt://", "neo4j://", "neo4j+s://")):
        raise ConnectionError(f"Invalid Neo4j URL scheme: {settings.neo4j_url}")

    try:
        driver = GraphDatabase.driver(
            settings.neo4j_url,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        return driver
    except Exception as e:
        raise ConnectionError(f"Failed to create Neo4j driver: {e}") from e


def check_neo4j_health(settings: Settings) -> bool:
    """
    Check if Neo4j is healthy and reachable.

    Args:
        settings: Application settings with Neo4j configuration

    Returns:
        True if Neo4j is healthy, False otherwise
    """
    try:
        driver = get_neo4j_driver(settings)
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


def check_neo4j_health_detailed(settings: Settings) -> dict[str, Any]:
    """
    Check Neo4j health with detailed information.

    Args:
        settings: Application settings with Neo4j configuration

    Returns:
        Dictionary with status, url, and additional health details
    """
    start_time = time.time()
    try:
        driver = get_neo4j_driver(settings)
        driver.verify_connectivity()

        # Run a simple query to verify database responsiveness
        with driver.session() as session:
            session.run("RETURN 1").single()

        latency_ms = (time.time() - start_time) * 1000
        driver.close()

        return {
            "status": "healthy",
            "url": settings.neo4j_url,
            "latency_ms": round(latency_ms, 2),
        }
    except ServiceUnavailable as e:
        return {
            "status": "unhealthy",
            "url": settings.neo4j_url,
            "error": f"Service unavailable: {e}",
        }
    except AuthError as e:
        return {
            "status": "unhealthy",
            "url": settings.neo4j_url,
            "error": f"Authentication failed: {e}",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "url": settings.neo4j_url,
            "error": str(e),
        }
