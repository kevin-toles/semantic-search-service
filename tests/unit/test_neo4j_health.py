"""
TDD RED Phase: Neo4j Health Check Tests

These tests define the expected behavior for Neo4j connectivity verification.
The implementation does not exist yet - these tests should FAIL initially.

WBS Reference: 1.4, 1.4.1
Acceptance Criteria: Neo4j health check returns True when connected, False otherwise
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Settings


class TestNeo4jHealthCheck:
    """Tests for Neo4j connectivity and health verification."""

    def test_neo4j_health_check_returns_true_when_connected(
        self, settings: Settings, mock_neo4j_driver
    ):
        """
        GIVEN a running Neo4j instance
        WHEN check_neo4j_health is called
        THEN it returns True
        """
        # Import will fail until implementation exists (RED phase)
        from src.graph.health import check_neo4j_health

        with patch("src.graph.health.get_neo4j_driver", return_value=mock_neo4j_driver):
            result = check_neo4j_health(settings)

        assert result is True

    def test_neo4j_health_check_returns_false_when_disconnected(self, settings: Settings):
        """
        GIVEN Neo4j is not reachable
        WHEN check_neo4j_health is called
        THEN it returns False
        """
        from src.graph.health import check_neo4j_health

        mock_driver = MagicMock()
        mock_driver.verify_connectivity.side_effect = Exception("Connection refused")

        with patch("src.graph.health.get_neo4j_driver", return_value=mock_driver):
            result = check_neo4j_health(settings)

        assert result is False

    def test_neo4j_health_check_returns_connection_details(
        self, settings: Settings, mock_neo4j_driver
    ):
        """
        GIVEN a running Neo4j instance
        WHEN check_neo4j_health_detailed is called
        THEN it returns a dict with status, url, and latency
        """
        from src.graph.health import check_neo4j_health_detailed

        with patch("src.graph.health.get_neo4j_driver", return_value=mock_neo4j_driver):
            result = check_neo4j_health_detailed(settings)

        assert "status" in result
        assert "url" in result
        assert result["status"] in ("healthy", "unhealthy")


class TestNeo4jDriver:
    """Tests for Neo4j driver initialization."""

    def test_get_neo4j_driver_creates_driver_with_settings(self, settings: Settings):
        """
        GIVEN valid Neo4j settings
        WHEN get_neo4j_driver is called
        THEN it returns a configured driver instance
        """
        from src.graph.health import get_neo4j_driver

        with patch("src.graph.health.GraphDatabase") as mock_gdb:
            mock_gdb.driver.return_value = MagicMock()
            driver = get_neo4j_driver(settings)

            mock_gdb.driver.assert_called_once_with(
                settings.neo4j_url,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
            assert driver is not None

    def test_get_neo4j_driver_raises_on_invalid_url(self, settings: Settings):
        """
        GIVEN invalid Neo4j URL
        WHEN get_neo4j_driver is called
        THEN it raises ConnectionError
        """
        from src.graph.health import get_neo4j_driver

        settings.neo4j_url = "invalid://bad-url"

        with pytest.raises(ConnectionError):
            get_neo4j_driver(settings)
