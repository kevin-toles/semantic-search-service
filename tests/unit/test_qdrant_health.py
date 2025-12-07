"""
TDD RED Phase: Qdrant Health Check Tests

These tests define the expected behavior for Qdrant connectivity verification.
The implementation does not exist yet - these tests should FAIL initially.

WBS Reference: 1.5, 1.5.1
Acceptance Criteria: Qdrant health check returns True when connected, False otherwise
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Settings


class TestQdrantHealthCheck:
    """Tests for Qdrant connectivity and health verification."""

    def test_qdrant_health_check_returns_true_when_connected(
        self, settings: Settings, mock_qdrant_client
    ):
        """
        GIVEN a running Qdrant instance
        WHEN check_qdrant_health is called
        THEN it returns True
        """
        from src.vector.health import check_qdrant_health

        with patch("src.vector.health.get_qdrant_client", return_value=mock_qdrant_client):
            result = check_qdrant_health(settings)

        assert result is True

    def test_qdrant_health_check_returns_false_when_disconnected(self, settings: Settings):
        """
        GIVEN Qdrant is not reachable
        WHEN check_qdrant_health is called
        THEN it returns False
        """
        from src.vector.health import check_qdrant_health

        mock_client = MagicMock()
        mock_client.get_collections.side_effect = Exception("Connection refused")

        with patch("src.vector.health.get_qdrant_client", return_value=mock_client):
            result = check_qdrant_health(settings)

        assert result is False

    def test_qdrant_health_check_returns_connection_details(
        self, settings: Settings, mock_qdrant_client
    ):
        """
        GIVEN a running Qdrant instance
        WHEN check_qdrant_health_detailed is called
        THEN it returns a dict with status, url, and collections count
        """
        from src.vector.health import check_qdrant_health_detailed

        with patch("src.vector.health.get_qdrant_client", return_value=mock_qdrant_client):
            result = check_qdrant_health_detailed(settings)

        assert "status" in result
        assert "url" in result
        assert result["status"] in ("healthy", "unhealthy")


class TestQdrantClient:
    """Tests for Qdrant client initialization."""

    def test_get_qdrant_client_creates_client_with_settings(self, settings: Settings):
        """
        GIVEN valid Qdrant settings
        WHEN get_qdrant_client is called
        THEN it returns a configured client instance
        """
        from src.vector.health import get_qdrant_client

        with patch("src.vector.health.QdrantClient") as mock_qc:
            mock_qc.return_value = MagicMock()
            client = get_qdrant_client(settings)

            mock_qc.assert_called_once_with(url=settings.qdrant_url)
            assert client is not None

    def test_get_qdrant_client_raises_on_invalid_url(self, settings: Settings):
        """
        GIVEN invalid Qdrant URL
        WHEN get_qdrant_client is called
        THEN it raises ConnectionError
        """
        from src.vector.health import get_qdrant_client

        settings.qdrant_url = "invalid://bad-url"

        with pytest.raises(ConnectionError):
            get_qdrant_client(settings)
