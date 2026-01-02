"""
Configuration module for semantic-search-service.

Uses pydantic-settings for environment-based configuration with feature flags
for Graph RAG hybrid search capabilities.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Feature flags control Graph RAG capabilities:
    - enable_graph_search: Enable Neo4j graph traversal queries
    - enable_hybrid_search: Enable combined vector + graph search
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===========================================
    # SERVICE CONFIGURATION
    # ===========================================
    semantic_search_port: int = Field(default=8081, description="Service port")
    sbert_model: str = Field(
        description="Sentence-BERT model for embeddings - REQUIRED, set via SBERT_MODEL env var",
    )

    # ===========================================
    # NEO4J CONFIGURATION (all required via env vars)
    # ===========================================
    neo4j_url: str = Field(
        description="Neo4j Bolt protocol URL - REQUIRED, set via NEO4J_URL env var",
    )
    neo4j_user: str = Field(description="Neo4j username - REQUIRED, set via NEO4J_USER env var")
    neo4j_password: str = Field(
        description="Neo4j password - REQUIRED, set via NEO4J_PASSWORD env var",
    )

    # ===========================================
    # QDRANT CONFIGURATION (required via env vars)
    # ===========================================
    qdrant_url: str = Field(
        description="Qdrant REST API URL - REQUIRED, set via QDRANT_URL env var",
    )

    # ===========================================
    # FEATURE FLAGS (Graph RAG) - Phase 6 Production Ready
    # ===========================================
    enable_graph_search: bool = Field(
        default=True,  # Enabled after WBS 6.1-6.3 validation
        description="Enable Neo4j graph traversal queries",
    )
    enable_hybrid_search: bool = Field(
        default=True,  # Enabled after WBS 6.1-6.3 validation
        description="Enable combined vector + graph search",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
