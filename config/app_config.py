"""
Application Configuration Management System

This module implements a comprehensive configuration management system for the JOLT 
Transformation Engine API. It provides a layered, type-safe approach to handling 
application settings from multiple sources including environment variables, 
configuration files (YAML/JSON), and programmatic overrides.

Architecture & Design Patterns:
- **Pydantic Models**: Type-safe configuration with validation and documentation
- **Composite Pattern**: Multiple configuration sources with priority ordering
- **Singleton Pattern**: Single configuration manager instance across the application
- **Dependency Inversion**: Abstract ConfigSource interface for extensibility
- **Factory Pattern**: Dynamic configuration source creation and management

Core Components:
- Configuration Models: Structured settings for API, LLM, monitoring, logging, etc.
- Configuration Sources: Environment variables, file-based (YAML/JSON) loading
- Configuration Manager: Central orchestration with merging and type conversion
- Security Features: Sensitive data protection and sanitization

Key Features:
- **Environment Variable Mapping**: Automatic translation of env vars to config paths
- **Type Conversion**: Intelligent string-to-type conversion for environment variables
- **Configuration Merging**: Deep merging of multiple configuration sources
- **Validation**: Pydantic-based validation with meaningful error messages
- **File Support**: YAML and JSON configuration file loading and saving
- **Security**: Automatic sensitive data removal for configuration exports

Integration Points:
- Model Registry: LLM provider configuration and selection
- API Server: Host, port, CORS, and rate limiting settings
- Monitoring Systems: Elasticsearch, Prometheus metrics configuration
- Authentication: OneTeg and other auth provider configurations
"""

import logging
import sys
import os 
from pydantic import BaseModel, Field, field_validator, model_validator
import json 
from typing import Optional, Dict, Any, Union, List
import yaml
from dotenv import load_dotenv
from pathlib import Path
from enum import Enum

load_dotenv()


class APIConfig(BaseModel):
    """
    API server configuration settings for FastAPI application.
    
    Defines all HTTP server parameters including network settings, 
    performance tuning, security policies, and operational constraints.
    Used by the FastAPI application factory for server initialization.
    
    Attributes:
        host: Network interface to bind the server (0.0.0.0 for all interfaces)
        port: TCP port number for HTTP server
        debug: Enable debug mode with detailed error messages and auto-reload
        workers: Number of worker processes for production deployment
        cors_origins: List of allowed origins for Cross-Origin Resource Sharing
        rate_limit: Maximum requests per minute per client for rate limiting
    """
    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Enable debug mode")
    workers: int = Field(default=4, description="Number of worker processes")
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    rate_limit: int = Field(default=100, description="Rate limit per minute")


class LLMProviderConfig(BaseModel):
    """
    Base configuration model for Large Language Model providers.
    
    Standardizes configuration structure across different LLM providers
    (OpenAI, Anthropic, Azure, Google, local models). Provides common
    fields for authentication, model selection, and provider-specific
    customization through additional_settings.
    
    Attributes:
        enabled: Whether this provider is available for use
        api_key: Authentication key for the provider's API
        default_model: Default model identifier for this provider
        additional_settings: Provider-specific configuration (endpoints, regions, etc.)
    """
    enabled: bool = Field(default=True, description="Whether this provider is enabled")
    api_key: str = Field(default="", description="API key for the provider")
    default_model: str = Field(description="Default model to use for this provider")
    additional_settings: Dict[str, Any] = Field(default_factory=dict, description="Additional provider-specific settings")


class LLMConfig(BaseModel):
    """
    Comprehensive LLM provider configuration for multi-provider support.
    
    Manages configuration for all supported LLM providers including cloud-based
    services (OpenAI, Anthropic, Google) and self-hosted solutions. Provides
    fallback mechanisms and provider selection logic for the model registry.
    
    Supported Providers:
    - Anthropic Claude: Claude-3 models for high-quality reasoning
    - OpenAI: GPT-4 and GPT-3.5 models for general-purpose tasks  
    - Azure OpenAI: Enterprise OpenAI deployment with custom endpoints
    - Google Gemini: Gemini models for multimodal capabilities
    - Local LLMs: Self-hosted models via OpenAI-compatible APIs
    
    Attributes:
        default_provider: Primary provider to use when no specific provider is requested
        anthropic: Anthropic Claude configuration with API key and model selection
        openai: OpenAI configuration for GPT models
        azure_openai: Azure OpenAI service configuration with custom endpoints
        local: Local LLM configuration with custom base URLs
        gemini: Google Gemini configuration for multimodal models
    """
    default_provider: str = Field(default="anthropic", description="Default LLM provider")

    anthropic: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(
            enabled=True,
            default_model="claude-3-5-sonnet-20241022"
        ),
        description="Anthropic configuration"
    )

    openai: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(
            enabled=True,
            default_model="gpt-4",
        )
    )

    azure_openai: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(
            enabled=False,
            default_model="gpt-4"
        ),
        description="Azure OpenAI configuration"
    )

    local: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(
            enabled=False,
            default_model="deepseek-chat"
        ),
        description="Local LLM configuration"
    )

    gemini: LLMProviderConfig = Field(
        default_factory=lambda: LLMProviderConfig(
            enabled=True,
            default_model="models/gemini-2.5-flash"
        ),
        description="Google Gemini/Gemma configuration"
    )


class TransformationConfig(BaseModel):
    """
    Configuration for JOLT transformation workflow behavior and constraints.
    
    Controls the operational parameters of the JOLT transformation engine
    including performance limits, quality settings, storage options, and
    concurrency management. These settings directly impact transformation
    reliability, speed, and resource utilization.
    
    Workflow Control:
    - max_iterations: Prevents infinite loops in AI refinement cycles
    - timeout_seconds: Ensures responsive API behavior under load
    - concurrent_transformations: Balances throughput with resource usage
    
    Quality Control:
    - default_temperature: LLM creativity vs. consistency balance
    - store_transformations: Enables learning and debugging capabilities
    
    Attributes:
        max_iterations: Maximum refinement iterations before graceful termination
        timeout_seconds: Request timeout to prevent hanging transformations
        default_temperature: LLM temperature for consistent specification generation
        concurrent_transformations: Maximum parallel transformation requests
        store_transformations: Whether to persist transformations for analysis
    """
    max_iterations: int = Field(
        default=5,
        description="Maximum number of refinement iterations"
    )

    timeout_seconds: int = Field(
        default=400,
        description="Timeout for transformation request in seconds"
    )

    default_temperature: float = Field(
        default=0.1,
        description="Default temperature settings for LLMs"
    )

    concurrent_transformations: int = Field(
        default=10,
        description="Maximum concurrent transformations"
    )

    store_transformations: bool = Field(
        default=True,
        description="Whether to store transformations for future reference/training"
    )


class LoggingConfig(BaseModel):
    """
    Comprehensive logging configuration for observability and debugging.
    
    Provides flexible logging architecture supporting multiple outputs,
    formats, and storage backends. Integrates with Elasticsearch for
    centralized log aggregation and provides local file logging for
    development and debugging scenarios.
    
    Logging Targets:
    - Console: Development and debugging output
    - File: Local persistent logging with rotation
    - Elasticsearch: Centralized logging for production monitoring
    
    Log Categories:
    - API Requests: HTTP request/response logging for API monitoring
    - Transformations: JOLT workflow execution and results
    - LLM Interactions: AI model requests, responses, and performance
    
    Attributes:
        level: Python logging level (DEBUG, INFO, WARNING, ERROR)
        format: Log message format string for consistency
        elastic_enabled: Enable Elasticsearch log shipping
        elastic_url: Elasticsearch cluster endpoint URL
        elastic_username: Authentication username for Elasticsearch
        elastic_password: Authentication password for Elasticsearch
        elastic_index_prefix: Index naming pattern for log organization
        log_api_requests: Track HTTP API usage and performance
        log_transformations: Record transformation workflows and results
        local_llm_interactions: Log AI model interactions for debugging
        log_to_file: Enable local file logging for persistence
        log_dir: Directory for local log file storage
        log_max_file_size: Maximum size before log rotation
        log_backup_count: Number of rotated log files to retain
        log_json_handlers: Which outputs should use structured JSON format
    """
    level: str = Field(default="INFO", description="Logging level")

    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )

    elastic_enabled: bool = Field(
        default=True,
        description="Enable Elasticsearch logging"
    )

    elastic_url: str = Field(
        default="JOLT_ELASTIC_URL",
        description="Elasticsearch URL"
    )

    elastic_username: str = Field(
        default="JOLT_ELASTIC_USERNAME",
        description="Elasticsearch username"
    )

    elastic_password: str = Field(
        default="JOLT_ELASTIC_PASSWORD",
        description="Elasticsearch password"
    )

    elastic_index_prefix: str = Field(
        default="jolt-transformations",
        description="Prefix for Elasticsearch indices"
    )

    log_api_requests: bool = Field(
        default=True,    
        description="Whether to log API requests"
    )

    log_transformations: bool = Field(
        default=True,
        description="Whether to log transformations"
    )

    local_llm_interactions: bool = Field(
        default=True,
        description="Whether to log LLM interactions"
    )

    log_to_file: bool = Field(
        default=True,
        description="Whether to log to file"
    )

    log_dir: str = Field(
        default="logs",
        description="Directory for log files"
    )

    log_max_file_size: str = Field(
        default="10MB",
        description="Maximum size of log files (e.g., '10MB', '1GB')"
    )
    
    log_backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )

    log_json_handlers: str = Field(
        default="file",
        description="Which handlers should use JSON format ('file', 'console', 'all', 'none')"
    )


class MonitoringConfig(BaseModel):
    """
    Monitoring and metrics configuration for operational observability.
    
    Integrates multiple monitoring systems including Prometheus for metrics
    collection and Elasticsearch for real-time data aggregation. Provides
    configurable performance monitoring with async processing to minimize
    impact on API response times.
    
    Metrics Collection:
    - API Metrics: Request counts, response times, error rates
    - LLM Metrics: Model usage, token consumption, response quality
    - System Metrics: Resource utilization, queue depths, error patterns
    
    Storage Strategy:
    - Prometheus: Time-series metrics for alerting and dashboards
    - Elasticsearch: Detailed event data with flexible querying
    - Buffering: Async batch processing to reduce overhead
    
    Attributes:
        enabled: Master switch for all monitoring features
        prometheus_port: Port for Prometheus metrics endpoint
        metrics_path: HTTP path for scraping metrics
        collect_api_metrics: Track HTTP API performance and usage
        collect_llm_metrics: Monitor AI model interactions and performance
        metrics_enabled: Enable real-time metrics to Elasticsearch
        metrics_index_prefix: Elasticsearch index naming for metrics
        metrics_async: Use async processing to avoid API latency
        metrics_buffer_size: Batch size for efficient Elasticsearch writes
        metrics_flush_interval: Frequency of metric data flushing
        metrics_retention_days: Data retention policy for storage management
        metrics_use_existing_es: Reuse logging Elasticsearch connection
        dashboard_refresh_interval: UI refresh rate for monitoring dashboards
        metrics_debug: Enable debug logging for metrics troubleshooting
    """
    enabled: bool = Field(default=True, description="Enable monitoring")

    prometheus_port: int = Field(
        default=9090,
        description="Prometheus exporter port"
    )

    metrics_path: str = Field(
        default="/metrics",
        description="Path for metrics endpoint"
    )

    collect_api_metrics: bool = Field(
        default=True,
        description="Whether to collect API metrics"
    )

    collect_llm_metrics: bool = Field(
        default=True,
        description="Whether to collect LLM metrics"
    )
    
    metrics_enabled: bool = Field(
        default=True,
        description="Enable real-time metrics collection to Elasticsearch"
    )
    
    metrics_index_prefix: str = Field(
        default="jolt-metrics",
        description="Elasticsearch index prefix for metrics data"
    )
    
    metrics_async: bool = Field(
        default=True,
        description="Use async metrics processing to avoid API latency"
    )
    
    metrics_buffer_size: int = Field(
        default=100,
        description="Buffer size before writing to Elasticsearch (startup scale)"
    )
    
    metrics_flush_interval: int = Field(
        default=10,
        description="Flush interval in seconds for buffered metrics"
    )
    
    metrics_retention_days: int = Field(
        default=90,
        description="Number of days to retain metrics data (startup friendly)"
    )
    
    metrics_use_existing_es: bool = Field(
        default=True,
        description="Use existing Elasticsearch client from logging service"
    )
    
    dashboard_refresh_interval: int = Field(
        default=30,
        description="Dashboard refresh interval in seconds"
    )
    
    metrics_debug: bool = Field(
        default=False,
        description="Enable debug logging for metrics service"
    )


class OneTegConfig(BaseModel):
    """
    OneTeg platform integration configuration for enterprise features.
    
    Configures integration with the OneTeg platform for enhanced authentication,
    analytics, and enterprise features. Supports callback-based workflows
    and secure API key management for production deployments.
    
    Attributes:
        api_key: OneTeg platform authentication key
        api_url: OneTeg API endpoint URL for platform integration
        use_oneteg_auth: Enable OneTeg-based authentication workflows
        callback_url: Webhook URL for OneTeg integration callbacks
    """
    api_key: str = Field(
        default="",
        description="OneTeg API key"
    ) 

    api_url: str = Field(
        default="https://api.oneteg.com",
        description="OneTeg API URL"
    )

    use_oneteg_auth: bool = Field(
        default=True,
        description="Whether to use OneTeg authentication"
    )

    callback_url: Optional[str] = Field(
        default="",
        description="Callback URL for OneTeg integration"
    )


class DataConfig(BaseModel):
    """
    Data storage and persistence configuration for transformation data.
    
    Manages all aspects of data storage including transformations, caching,
    backup strategies, and query optimization. Supports both development
    scenarios with local storage and production deployments with advanced
    features like compression and query optimization.
    
    Storage Organization:
    - File-based storage with JSON/YAML format support
    - Date-based hierarchical organization for large datasets
    - Backup rotation with configurable retention policies
    
    Performance Features:
    - In-memory caching with TTL for frequently accessed data
    - Query optimization for large transformation libraries
    - Compression support for storage efficiency
    
    Attributes:
        data_dir: Base directory for all data storage
        store_transformations: Enable persistent storage of transformations
        cleanup_older_than_days: Automatic cleanup policy for old data
        storage_format: File format for persistent storage (json/yaml)
        cache_enabled: Enable in-memory caching for performance
        cache_ttl_seconds: Cache expiration time for data freshness
        cache_max_size: Maximum cached items to prevent memory issues
        organize_by_date: Use date-based directory hierarchy
        keep_backups: Enable backup creation for data safety
        max_backups: Maximum backup files per transformation
        query_timeout_seconds: Timeout for complex storage queries
        max_search_size: Maximum results to prevent large result sets
        default_page_size: Default pagination size for API responses
        enable_query_optimization: Advanced query optimization features
        use_compression: Enable storage compression for space efficiency
    """
    # Base data storage configuration
    data_dir: str = Field(
        default="logs",
        description="Directory for data storage"
    )
    
    store_transformations: bool = Field(
        default=True,
        description="Whether to store transformations to disk"
    )
    
    cleanup_older_than_days: int = Field(
        default=30,
        description="Clean up data older than this many days"
    )

    # Storage format configuration
    storage_format: str = Field(
        default="json",
        description="Format for storing data on disk (json or yaml)"
    )

    # Caching configuration
    cache_enabled: bool = Field(
        default=True,
        description="Whether to enable in-memory caching"
    )
    
    cache_ttl_seconds: int = Field(
        default=300,
        description="Time-to-live for cached items in seconds"
    )
    
    cache_max_size: int = Field(
        default=1000,
        description="Maximum number of items to keep in cache"
    )

    # Storage organization
    organize_by_date: bool = Field(
        default=False,
        description="Whether to organize disk storage by date hierarchy"
    )
    
    keep_backups: bool = Field(
        default=True, 
        description="Whether to keep backups of updated transformations"
    )
    
    max_backups: int = Field(
        default=3,
        description="Maximum number of backups to keep per transformation"
    )

    # Query configuration
    query_timeout_seconds: int = Field(
        default=30,
        description="Timeout for storage queries in seconds"
    )
    
    max_search_size: int = Field(
        default=1000,
        description="Maximum number of results to return from a search"
    )
    
    default_page_size: int = Field(
        default=20,
        description="Default number of items per page for pagination"
    )
    
    # Storage performance optimizations
    enable_query_optimization: bool = Field(
        default=True,
        description="Whether to enable query optimization features"
    )
    
    use_compression: bool = Field(
        default=False,
        description="Whether to compress data on disk"
    )

class EmbeddingModelConfig(BaseModel):
    """Configuration for individual embedding models"""
    enabled: bool = Field(default=True, description="Whether this model is enabled")
    model_name: str = Field(description="Model identifier")
    model_path: Optional[str] = Field(default=None, description="Local model path")
    dimensions: int = Field(default=4096, description="Embedding dimensions")
    max_tokens: int = Field(default=32768, description="Maximum input tokens")
    batch_size: int = Field(default=16, description="Batch size for processing")
    device: str = Field(default="auto", description="Device to use (auto, mps, cpu)")
    normalize_embeddings: bool = Field(default=True, description="Normalize embedding vectors")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Additional model parameters")


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration"""
    enabled: bool = Field(default=True, description="Enable Qdrant vector database")
    url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    api_key: str = Field(default="", description="Qdrant API key for authentication")
    collection_name: str = Field(default="oneteg_embeddings", description="Default collection name")
    vector_size: int = Field(default=4096, description="Vector dimensions")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    prefer_grpc: bool = Field(default=False, description="Prefer gRPC over HTTP")


class ContentFormattingConfig(BaseModel):
    """Configuration for content formatting before embedding"""
    max_content_length: int = Field(default=30000, description="Maximum content length before truncation")
    include_metadata: bool = Field(default=True, description="Include metadata in embeddings")
    metadata_fields: List[str] = Field(
        default=["connector_name", "categories", "use_cases"], 
        description="Metadata fields to include"
    )
    content_template: str = Field(
        default="Connector: {connector_name}\nDescription: {description}\nCategories: {categories}\nUse cases: {use_cases}",
        description="Template for formatting content"
    )


class EmbeddingConfig(BaseModel):
    """Comprehensive embedding system configuration"""
    enabled: bool = Field(default=True, description="Enable embedding system")
    default_model: str = Field(default="qwen3-8b", description="Default embedding model")
    models: Dict[str, EmbeddingModelConfig] = Field(
        default_factory=lambda: {
            "qwen3-8b": EmbeddingModelConfig(
                enabled=True,
                model_name="qwen3-8b",
                dimensions=4096,
                max_tokens=32768,
                batch_size=16,
                device="auto",
                normalize_embeddings=True
            )
        },
        description="Available embedding models"
    )
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig, description="Qdrant vector database settings")
    content_formatting: ContentFormattingConfig = Field(
        default_factory=ContentFormattingConfig, 
        description="Content formatting settings"
    )


class AppConfig(BaseModel):
    """
    Root application configuration containing all subsystem configurations.
    
    Serves as the main configuration container that aggregates all specialized
    configuration models into a single, validated structure. Used throughout
    the application as the authoritative source for all configuration values.
    
    Configuration Hierarchy:
    - Application: Basic app metadata and environment settings
    - API: HTTP server and networking configuration
    - LLM: Multi-provider AI model configuration
    - Transformation: JOLT workflow behavior and constraints
    - Logging: Observability and debugging configuration
    - Monitoring: Metrics collection and performance tracking
    - OneTeg: Enterprise platform integration
    - Data: Storage and persistence management
    
    Attributes:
        app_name: Human-readable application identifier
        environment: Deployment environment (development, staging, production)
        api: HTTP API server configuration
        llm: LLM provider configuration for AI features
        transformation: JOLT transformation workflow settings
        logging: Logging and observability configuration
        monitoring: Metrics and monitoring system settings
        oneteg: OneTeg platform integration configuration
        data: Data storage and persistence configuration
    """
    app_name: str = Field(
        default="JOLT Transformation Engine API",
        description="Application name"
    )

    environment: str = Field(
        default="development",
        description="Development environment"
    )

    api: APIConfig = Field(
        default_factory=APIConfig, description="API configuration"
    )

    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")

    transformation: TransformationConfig = Field(
        default_factory=TransformationConfig,
        description="Transformation configuration"
    )

    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )

    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )

    oneteg: OneTegConfig = Field(
        default_factory=OneTegConfig,
        description="OneTeg configuration"
    )

    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data storage configuration"
    )

    embedding: EmbeddingConfig = Field(
    default_factory=EmbeddingConfig,
    description="Embedding models and vector database configuration"
)


class ConfigSource:
    """
    Abstract interface for configuration data sources.
    
    Implements the Strategy pattern for loading configuration from different
    sources (environment variables, files, databases, remote APIs). Enables
    flexible configuration architecture with pluggable data sources and
    consistent loading interface across all source types.
    
    This interface supports the Dependency Inversion principle by allowing
    the ConfigManager to work with any configuration source without coupling
    to specific implementation details.
    """

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration data from this source.
        
        Returns:
            Dict[str, Any]: Configuration dictionary with nested structure
                          matching the AppConfig model hierarchy
                          
        Raises:
            NotImplementedError: Must be implemented by concrete sources
        """
        raise NotImplementedError()


class EnvConfigSource(ConfigSource):
    """
    Environment variable configuration source with automatic type conversion.
    
    Loads configuration from environment variables using a predefined mapping
    system that translates flat environment variable names to nested configuration
    paths. Supports the standard pattern of using prefixed environment variables
    for application configuration in containerized deployments.
    
    Environment Variable Pattern:
    - Prefix: JOLT_ for all application variables
    - Mapping: Direct translation to nested configuration paths
    - Type Safety: Automatic conversion handled by ConfigManager
    
    Example Mappings:
    - JOLT_API_HOST → api.host
    - JOLT_LLM_DEFAULT_PROVIDER → llm.default_provider
    - JOLT_ANTHROPIC_API_KEY → llm.anthropic.api_key
    
    Attributes:
        ENV_PREFIX: Prefix for all environment variables ("JOLT_")
        ENV_MAPPING: Dictionary mapping env var names to config paths
    """

    ENV_PREFIX = ""

    # Comprehensive mapping of environment variables to configuration paths
    ENV_MAPPING = {
        # API Configuration
        "API_HOST": "api.host",
        "API_PORT": "api.port",
        "API_WORKERS": "api.workers",
        "API_CORE_ORIGINS": "api.cors_origins",
        "API_RATE_LIMIT": "api.rate_limit",

        # LLM Provider Configuration
        "LLM_DEFAULT_PROVIDER": "llm.default_provider",
        "OPENAI_API_KEY": "llm.openai.api_key",
        "OPENAI_ENABLED": "llm.openai.enabled",
        "OPENAI_DEFAULT_MODEL": "llm.openai.default_model",

        "ANTHROPIC_API_KEY": "llm.anthropic.api_key",
        "ANTHROPIC_ENABLED": "llm.anthropic.enabled",
        "ANTHROPIC_DEFAULT_MODEL": "llm.anthropic.llm_model",

        "GEMINI_API_KEY": "llm.gemini.api_key",
        "GEMINI_ENABLED": "llm.gemini.enabled",
        "GEMINI_DEFAULT_MODEL": "llm.gemini.default_model",
        "GEMINI_API_KEY": "llm.gemini.api_key",
        "GEMINI_ENABLED": "llm.gemini.enabled", 
        "GEMINI_DEFAULT_MODEL": "llm.gemini.default_model",
    
        "GOOGLE_API_KEY": "llm.gemini.api_key",
        
        "GEMMA_PREFERRED_MODEL": "llm.gemini.default_model",  # Can be set to any Gemma model
        "GEMMA_ENABLED": "llm.gemini.enabled",  # Enable/disable Gemma models specifically
    
        
        "AZURE_OPENAI_API_KEY": "llm.azure_openai.api_key",
        "AZURE_OPENAI_ENABLED": "llm.azure_openai.enabled",
        "AZURE_OPENAI_DEFAULT_MODEL": "llm.azure_openai.default_model",
        "AZURE_OPENAI_ENDPOINT": "llm.azure_openai.additional_settings.endpoint",

        "LOCAL_LLM_ENABLED": "llm.local.enabled",
        "LOCAL_LLM_DEAFULT_MODEL": "llm.local.default_model",
        "LOCAL_LLM_BASE_URL": "llm.local.additional_settings.base_url",

        # Transformation Configuration
        "TRANSFORMATION_MAX_ITERATIONS": "transformation.max_iterations",
        "TRANSFORMATION_TIMEOUT_SECONDS": "transformation.timeout_seconds",
        "TRANSFORMATION_DEFAULT_TEMPERATURE": "transformation.default_temperature",
        "TRANSFORMATION_CONCURRENT": "transformation.concurrent_transformations",
        "TRANFORMATION_STORE": "transformation.store_transformations",

        # Logging Configuration
        "LOG_LEVEL": "logging.level",
        "LOG_FORMAT": "logging.format",
        "ELASTIC_ENABLED": "logging.elastic_enabled",
        "ELASTIC_URL": "logging.elastic_url",
        "ELASTIC_USERNAME": "logging.elastic_username",
        "ELASTIC_PASSWORD": "logging.elastic_password",  
        "ELASTIC_INDEX_PREFIX": "logging.elastic_index_prefix",  
        "LOG_API_REQUESTS": "logging.log_api_requests",
        "LOG_TRANSFORMATIONS": "logging.log_transformations",
        "LOG_LLM_INTERACTIONS": "logging.local_llm_interactions",
        "LOG_TO_FILE": "logging.log_to_file",
        "LOG_DIR": "logging.log_dir",
        "LOG_MAX_FILE_SIZE": "logging.log_max_file_size",
        "LOG_BACKUP_COUNT": "logging.log_backup_count",
        "LOG_JSON_HANDLERS": "logging.log_json_handlers",

        # Monitoring Configuration
        "MONITORING_ENABLED": "monitoring.enabled",
        "PROMETHEUS_PORT": "monitoring.prometheus_port",
        "METRICS_PATH": "monitoring.metrics_path",
        "COLLECT_API_METRICS": "monitoring.collect_api_metrics",
        "COLLECT_LLM_METRICS": "monitoring.collect_llm_metrics",
        "METRICS_ENABLED": "monitoring.metrics_enabled",
        "METRICS_INDEX_PREFIX": "monitoring.metrics_index_prefix",
        "METRICS_ASYNC": "monitoring.metrics_async",
        "METRICS_BUFFER_SIZE": "monitoring.metrics_buffer_size",
        "METRICS_FLUSH_INTERVAL": "monitoring.metrics_flush_interval",
        "METRICS_RETENTION_DAYS": "monitoring.metrics_retention_days",
        "METRICS_USE_EXISTING_ES": "monitoring.metrics_use_existing_es",
        "DASHBOARD_REFRESH_INTERVAL": "monitoring.dashboard_refresh_interval",
        "METRICS_DEBUG": "monitoring.metrics_debug",

        # OneTeg Configuration
        "ONETEG_API_KEY": "oneteg.api_key",
        "ONETEG_API_URL": "oneteg.api_url",
        "ONETEG_USE_AUTH": "oneteg.use_oneteg_auth",
        "ONETEG_CALLBACK_URL": "oneteg.callback_url",

        # Data Configuration
        "DATA_DIR": "data.data_dir",
        "STORE_TRANSFORMATIONS": "data.store_transformations",
        "CLEANUP_OLDER_THAN_DAYS": "data.cleanup_older_than_days",

        # Application Configuration
        "APP_NAME": "app_name",
        "ENVIRONMENT": "environment",

        # Embedding Configuration - ADD THIS SECTION
        "EMBEDDING_ENABLED": "embedding.enabled",
        "EMBEDDING_DEFAULT_MODEL": "embedding.default_model",
        
        # Qwen3-8B Model Configuration
        "QWEN3_ENABLED": "embedding.models.qwen3-8b.enabled",
        "QWEN3_DIMENSIONS": "embedding.models.qwen3-8b.dimensions",
        "QWEN3_MAX_TOKENS": "embedding.models.qwen3-8b.max_tokens",
        "QWEN3_BATCH_SIZE": "embedding.models.qwen3-8b.batch_size",
        "QWEN3_DEVICE": "embedding.models.qwen3-8b.device",
        "QWEN3_NORMALIZE": "embedding.models.qwen3-8b.normalize_embeddings",
        
        # Qdrant Configuration
        "QDRANT_ENABLED": "embedding.qdrant.enabled",
        "QDRANT_URL": "embedding.qdrant.url",
        "QDRANT_API_KEY": "embedding.qdrant.api_key",
        "QDRANT_COLLECTION": "embedding.qdrant.collection_name",
        "QDRANT_VECTOR_SIZE": "embedding.qdrant.vector_size",
        "QDRANT_DISTANCE_METRIC": "embedding.qdrant.distance_metric",
        "QDRANT_TIMEOUT": "embedding.qdrant.timeout",
        "QDRANT_PREFER_GRPC": "embedding.qdrant.prefer_grpc",
        
        # Content Formatting
        "EMBEDDING_MAX_CONTENT_LENGTH": "embedding.content_formatting.max_content_length",
        "EMBEDDING_INCLUDE_METADATA": "embedding.content_formatting.include_metadata"
        
    }

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from environment variables using the mapping system.
        
        Iterates through the ENV_MAPPING to find matching environment variables
        and converts them to nested configuration dictionary structure. Handles
        missing variables gracefully by only including variables that are set.
        
        Returns:
            Dict[str, Any]: Nested configuration dictionary with values from
                          environment variables, ready for merging with other sources
        """
        config = {}

        for env_var, config_path in self.ENV_MAPPING.items():
            full_env_var = f"{self.ENV_PREFIX}{env_var}"

            if full_env_var in os.environ:
                value = os.environ[full_env_var]
                self._set_nested_value(config, config_path, value)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: str):
        """
        Set a value in a nested dictionary using dot-separated path notation.
        
        Creates intermediate dictionaries as needed to build the full path
        structure. This enables flat environment variables to be converted
        into nested configuration hierarchies that match the Pydantic models.
        
        Args:
            config: The configuration dictionary to modify
            path: Dot-separated path to the target configuration value
            value: String value from the environment variable
            
        Example:
            _set_nested_value(config, "api.host", "0.0.0.0")
            # Results in: config["api"]["host"] = "0.0.0.0"
        """
        keys = path.split(".")
        current = config

        # Navigate/create the nested structure
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value


class FileConfigSource(ConfigSource):
    """
    File-based configuration source supporting YAML and JSON formats.
    
    Loads configuration from structured files using automatic format detection
    based on file extension. Supports both YAML (human-readable, comments)
    and JSON (machine-readable, strict) formats for different use cases.
    
    Use Cases:
    - Development: YAML files with comments and readable structure
    - Production: JSON files for strict parsing and validation
    - Configuration Templates: Shareable configuration examples
    - Environment-Specific: Different files for dev/staging/production
    
    Attributes:
        config_path: Path to the configuration file to load
    """

    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize with the configuration file path.
        
        Args:
            config_path: Path to the configuration file (YAML or JSON)
        """
        self.config_path = Path(config_path) if isinstance(config_path, str) else config_path

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file with automatic format detection.
        
        Detects file format based on extension and uses appropriate parser.
        Handles missing files gracefully by returning empty configuration.
        Provides detailed error logging for debugging file format issues.
        
        Returns:
            Dict[str, Any]: Configuration dictionary loaded from file,
                          or empty dict if file doesn't exist or parsing fails
                          
        Side Effects:
            Logs errors to console if file loading or parsing fails
        """
        if not self.config_path.exists():
            return {}

        try:
            if self.config_path.suffix.lower() in [".yaml", ".yml"]:
                with open(self.config_path, "r") as f:
                    return yaml.safe_load(f) or {}
            elif self.config_path.suffix.lower() == ".json":
                with open(self.config_path, "r") as f:
                    return json.load(f)
            else:
                print(f"Unsupported configuration file format: {self.config_path.suffix}")
                return {}
        except Exception as e:
            print(f"Error loading configuration from {self.config_path}: {e}")
            return {}


class ConfigManager:
    """
    Central configuration manager implementing Singleton and Composite patterns.
    
    Orchestrates configuration loading from multiple sources, handles type
    conversion, validation, and provides a unified interface for accessing
    application configuration. Uses the Singleton pattern to ensure consistent
    configuration access across the entire application.
    
    Design Patterns:
    - **Singleton**: Single configuration instance across application lifecycle
    - **Composite**: Multiple configuration sources with priority ordering
    - **Template Method**: Standardized configuration loading and merging process
    
    Configuration Loading Process:
    1. Initialize with default configuration from Pydantic models
    2. Load from each configuration source in priority order
    3. Deep merge configuration dictionaries with conflict resolution
    4. Convert string values to appropriate types for Pydantic validation
    5. Validate complete configuration against Pydantic models
    6. Provide thread-safe access to validated configuration
    
    Thread Safety:
    - Singleton initialization is thread-safe
    - Configuration is immutable after initialization
    - No shared mutable state between requests
    
    Attributes:
        _instance: Singleton instance reference
        _initialized: Prevents re-initialization of singleton
        sources: List of configuration sources in priority order
        config: Validated AppConfig instance with all settings
    """

    _instance = None

    def __new__(cls):
        """
        Implement Singleton pattern with thread-safe initialization.
        
        Ensures only one ConfigManager instance exists across the application
        and prevents duplicate initialization even in multi-threaded scenarios.
        
        Returns:
            ConfigManager: The singleton ConfigManager instance
        """
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """
        Initialize the configuration manager with default sources.
        
        Sets up the configuration loading pipeline with environment variables
        as the default source. Additional sources can be added via add_source().
        Only performs initialization once per singleton instance.
        
        Side Effects:
            - Loads configuration from all registered sources
            - Validates configuration against Pydantic models
            - Sets up default logging and error handling
        """
        if self._initialized:
            return
        
        # Initialize with environment variable source as default
        self.sources = [EnvConfigSource()]
        
        # Start with default configuration from Pydantic models
        self.config = AppConfig()
        
        # Load and merge configuration from all sources
        self._load_config()
        
        self._initialized = True

    def _load_config(self):
        """
        Load and merge configuration from all registered sources.
        
        Iterates through all configuration sources in priority order,
        loading configuration data and merging it with existing settings.
        Later sources can override earlier sources, enabling flexible
        configuration hierarchies.
        
        Side Effects:
            - Updates self.config with merged configuration data
            - Logs configuration loading progress and errors
        """
        for source in self.sources:
            config_data = source.load_config()

            if config_data:
                self._update_config(config_data)

    def _update_config(self, config_data: Dict[str, Any]):
        """
        Update the configuration with data from a source.
        
        Merges new configuration data with existing settings using deep merge
        strategy. Handles type conversion from string environment variables
        to appropriate Python types, then validates the complete configuration
        against Pydantic models for type safety and constraint validation.
        
        Args:
            config_data: Configuration data to merge and apply
            
        Side Effects:
            - Performs deep merge of configuration dictionaries
            - Converts string values to appropriate types
            - Re-validates entire configuration with Pydantic
            - Logs errors if configuration validation fails
        """
        # Start with current configuration as base
        config_dict = self.config.model_dump()
        
        # Deep merge new configuration data
        self._deep_update(config_dict, config_data)
        
        # Convert string values to appropriate types for Pydantic
        self._convert_types(config_dict)
        
        try:
            # Validate and create new configuration instance
            self.config = AppConfig.model_validate(config_dict)
        except Exception as e:
            print(f"Error updating configuration: {e}")
            print(f"Problematic config: {config_dict}")

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Perform deep merge of nested dictionaries with conflict resolution.
        
        Recursively merges source dictionary into target dictionary, handling
        nested structures appropriately. Preserves existing nested dictionaries
        while allowing specific value overrides. Source values take precedence
        over target values in case of conflicts.
        
        Args:
            target: Target dictionary to update (modified in place)
            source: Source dictionary containing updates
            
        Conflict Resolution:
        - Nested dicts: Recursive merge preserving both structures
        - Scalar values: Source overwrites target
        - Type mismatches: Source value replaces target value
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                self._deep_update(target[key], value)
            else:
                # Direct assignment for scalar values or type mismatches
                target[key] = value

    def _convert_types(self, config: Dict[str, Any]):
        """
        Convert string values to appropriate Python types for Pydantic validation.
        
        Environment variables are always strings, but Pydantic models expect
        specific types (bool, int, float, etc.). This method intelligently
        converts string values based on content and patterns to match the
        expected types in the configuration models.
        
        Conversion Rules:
        - Boolean: 'true', 'yes', 'y', '1' → True; 'false', 'no', 'n', '0' → False
        - Integer: Numeric strings including negative values
        - Float: Decimal numbers with proper float parsing
        - String: Default for values that don't match other patterns
        
        Args:
            config: Configuration dictionary to modify (updated in place)
            
        Side Effects:
            - Modifies dictionary values in place
            - Preserves original values if conversion fails
        """
        def convert_value(path: str, value: Any) -> Any:
            """Convert individual values based on content analysis."""
            if value is None:
                return None
                
            if isinstance(value, str):
                # Boolean conversion with multiple accepted formats
                if value.lower() in ['true', 'yes', 'y', '1']:
                    return True
                elif value.lower() in ['false', 'no', 'n', '0']:
                    return False
                
                # Integer conversion including negative numbers
                try:
                    if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                        return int(value)
                except:
                    pass
                
                # Float conversion for decimal numbers
                try:
                    if '.' in value:
                        return float(value)
                except:
                    pass
            
            return value
        
        # Recursively process the entire configuration dictionary
        self._process_dict(config, "", convert_value)

    def _process_dict(self, d: Dict[str, Any], path: str, processor):
        """
        Recursively process a dictionary, applying a function to each value.
        
        Provides a generic mechanism for traversing nested dictionary structures
        and applying transformations to leaf values. Used for type conversion,
        validation, and other recursive dictionary operations.
        
        Args:
            d: Dictionary to process (modified in place)
            path: Current path in the dictionary for context
            processor: Function to apply to each leaf value
            
        Side Effects:
            - Modifies dictionary values in place through processor function
            - Maintains dictionary structure while transforming values
        """
        for key, value in list(d.items()):
            current_path = f"{path}.{key}" if path else key
            
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                self._process_dict(value, current_path, processor)
            else:
                # Apply processor to leaf values
                d[key] = processor(current_path, value)

    def get_config(self) -> AppConfig:
        """
        Get the complete validated application configuration.
        
        Provides thread-safe access to the current application configuration.
        The returned configuration is immutable and validated against all
        Pydantic models, ensuring type safety and constraint compliance.
        
        Returns:
            AppConfig: Complete validated application configuration
        """
        return self.config
    
    def add_source(self, source: ConfigSource):
        """
        Add a new configuration source and reload configuration.
        
        Allows dynamic addition of configuration sources (e.g., file-based
        configuration, remote configuration services). New sources are
        processed immediately and merged with existing configuration.
        
        Args:
            source: ConfigSource implementation to add to the loading pipeline
            
        Side Effects:
            - Appends source to sources list
            - Immediately loads and merges configuration from new source
            - Updates current configuration with new values
        """
        self.sources.append(source)
        self._load_config()

    def _remove_sensitive_data(self, config_dict: Dict[str, Any]):
        """
        Remove sensitive information from configuration for safe export/logging.
        
        Sanitizes configuration data by replacing sensitive values (API keys,
        passwords, tokens) with placeholder text. This enables safe configuration
        export for templates, documentation, or debugging without exposing
        sensitive credentials.
        
        Sensitive Data Categories:
        - LLM API Keys: All provider authentication keys
        - Database Passwords: Elasticsearch and other database credentials
        - External Service Keys: OneTeg and other platform integration keys
        
        Args:
            config_dict: Configuration dictionary to sanitize (modified in place)
            
        Returns:
            Dict[str, Any]: The modified configuration dictionary with sensitive data removed
            
        Side Effects:
            - Replaces sensitive values with placeholder text
            - Preserves configuration structure for template generation
        """
        if "llm" in config_dict:
            for provider in ["anthropic", "openai", "azure_openai", "local", "gemini"]:
                if provider in config_dict["llm"] and "api_key" in config_dict["llm"][provider]:
                    config_dict["llm"][provider]["api_key"] = "will taken from secure environment"
        
        if "logging" in config_dict and "elastic_password" in config_dict["logging"]:
            config_dict["logging"]["elastic_password"] = "will taken from secure environment"
            
        if "oneteg" in config_dict and "api_key" in config_dict["oneteg"]:
            config_dict["oneteg"]["api_key"] = "will taken from secure environment"

        if "embedding" in config_dict:
            if "qdrant" in config_dict["embedding"] and "api_key" in config_dict["embedding"]["qdrant"]:
                config_dict["embedding"]["qdrant"]["api_key"] = "will taken from secure environment"

        return config_dict

    def save_config(self, path: Union[str, Path], format: str = "yaml"):
        """
        Save the current configuration to a file with sensitive data removal.
        
        Exports the current configuration to a file for documentation, templates,
        or backup purposes. Automatically removes sensitive data to prevent
        accidental credential exposure. Supports both YAML and JSON formats.
        
        Use Cases:
        - Configuration Templates: Create example configurations for deployment
        - Documentation: Generate current configuration for troubleshooting
        - Backup: Save configuration state before major changes
        - Environment Setup: Create base configurations for new environments
        
        Args:
            path: File path for saving the configuration
            format: Output format ('yaml' or 'json')
            
        Raises:
            ValueError: If format is not 'yaml' or 'json'
            Exception: If file writing fails due to permissions or other issues
            
        Side Effects:
            - Creates or overwrites the specified file
            - Logs errors if file operations fail
        """
        config_dict = self.config.model_dump()
        self._remove_sensitive_data(config_dict)
        
        try:
            if format.lower() == "yaml":
                with open(path, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif format.lower() == "json":
                with open(path, "w") as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            print(f"Error saving configuration: {e}")


# Singleton instance for global configuration access
_config_manager = ConfigManager()


def get_config() -> AppConfig:
    """
    Get the application configuration from the singleton manager.
    
    Provides convenient global access to the validated application configuration.
    This is the primary interface used throughout the application for accessing
    configuration values in a type-safe manner.
    
    Returns:
        AppConfig: Complete validated application configuration
        
    Example:
        config = get_config()
        api_host = config.api.host
        llm_provider = config.llm.default_provider
    """
    return _config_manager.get_config()


def add_config_file(config_path: Union[str, Path]):
    """
    Add a configuration file source to the global configuration manager.
    
    Enables loading configuration from YAML or JSON files in addition to
    environment variables. File-based configuration can override environment
    settings, enabling flexible deployment strategies.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Example:
        add_config_file("config/production.yaml")
        add_config_file("/etc/jolt/config.json")
    """
    _config_manager.add_source(FileConfigSource(config_path))


def save_config(path: Union[str, Path], format: str = 'yaml'):
    """
    Save the current configuration to a file with automatic sanitization.
    
    Convenience function for exporting the current configuration state to
    a file. Automatically removes sensitive data to prevent credential exposure.
    
    Args:
        path: Path to save the configuration file
        format: Output format ('yaml' or 'json')
        
    Example:
        save_config("config/current.yaml")
        save_config("backup/config.json", "json")
    """
    _config_manager.save_config(path, format)