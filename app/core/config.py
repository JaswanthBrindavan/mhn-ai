"""Application settings, sourced entirely from the environment.

Every external dependency must be swappable by environment variable alone, with no
code change: ``database_url`` for Postgres, ``aws_endpoint_url`` for S3/SQS. Nothing
in this codebase may hardcode a host, bucket, queue URL, or credential.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database -----------------------------------------------------------
    database_url: str

    # --- AWS ----------------------------------------------------------------
    aws_region: str = "ap-south-1"
    # Empty means real AWS: boto3 resolves endpoints and credentials itself.
    aws_endpoint_url: str = ""
    s3_force_path_style: bool = False
    s3_bucket: str = ""
    sqs_queue_url: str = ""
    sqs_dlq_url: str = ""
    sqs_max_receive_count: int = 5

    # --- Worker -------------------------------------------------------------
    worker_max_concurrency: int = 4
    max_attempts: int = 3
    stale_item_timeout_seconds: int = 900

    # --- File validation ----------------------------------------------------
    max_file_bytes: int = 26_214_400
    # Comma-separated rather than a JSON list so .env stays human-editable.
    allowed_content_types: str = "application/pdf,image/jpeg,image/png"

    # --- AI provider --------------------------------------------------------
    ai_provider: str = "anthropic"
    ai_model: str = ""
    ai_max_tokens: int = 4096
    anthropic_api_key: str = ""

    # --- Service ------------------------------------------------------------
    mhn_service_token: str = ""
    log_level: str = "INFO"

    @property
    def allowed_content_type_set(self) -> frozenset[str]:
        return frozenset(
            item.strip().lower() for item in self.allowed_content_types.split(",") if item.strip()
        )

    @property
    def uses_local_aws(self) -> bool:
        """True when pointed at LocalStack rather than real AWS."""
        return bool(self.aws_endpoint_url)


@lru_cache
def get_settings() -> Settings:
    """Cached so the environment is read once per process."""
    # The pydantic mypy plugin understands that values come from env/.env.
    return Settings()
