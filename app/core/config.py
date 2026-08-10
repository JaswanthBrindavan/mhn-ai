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
    # No sqs_dlq_url / sqs_max_receive_count here. Both were defined, advertised, and read
    # by nothing: the dead-letter queue and its maxReceiveCount are attributes of the queue
    # in AWS, not something this service configures or verifies. Carrying them implied the
    # redrive policy was managed here — which matters, because "it will end up in the DLQ"
    # is load-bearing reasoning in app/integrations/sqs.py. Check the queue, not this file.

    # --- Worker -------------------------------------------------------------
    worker_max_concurrency: int = 4
    max_attempts: int = 3
    # No stale_item_timeout_seconds: it was the threshold for a stale-item reaper that was
    # promised in five places and never built. A publish failure now ends the item `failed`
    # rather than waiting for a sweep that does not exist. If the reaper is ever built, the
    # setting comes back with it.
    # How long a received message stays invisible while a worker holds it. The
    # heartbeat re-extends this before it lapses, so a stage may run longer than
    # this value without the message being redelivered.
    sqs_visibility_timeout_seconds: int = 300
    # Long-poll wait; SQS caps this at 20. Fewer empty receives, lower cost.
    sqs_wait_time_seconds: int = 20

    # --- File validation ----------------------------------------------------
    # 50 MiB. Merged multi-image PDFs (Spring stitches reordered photos into one PDF)
    # run larger than native reports; uploads go through the Files API (500 MB ceiling),
    # so this is our own guard, not an API limit.
    max_file_bytes: int = 52_428_800
    # Comma-separated rather than a JSON list so .env stays human-editable.
    allowed_content_types: str = "application/pdf,image/jpeg,image/png"

    # Pages of a PDF sent to the classifier (the type is evident from the first pages,
    # so extraction re-reads the whole document but classification need not). 0 disables
    # trimming and sends the full document.
    classify_max_pages: int = 2

    # --- AI provider --------------------------------------------------------
    ai_provider: str = "anthropic"
    ai_model: str = ""
    # Insights run on their own model (typically a stronger one); empty falls back to ai_model.
    ai_model_insights: str = ""
    # No global token ceiling: every stage sets its own, because they differ by an order
    # of magnitude (a classification label against a 130-result panel). A single setting
    # here was read by nothing while appearing to govern all of them.
    anthropic_api_key: str = ""

    # Per-stage provider overrides. Empty = same provider as everything else (the default,
    # so single-provider deploys are unchanged). Only "gemini" is recognised today.
    #
    # Classification picks one label from the first two pages, and extraction transcribes a
    # table — neither needs a frontier model, and extraction is the most expensive stage
    # because the whole document goes to the model. Insight generation deliberately has no
    # override: it reasons about health for a patient to read, and stays on Claude.
    # Measured: same results, 41% less cost, 20% faster. See docs/ai-provider-comparison.md.
    classification_provider: str = ""
    ai_model_classification: str = ""
    extraction_provider: str = ""
    ai_model_extraction: str = ""
    # Prescriptions are read by the same mechanism but measured separately: the document
    # goes to the model as a file (layout carries the dosing), and many are photographs
    # rather than digital PDFs. Measured on 21 real prescriptions and bills, Gemini
    # gemini-3.1-flash-lite read every one at ~2,100 in / 280 out tokens; the larger
    # flash tiers returned 503 on nearly every request. See app/services/prescriptions.py.
    prescription_provider: str = ""
    ai_model_prescription: str = ""
    google_api_key: str = ""

    # --- Ideal ranges (approved-THP override) -------------------------------
    # When on, extraction overrides the report's reference range with the R&D-approved
    # ideal range for the patient's age bracket (Spring-owned THP tables:
    # traditional_health_parameters / thp_age_range / thp_alternate_units). OFF by default:
    # the tables ship in Spring's migration but are not in every database this service
    # points at, and an empty master would send every test to the fallback worklist. Flip
    # on once they exist and carry approved rows. See app/services/ideal_ranges.py.
    ideal_ranges_enabled: bool = False

    # --- Prescriptions ------------------------------------------------------
    # When on, a document classified as a prescription is filed into Spring's
    # `prescriptions` table and extracted. OFF by default because filing is never undone
    # and Spring's `listMyFiles` still returns 501 for this section: a filed prescription
    # would render nowhere, and could not be moved back. Rejected while off, exactly as
    # before the extractor existed — the document stays visible in Unclassified.
    # Flip on once Spring lists and serves prescriptions.
    prescriptions_enabled: bool = False

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


# A token shorter than this is not a credible secret. 32 bytes of randomness is the
# documented minimum for MHN_SERVICE_TOKEN.
MIN_SERVICE_TOKEN_LENGTH = 32


def verify_required_settings(settings: Settings) -> None:
    """Fail closed at startup rather than silently running without authentication.

    An unset ``MHN_SERVICE_TOKEN`` must never mean "allow everyone". Since this
    service trusts Spring's access decisions (see the authentication design notes),
    the token is the only thing separating callers from every report's extracted lab
    values — so a misconfiguration has to stop the process, not degrade quietly.
    """
    token = settings.mhn_service_token
    if not token:
        raise RuntimeError(
            "MHN_SERVICE_TOKEN is not set. Refusing to start: an empty token would "
            "disable service authentication entirely."
        )
    if len(token) < MIN_SERVICE_TOKEN_LENGTH:
        raise RuntimeError(
            f"MHN_SERVICE_TOKEN is too short ({len(token)} chars); "
            f"at least {MIN_SERVICE_TOKEN_LENGTH} are required."
        )


@lru_cache
def get_settings() -> Settings:
    """Cached so the environment is read once per process."""
    # The pydantic mypy plugin understands that values come from env/.env.
    return Settings()
