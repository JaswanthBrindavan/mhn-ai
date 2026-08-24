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
    # Read by /ready, to report how many messages are sitting in the dead-letter queue.
    # It is NOT how the DLQ is configured: the redrive policy and its maxReceiveCount are
    # attributes of the main queue in AWS, and this service neither sets nor verifies them.
    # What it does is make the depth visible — a message here is a document nobody is
    # processing, and until this existed the only way to notice was to open the console.
    # Empty simply omits the report.
    #
    # No sqs_max_receive_count: that one really was read by nothing, and it belongs to the
    # queue. Check `aws sqs get-queue-attributes` for the policy, not this file.
    sqs_dlq_url: str = ""

    # --- Worker -------------------------------------------------------------
    worker_max_concurrency: int = 4
    max_attempts: int = 3
    # How long a non-terminal item may sit untouched before the reaper re-queues it
    # (app/workers/reaper.py). Must stay comfortably above the longest SINGLE stage, not
    # the longest pipeline: `updated_at` moves at each stage transition, so a live worker
    # is only invisible to the sweep for the length of one stage. Measured worst case is
    # insight generation at ~75s, against this 900s.
    #
    # The heartbeat does NOT keep this fresh — it extends SQS visibility and touches no
    # row — so do not lower this on the assumption that it does.
    stale_item_timeout_seconds: int = 900
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
    # `prescriptions` table and extracted.
    #
    # ON by default since 2026-08-18. It shipped off because Spring's `listMyFiles` still
    # returned 501 for this section, so a filed prescription would have rendered nowhere and
    # filing is never undone. Spring lists and serves prescriptions now, and production has
    # run with this on since 2026-08-07 — a default of False meant the code contradicted
    # every deployment, and any environment brought up without the variable (a new service, a
    # replica, a local worker) rejected every prescription. That rejection is *routing*, so
    # nothing logs an error and nothing surfaces a failure: it fails exactly as silently as
    # the shared-SQS-queue bug did.
    #
    # **Turning this off is no longer a graceful kill switch, and it was one when it was
    # written.** Spring did not route typed uploads through intake then, so "off" left a
    # prescription sitting in the prescriptions table, merely unread. Spring added
    # `prescriptions` to `AiClient.PROCESSABLE`, so every prescription now lands in
    # `unclassified_files` first — and off means we reject it there, leaving it in
    # Unclassified rather than where the user filed it. Misfiled, not degraded.
    #
    # To stop prescriptions being processed, prefer Spring's side: drop `prescriptions` from
    # `PROCESSABLE` (or `app.ai.enabled=false` for everything). The document then goes
    # straight into its own table unread, which is the pre-AI behaviour and the graceful one.
    # This flag remains for a fast local or emergency stop, knowing what it costs.
    prescriptions_enabled: bool = True

    # --- Name matching ------------------------------------------------------
    # When on, a document whose printed patient name disagrees with the account holder's
    # is rejected with `name_mismatch` before filing, and the app offers the user three
    # ways out (keep / move to a family member / delete).
    #
    # ON by default. All three repos ship from one branch and deploy together, so the
    # partial-deploy hazard PRESCRIPTIONS_ENABLED was invented for does not arise here:
    # there is no window in which this service produces `name_mismatch` and the app has
    # no dialog to render it. CLAUDE.md records what the alternative costs — a flag
    # defaulting false while every deployment runs true means any environment brought up
    # without the variable silently stops checking, and that failure is invisible because
    # a passed gate logs nothing.
    #
    # Kept rather than deleted, as an emergency stop, exactly as prescriptions_enabled is.
    # Off means no name is ever checked and every document files as it did before this
    # existed — a safe degradation, unlike that flag's, because nothing is misfiled.
    name_matching_enabled: bool = True

    # --- On-demand analysis -------------------------------------------------
    # When on, the worker files a document and stops. The user sees it in its section
    # within seconds, correctly named and dated, and nothing expensive has run yet;
    # POST /v1/documents/{id}/analyze runs the rest when they ask for it.
    #
    # ON by default, for the reason `name_matching_enabled` is: all three repos ship from
    # one branch and deploy together, so the partial-deploy hazard does not arise. The
    # alternative is worse and is a mistake this codebase has already made once — a flag
    # defaulting false while every deployment sets it true means any environment brought
    # up without the variable silently behaves unlike all the others, and that failure is
    # invisible because a document that files and stops looks exactly like one that files
    # and is still working.
    #
    # **The ordering constraint this replaces the old default with:** the app must be able
    # to render the Analyse button before this service reaches an environment. Without it
    # every document files, stops, and has no way to be resumed — filed, named and dated,
    # but never read, with nothing on screen saying why. Deploy mhn-react before or with
    # this, never after.
    #
    # Kept rather than deleted, as an emergency stop. Off degrades SAFELY, unlike
    # `prescriptions_enabled`: every document simply runs the full pipeline on upload, as
    # it did before this existed, and nothing is misfiled or left anywhere unexpected.
    #
    # It must be set the SAME on the api and the worker. Only the worker reads it, but a
    # split would make the two services disagree about what a submitted document does.
    analysis_on_demand: bool = True

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
