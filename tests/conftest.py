import os

# Settings are read at import time, so the environment must be prepared before any
# app module loads.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/mhn_ai"
)
# AWS is faked with moto, never reached over the network. AWS_ENDPOINT_URL must be
# cleared: botocore honours it directly, so a developer's LocalStack setting would
# send "mocked" calls to a real service and leak state between test runs.
os.environ.pop("AWS_ENDPOINT_URL", None)
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "ap-south-1"
os.environ.setdefault("LOG_LEVEL", "WARNING")
# create_app() fails closed without this; a test-only value, never a real secret.
os.environ.setdefault("MHN_SERVICE_TOKEN", "test-service-token-at-least-32-chars-long")

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

# Clearing os.environ above is not enough on its own: pydantic-settings also reads the .env
# *file* for every field an environment variable does not set, so a developer's local
# CLASSIFICATION_PROVIDER=gemini or IDEAL_RANGES_ENABLED=true would silently decide what the
# suite asserts. Detaching the file covers every Settings() a test builds — including the
# ones inside test modules — instead of neutralising one field at a time. Safe here because
# nothing constructs Settings at import time; get_settings() is called inside create_app().
Settings.model_config["env_file"] = None


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)
