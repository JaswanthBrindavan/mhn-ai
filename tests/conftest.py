import os

# Settings are read at import time, so the environment must be prepared before any
# app module loads. Tests never touch the developer's real .env values.
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

from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)
