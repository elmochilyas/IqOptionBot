"""Pytest configuration, shared fixtures, and offline mocks."""

import json
from pathlib import Path
from typing import Any

import pytest

from iqoption_research.security.credentials import Credentials


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture_json(fixtures_dir: Path) -> Any:
    def _loader(filename: str) -> Any:
        path = fixtures_dir / filename
        return json.loads(path.read_text(encoding="utf-8"))

    return _loader


@pytest.fixture
def mock_credentials() -> Credentials:
    return Credentials(email="researcher@example.com", password="mock_secure_password_123")
