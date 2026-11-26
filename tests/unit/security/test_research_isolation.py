"""Regression tests for offline test isolation (network, credentials, sealed datasets)."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from pathlib import Path

import pytest

from tests.research_isolation import (
    CREDENTIAL_ENV_VARS,
    discover_sealed_holdout_paths,
    install_network_isolation_guard,
    install_sealed_read_guard,
    is_network_isolation_installed,
    scrub_credential_environment,
)


def test_network_guard_installed_without_catalog_db(tmp_path: Path) -> None:
    """Clean CI environments lack catalog.db; network isolation must still be active."""
    denied = install_sealed_read_guard(tmp_path)
    assert denied == set()
    assert is_network_isolation_installed() is True
    assert install_network_isolation_guard() is False

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(PermissionError, match="EXTERNAL_NETWORK_DENIED_BY_OFFLINE_TEST_GUARD"):
            sock.connect(("93.184.216.34", 443))
    finally:
        sock.close()


def test_discover_sealed_holdout_paths_with_mock_catalog(tmp_path: Path) -> None:
    """Dataset discovery returns empty when missing and resolves matching shards when present."""
    assert discover_sealed_holdout_paths(tmp_path) == set()

    audit_path = tmp_path / "docs/phase29/review/dataset_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps({
            "original_holdout": {
                "original_start_ts": 1000,
                "original_last_open_ts": 2000,
            }
        }),
        encoding="utf-8",
    )

    datasets_dir = tmp_path / "data/datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    shard_path = datasets_dir / "eurusd_holdout.parquet"
    shard_path.write_bytes(b"mock")

    catalog_db = datasets_dir / "catalog.db"
    conn = sqlite3.connect(catalog_db)
    try:
        conn.execute(
            "CREATE TABLE dataset_catalog (file_path TEXT, from_ts INTEGER, to_ts INTEGER)"
        )
        conn.execute(
            "INSERT INTO dataset_catalog VALUES (?, ?, ?)",
            ("eurusd_holdout.parquet", 1100, 1900),
        )
        conn.commit()
    finally:
        conn.close()

    discovered = discover_sealed_holdout_paths(tmp_path)
    assert discovered == {str(shard_path.resolve()).casefold()}


def test_scrub_credential_environment_prevents_local_secret_leak() -> None:
    """Existing real credential env vars in local shells must be scrubbed unconditionally."""
    for var in CREDENTIAL_ENV_VARS:
        assert os.environ.get(var, "") == ""

    fake_env = {
        "IQOPTION_EMAIL": "trader@real.example",
        "IQOPTION_PASSWORD": "super_secret_password",
        "IQOPTION_SSID": "ssid_12345",
        "IQOPTION_CUSTOM_TOKEN": "tok_abc",
        "OPENAI_API_KEY": "sk-real",
        "PATH": "/usr/bin",
    }
    scrubbed = scrub_credential_environment(fake_env)
    assert set(scrubbed) == {
        "IQOPTION_EMAIL",
        "IQOPTION_PASSWORD",
        "IQOPTION_SSID",
        "IQOPTION_CUSTOM_TOKEN",
        "OPENAI_API_KEY",
    }
    for key in scrubbed:
        assert fake_env[key] == ""
    assert fake_env["PATH"] == "/usr/bin"
