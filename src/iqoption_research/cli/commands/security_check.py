"""Offline security checks.

Runs entirely locally: validates TLS configuration, endpoint
allowlisting, absence of insecure code patterns, read-only posture,
dependency pinning, and secret non-persistence.
"""

from __future__ import annotations

import re
import ssl
from dataclasses import dataclass
from pathlib import Path

from iqoption_research.config.settings import Settings
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.security.endpoints import validate_endpoint_allowlist
from iqoption_research.security.tls import (
    create_tls_context,
    validate_https_endpoint,
    validate_wss_endpoint,
)

# Actual insecure code constructs. This module is excluded from the scan.
INSECURE_PATTERNS: dict[str, re.Pattern[str]] = {
    "verify_false": re.compile(r"verify\s*=\s*False"),
    "cert_none": re.compile(r"\bCERT_NONE\b"),
    "check_hostname_false": re.compile(r"check_hostname\s*=\s*False"),
    "unverified_context": re.compile(r"_create_unverified_context"),
    "disabled_tls": re.compile(r"\bssl\s*=\s*None\b"),
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def _project_root() -> Path:
    # src/iqoption_research/cli/commands/security_check.py -> parents[3] == src
    return Path(__file__).resolve().parents[3]


def check_tls_context() -> CheckResult:
    try:
        context = create_tls_context()
    except Exception as exc:  # noqa: BLE001 - report failure safely
        return CheckResult("tls_context", False, f"Could not create TLS context: {exc}")
    problems: list[str] = []
    if not context.check_hostname:
        problems.append("hostname verification disabled")
    if context.verify_mode != ssl.CERT_REQUIRED:
        problems.append("certificate verification is not required")
    if context.minimum_version < ssl.TLSVersion.TLSv1_2:
        problems.append("TLS version below 1.2 permitted")
    if problems:
        return CheckResult("tls_context", False, "; ".join(problems))
    return CheckResult(
        "tls_context",
        True,
        "hostname + certificate verification required, minimum TLS 1.2",
    )


def check_endpoint_allowlist() -> CheckResult:
    try:
        validate_endpoint_allowlist()
    except Exception as exc:  # noqa: BLE001
        return CheckResult("endpoint_allowlist", False, f"Allowlist invalid: {exc}")

    rejected = 0
    for bad in ("https://evil.example/login", "wss://evil.example/ws"):
        try:
            if bad.startswith("https"):
                validate_https_endpoint(bad)
            else:
                validate_wss_endpoint(bad)
        except ValueError:
            rejected += 1
    if rejected != 2:
        return CheckResult("endpoint_allowlist", False, "Non-allowlisted endpoint was accepted")
    return CheckResult(
        "endpoint_allowlist",
        True,
        "hardcoded IQ Option endpoints only; unknown URLs rejected",
    )


def check_no_insecure_tls_patterns() -> CheckResult:
    root = _project_root() / "iqoption_research"
    self_path = Path(__file__).resolve()
    findings: list[str] = []
    for path in sorted(root.rglob("*.py")):
        if path.resolve() == self_path:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for name, pattern in INSECURE_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{path.name}:{name}")
    if findings:
        return CheckResult(
            "no_insecure_tls_patterns",
            False,
            "Insecure TLS constructs found: " + ", ".join(findings),
        )
    return CheckResult(
        "no_insecure_tls_patterns",
        True,
        "no verify=False / CERT_NONE / disabled hostname verification",
    )


def check_read_only_posture() -> CheckResult:
    try:
        AsyncIQOption.assert_read_only()
    except Exception as exc:  # noqa: BLE001
        return CheckResult("read_only_posture", False, str(exc))
    return CheckResult(
        "read_only_posture",
        True,
        "no buy/sell/close/change_balance attributes on AsyncIQOption",
    )


def check_dependency_pinning() -> CheckResult:
    requirements = _project_root().parent / "requirements.txt"
    if not requirements.exists():
        return CheckResult("dependency_pinning", False, f"Missing {requirements.name}")
    unpinned: list[str] = []
    for line in requirements.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "==" not in stripped:
            unpinned.append(stripped)
    if unpinned:
        return CheckResult(
            "dependency_pinning",
            False,
            "Unpinned requirements: " + ", ".join(unpinned),
        )
    return CheckResult("dependency_pinning", True, "all runtime dependencies pinned with ==")


def check_no_credential_settings() -> CheckResult:
    fields = set(Settings.model_fields)
    forbidden = {"password", "passwd", "email", "token", "ssid", "cookie"}
    leaked = fields & forbidden
    if leaked:
        return CheckResult(
            "no_credential_settings",
            False,
            f"Settings exposes secret fields: {sorted(leaked)}",
        )
    return CheckResult(
        "no_credential_settings",
        True,
        "Settings has no credential fields; getpass is required at runtime",
    )


def check_offline_default() -> CheckResult:
    settings = Settings()
    if not settings.offline:
        return CheckResult(
            "offline_default",
            False,
            "offline mode is not the default (IQOPTION_OFFLINE must default true)",
        )
    return CheckResult("offline_default", True, "network I/O disabled by default")


def run_security_checks() -> list[CheckResult]:
    """Run all offline security checks."""
    return [
        check_tls_context(),
        check_endpoint_allowlist(),
        check_no_insecure_tls_patterns(),
        check_read_only_posture(),
        check_dependency_pinning(),
        check_no_credential_settings(),
        check_offline_default(),
    ]
