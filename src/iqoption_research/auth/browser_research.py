"""Architectural feasibility analysis of browser-based delegated authentication.

This module documents the security, protocol, and architectural findings regarding
browser-based login mechanisms for the IQ Option Python trading bot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class BrowserLoginFeasibilityReport:
    """Structured report on browser-based authorization feasibility."""

    delegated_oauth_supported: bool
    pkce_supported: bool
    cookie_extraction_recommended: bool
    browser_scraping_recommended: bool
    verdict: str
    summary: str
    security_risks: tuple[str, ...]
    architectural_recommendation: str


BROWSER_FEASIBILITY_ANALYSIS: Final[BrowserLoginFeasibilityReport] = BrowserLoginFeasibilityReport(
    delegated_oauth_supported=False,
    pkce_supported=False,
    cookie_extraction_recommended=False,
    browser_scraping_recommended=False,
    verdict="SEPARATE_AUTHENTICATION_MANDATED",
    summary=(
        "IQ Option does not document or support a public delegated browser authorization "
        "mechanism (such as OAuth 2.0 with PKCE per RFC 7636 / RFC 8252 for native apps). "
        "Browser login cannot establish a session for the Python connector without exposing, "
        "extracting, or hijacking browser session cookies. Therefore, browser login must "
        "remain strictly decoupled from the Python bot's authentication subsystem."
    ),
    security_risks=(
        "1. Browser Sandboxing Violation: Extracting cookies from Chrome, Edge, or Firefox "
        "requires unauthorized process introspection into local browser profile directories "
        "(e.g. %LOCALAPPDATA%/Google/Chrome/User Data/Default/Network/Cookies).",
        "2. Cryptographic Incompatibility: Modern Chromium versions (Chrome 127+) employ "
        "App-Bound Encryption in addition to Windows DPAPI, restricting cookie decryption "
        "strictly to the calling browser binary.",
        "3. Session Fingerprint Mismatch: Reusing an SSID cookie obtained in a browser inside "
        "a Python aiohttp/websockets process creates a severe discrepancy in TLS JA3/JA4 "
        "fingerprints, HTTP/2 multiplexing signatures, and User-Agent headers, triggering "
        "broker fraud detection and automated account lockouts.",
        "4. Anti-Bot and CAPTCHA Interference: Browser automation frameworks (Selenium, "
        "Playwright) introduce automation markers (navigator.webdriver) that trigger "
        "anti-bot challenges that violate autonomous execution constraints.",
        "5. Secret Leakage & Fragility: Browser cookie files store long-lived tokens across "
        "all user domains on disk, introducing unnecessary exposure to adjacent host processes.",
    ),
    architectural_recommendation=(
        "Maintain an independent, owner-controlled authentication architecture for the Python "
        "connector. Use direct HTTPS credential authentication with interactive getpass "
        "prompts or native Windows Credential Manager storage. Do not implement browser cookie "
        "extraction, session hijacking, secret scraping, or unsupported browser instrumentation."
    ),
)


def get_browser_login_feasibility() -> BrowserLoginFeasibilityReport:
    """Return the authoritative feasibility assessment for browser-based login."""
    return BROWSER_FEASIBILITY_ANALYSIS
