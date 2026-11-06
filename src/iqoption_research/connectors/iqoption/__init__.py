"""IQ Option read-only connector implementation.

Modules:
    http_auth  — HTTPS login/logout with strict TLS
    protocol   — frame builders/parsers (no I/O)
    ws_client  — WebSocket transport, auth, heartbeat, reconnection
    client     — AsyncIQOption read-only facade
    exceptions — typed, sanitized errors
"""
