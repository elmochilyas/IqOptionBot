"""Connectors package.

Only read-only connectors live here. Broker-specific code is isolated
under :mod:`iqoption_research.connectors.iqoption`. A deterministic
offline simulator is provided in :mod:`iqoption_research.connectors.simulation`.

This package intentionally does not re-export concrete classes at import
time to keep module dependencies acyclic and auditable.
"""
