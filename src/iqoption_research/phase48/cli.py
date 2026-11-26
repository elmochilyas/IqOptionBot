"""Phase 48 CLI commands for explicit owner campaign authorization and verification."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from iqoption_research.phase48.artifacts import default_models_dir, verify_exploratory_bundle
from iqoption_research.phase48.constants import CANDIDATE_SPECS
from iqoption_research.phase48.evidence import verify_phase48_session_integrity
from iqoption_research.phase48.governance import (
    CampaignUsageStore,
    authorize_campaign_contract,
    build_default_proposed_campaign_contract,
    default_campaign_contract_path,
    default_campaign_usage_path,
    load_campaign_contract,
    revoke_campaign_contract,
    save_campaign_contract,
)

phase48_app = typer.Typer(
    name="phase48",
    help="Phase 48 exploratory governance, owner authorization, and evidence verification.",
)


@phase48_app.command("status")
def status_cmd(
    repo_root: Annotated[Path, typer.Option("--repo-root", help="Repository root path.")] = Path(),
) -> None:
    """Display frozen exploratory model integrity and campaign authorization status."""
    root = repo_root.resolve()
    mdir = default_models_dir(root)
    bundles: dict[str, str] = {}
    for spec in CANDIDATE_SPECS:
        cid = spec.frozen_candidate_id
        bdir = mdir / cid
        if not bdir.is_dir():
            bundles[cid] = "MISSING"
            continue
        try:
            man = verify_exploratory_bundle(bdir)
            bundles[cid] = f"VERIFIED:{man.bundle_sha256[:12]}"
        except Exception as exc:  # noqa: BLE001
            bundles[cid] = f"INVALID:{exc}"

    cpath = default_campaign_contract_path(root)
    if cpath.is_file():
        contract = load_campaign_contract(cpath)
        usage = CampaignUsageStore(default_campaign_usage_path(root, contract.contract_id))
        consumed = (
            len(usage.list_consumed(contract.contract_id, contract.authorized_stage))
            if contract.authorized_stage
            else 0
        )
        contract_info = {
            "contract_id": contract.contract_id,
            "authorization_status": contract.authorization_status,
            "authorized_stage": contract.authorized_stage,
            "authorized_by": contract.authorized_by,
            "expires_at_utc": contract.expires_at_utc,
            "consumed_sessions_in_stage": consumed,
            "authorization_mechanism": contract.authorization_mechanism,
        }
    else:
        contract_info = {"authorization_status": "NOT_INITIALIZED"}

    typer.echo(json.dumps({"bundles": bundles, "contract": contract_info}, indent=2))


@phase48_app.command("authorize")
def authorize_cmd(
    account_id: str = typer.Option(..., "--account-id", help="Canonical Windows Vault account ID."),
    stage: str = typer.Option(
        ...,
        "--stage",
        help="Authorized stage (STAGE_A_15M, STAGE_B_120M, STAGE_C_240M, STAGE_D_480M).",
    ),
    owner: str = typer.Option(..., "--owner", help="Explicit owner identifier signing off."),
    hours_valid: int = typer.Option(24, "--hours-valid", help="Hours until authorization expires."),
    note: str = typer.Option("", "--note", help="Owner review note."),
    allow_unverified_vault: bool = typer.Option(
        False,
        "--allow-unverified-vault",
        help="Skip Windows Vault account lookup (for offline testing only).",
    ),
    repo_root: Annotated[Path, typer.Option("--repo-root", help="Repository root path.")] = Path(),
) -> None:
    """Explicitly authorize a bounded Phase 48 exploratory campaign contract."""
    root = repo_root.resolve()
    mdir = default_models_dir(root)
    digests: dict[str, str] = {}
    for spec in CANDIDATE_SPECS:
        cid = spec.frozen_candidate_id
        man = verify_exploratory_bundle(mdir / cid)
        digests[cid] = man.bundle_sha256

    if not allow_unverified_vault:
        from iqoption_research.auth.vault import WindowsCredentialVault

        store = WindowsCredentialVault().account_store()
        known_ids = {a.account_id for a in store.list_accounts()}
        if account_id not in known_ids:
            typer.echo(f"Error: account_id {account_id!r} not found in Windows Vault.", err=True)
            raise typer.Exit(code=2)

    cpath = default_campaign_contract_path(root)
    base_contract = (
        load_campaign_contract(cpath)
        if cpath.is_file()
        else build_default_proposed_campaign_contract(digests)
    )
    key_env = os.environ.get("IQOPTION_PHASE48_OWNER_KEY")
    hmac_key = key_env.encode("utf-8") if key_env else None
    signed = authorize_campaign_contract(
        base_contract,
        authorized_by=owner,
        allowed_account_ref=account_id,
        authorized_stage=stage,
        authorization_note=note,
        hours_valid=hours_valid,
        hmac_key=hmac_key,
    )
    save_campaign_contract(signed, cpath)
    typer.echo(
        json.dumps(
            {
                "status": signed.authorization_status,
                "contract_path": str(cpath),
                "authorized_stage": signed.authorized_stage,
                "expires_at_utc": signed.expires_at_utc,
                "authorization_mechanism": signed.authorization_mechanism,
                "contract_sha256": signed.contract_sha256,
            },
            indent=2,
        )
    )


@phase48_app.command("revoke")
def revoke_cmd(
    owner: str = typer.Option(..., "--owner", help="Owner revoking the contract."),
    reason: str = typer.Option("Manual revocation", "--reason", help="Revocation reason."),
    repo_root: Annotated[Path, typer.Option("--repo-root", help="Repository root path.")] = Path(),
) -> None:
    """Revoke any active Phase 48 exploratory campaign authorization."""
    root = repo_root.resolve()
    cpath = default_campaign_contract_path(root)
    if not cpath.is_file():
        typer.echo("No campaign contract file found.", err=True)
        raise typer.Exit(code=1)
    contract = load_campaign_contract(cpath)
    revoked = revoke_campaign_contract(contract, revoked_by=owner, reason=reason)
    save_campaign_contract(revoked, cpath)
    typer.echo(f"Revoked contract {revoked.contract_id} -> {revoked.authorization_status}")


@phase48_app.command("verify-session")
def verify_session_cmd(
    session_dir: Annotated[Path, typer.Argument(help="Path to Phase 48 session directory.")],
) -> None:
    """Independently verify a Phase 48 session's manifest, commitments, and outcomes."""
    ok, errs = verify_phase48_session_integrity(session_dir)
    typer.echo(json.dumps({"verified": ok, "errors": errs}, indent=2))
    if not ok:
        raise typer.Exit(code=1)
