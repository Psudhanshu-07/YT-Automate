#!/usr/bin/env python3
"""One-time helper: turn a Google OAuth Desktop client into a refresh token.

Run this locally (never in CI) and paste the printed value into the
YT_REFRESH_TOKEN GitHub secret.

    # with a downloaded OAuth "Desktop app" client JSON
    python get_refresh_token.py --client-secrets client_secret.json

    # or with the client id/secret already exported
    export YT_CLIENT_ID="....apps.googleusercontent.com"
    export YT_CLIENT_SECRET="...."
    python get_refresh_token.py

Sign in with the Google account that owns the YouTube channel, approve the
"Manage your YouTube videos" scope, and copy the refresh token.

Security notes:
* The token grants upload access to your channel. Treat it like a password.
* This script prints it to your terminal; do not paste it into chats, issues or
  commits. Use --write-env to store it in the untracked .env file instead.
* While the OAuth consent screen is in "Testing", refresh tokens expire after
  7 days. Publish the app (or keep re-running this script) for long-lived tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCOPE = "https://www.googleapis.com/auth/youtube.upload"
SCOPES = [SCOPE]
DEFAULT_CLIENT_FILE = "client_secret.json"


def _load_client_config(path: str) -> dict:
    client_file = Path(path).expanduser()
    if not client_file.is_file():
        raise SystemExit(
            f"Client secrets file not found: {client_file}\n"
            "Download it from Google Cloud Console -> APIs & Services -> Credentials -> "
            "your OAuth 2.0 Client ID (Desktop app) -> Download JSON."
        )
    try:
        payload = json.loads(client_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemExit(f"{client_file} is not valid JSON: {exc}") from exc
    if "installed" not in payload and "web" not in payload:
        raise SystemExit(
            f"{client_file} does not look like an OAuth client file "
            "(expected an 'installed' or 'web' key)."
        )
    return payload


def _client_config_from_env() -> dict:
    client_id = os.environ.get("YT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("YT_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SystemExit(
            "No --client-secrets file given and YT_CLIENT_ID / YT_CLIENT_SECRET are not both set.\n"
            "Either download the Desktop client JSON and pass it with --client-secrets, or export "
            "both variables first."
        )
    return {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": ["http://localhost"],
        }
    }


def _write_env(path: Path, refresh_token: str) -> None:
    """Insert/replace YT_REFRESH_TOKEN in an untracked .env file."""

    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()

    replaced = False
    for index, line in enumerate(lines):
        if line.strip().startswith("YT_REFRESH_TOKEN="):
            lines[index] = f"YT_REFRESH_TOKEN={refresh_token}"
            replaced = True
            break
    if not replaced:
        lines.append(f"YT_REFRESH_TOKEN={refresh_token}")

    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")
    try:  # POSIX only; harmless elsewhere.
        path.chmod(0o600)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a YouTube upload refresh token via the local OAuth consent flow."
    )
    parser.add_argument(
        "--client-secrets",
        default=os.environ.get("YT_CLIENT_SECRETS_FILE", DEFAULT_CLIENT_FILE),
        help="Path to the OAuth Desktop client JSON (default: client_secret.json).",
    )
    parser.add_argument(
        "--write-env",
        action="store_true",
        help="Write the token into .env (untracked) instead of only printing it.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path used by --write-env (default: .env).",
    )
    args = parser.parse_args(argv)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise SystemExit(
            "google-auth-oauthlib is not installed.\nRun: pip install -r requirements.txt"
        ) from exc

    if args.client_secrets and Path(args.client_secrets).expanduser().is_file():
        flow = InstalledAppFlow.from_client_secrets_file(str(Path(args.client_secrets).expanduser()), SCOPES)
    else:
        flow = InstalledAppFlow.from_client_config(_client_config_from_env(), SCOPES)

    print("Opening your browser. Sign in with the account that owns the YouTube channel.")
    print("If the app is unverified you may need to click 'Advanced' -> 'Go to <app> (unsafe)'.")
    print("Make sure the account is listed under OAuth consent screen -> Test users.\n")

    credentials = flow.run_local_server(
        port=0,
        open_browser=True,
        access_type="offline",
        prompt="consent",
        authorization_prompt_message="Visit this URL to authorise shorts-agent: {url}",
        success_message="Authorised. You can close this tab and return to the terminal.",
    )

    refresh_token = getattr(credentials, "refresh_token", None)
    if not refresh_token:
        raise SystemExit(
            "Google did not return a refresh token.\n"
            "Remove the app's previous access at https://myaccount.google.com/permissions and run "
            "this script again (it forces prompt=consent)."
        )

    print("\n" + "=" * 72)
    print("Add this as the YT_REFRESH_TOKEN GitHub secret:")
    print("=" * 72)
    print(refresh_token)
    print("=" * 72)
    print(f"Scope granted: {SCOPE}")
    print("\nKeep it private. Revoke access any time at https://myaccount.google.com/permissions")

    if args.write_env:
        env_path = Path(args.env_file)
        _write_env(env_path, refresh_token)
        print(f"\nWrote YT_REFRESH_TOKEN to {env_path} (make sure .env is git-ignored).")

    print(
        "\nIf your OAuth consent screen is still in 'Testing', this token expires in 7 days. "
        "Publish the app (OAuth consent screen -> Publish app) for a long-lived token."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
