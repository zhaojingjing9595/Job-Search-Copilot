"""
sheets_auth.py

description: One-time OAuth setup for Google Sheets access. First call opens a
browser for consent and caches the resulting refresh token to
.google_tokens/sheets.json; later calls reuse it silently (google-auth
refreshes the access token on its own when it expires).

Note: this uses the regular Sheets API rather than Google's remote Sheets MCP
server (https://sheetsmcp.googleapis.com/mcp/v1). That server is gated behind
the Google Workspace Developer Preview Program - verified 2026-08-09 that a
token with full scopes reads fine via sheets.googleapis.com but gets "The
caller does not have permission" from the MCP endpoint on every call. Swap
back to MCP once preview access is granted; log_job_match()'s signature is
designed to stay identical either way.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from core.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

_REDIRECT_PORT = 8765
_TOKEN_PATH = Path(__file__).resolve().parent.parent / ".google_tokens" / "sheets.json"

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",
]


def _client_config() -> dict:
    """Build the OAuth client config google-auth-oauthlib expects, from .env
    rather than a downloaded client_secret.json (keeps secrets in one place)."""
    return {
        "installed": {
            "client_id": os.environ["GOOGLE_SHEETS_CLIENT_ID"],
            "client_secret": os.environ["GOOGLE_SHEETS_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [f"http://localhost:{_REDIRECT_PORT}/"],
        }
    }


def materialize_oauth_client_file() -> Path:
    """Write the OAuth client config to a JSON file and return its path.

    The mcp-google-sheets server wants a Google-format OAuth client JSON file
    (CREDENTIALS_PATH) rather than inline values, but that file is just the
    same client id/secret already in .env - so generate it instead of asking
    the user to download and track a second copy of the same secret.
    """
    path = _TOKEN_PATH.parent / "oauth_client.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_client_config()), encoding="utf-8")
    return path


def get_sheets_credentials() -> Credentials:
    """Return usable Google credentials, running the browser consent flow only
    if there's no cached token (or the cached one can't be refreshed).

    Returns:
        Credentials: ready to hand to gspread.authorize().
    """
    creds = None
    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), _SCOPES)

    if creds and creds.valid:
        logger.info("Reusing cached Google Sheets credentials from %s", _TOKEN_PATH)
        return creds

    if creds and creds.expired and creds.refresh_token:
        logger.info("Refreshing expired Google Sheets access token")
        try:
            creds.refresh(Request())
        except Exception:
            logger.warning("Failed to refresh Google Sheets access token; falling back to OAuth consent flow")
            creds = None

    if not (creds and creds.valid):
        logger.info("No usable cached token; starting OAuth consent flow")
        try:
            flow = InstalledAppFlow.from_client_config(_client_config(), _SCOPES)
            creds = flow.run_local_server(port=_REDIRECT_PORT)
        except Exception:
            logger.exception("Google Sheets OAuth consent flow failed")
            raise

    _TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_PATH.write_text(creds.to_json(), encoding="utf-8")
    logger.info("Saved Google Sheets credentials to %s", _TOKEN_PATH)
    return creds
