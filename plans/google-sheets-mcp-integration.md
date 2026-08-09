# Google Sheets integration for Job-Search-Copilot (Phase 3)

## OUTCOME (2026-08-09): official MCP server blocked, shipped via regular Sheets API

The plan below (Google's remote Sheets MCP server as a second MCP server) was implemented
and **does not work** — `https://sheetsmcp.googleapis.com/mcp/v1` is gated behind the
Google Workspace Developer Preview Program, which requires a separate application and
~2 days' verification. Symptom: every tool call, including read-only `get_spreadsheet`,
returns `"The caller does not have permission"`.

Ruled out as causes, empirically:
- Scopes — token carries all five (`drive`, `drive.file`, `drive.readonly`, `spreadsheets`, `spreadsheets.readonly`), confirmed via `oauth2/v3/tokeninfo`.
- Identity / file ownership — same failure on a brand-new blank sheet owned by the user.
- Wrong spreadsheet ID or tab name — both verified correct.
- **Decisive test:** the *same access token*, at the same moment, reads both sheets fine
  through `sheets.googleapis.com` (returned the real header row) while every call to
  `sheetsmcp.googleapis.com` fails. The credentials are good; the MCP endpoint alone rejects us.

**Shipped instead:** the regular Sheets API via `gspread`, reusing the same OAuth client.
- `tools/sheets_auth.py` — `google-auth-oauthlib` `InstalledAppFlow`, token cached to `.google_tokens/sheets.json` (gitignored).
- `tools/sheets_logging.py` — `log_job_match(row, spreadsheet_id=None)` using `worksheet.append_row()`; no read-then-write dance needed.
- `tools/mcp_config.py` — `sheets` entry removed, with a comment explaining why and how to restore it.

To switch back once preview access is granted, only `log_job_match`'s body changes — the
signature is deliberately identical, so Phase 4's `@tool` wrapper and Phase 5's graph are unaffected.

---

## Original plan (kept for reference / post-preview-access restoration)

## Context

This is Phase 3 of `job-application-copilot-guide.md` — the "sheets tool" that appends matched job postings (company, title, match reasoning, status, link, date) to a Google Sheet. Phase 2 (JobSpy MCP) is done: `tools/mcp_config.py` holds an `MCP_SERVERS` registry, `tools/mcp_client.py` exposes a shared `MultiServerMCPClient`, and `tools/jobs_searching.py` filters that client's tools down to the `jobspy_`-prefixed ones. The registry's `# Future: adzuna...` comment was written specifically so a second server slots in with no changes to existing code.

**Decision (user-confirmed):** use Google's **official remote Sheets MCP server** as that second server, rather than a plain `gspread` function or a third-party community MCP server. This is a deliberate tradeoff — it's heavier to set up than a service account (OAuth consent flow + browser-based auth vs. a JSON key file) — but keeps the whole project on one architectural pattern (MCP host → registry → filtered tool loader) instead of mixing in a one-off direct API call.

**Server facts (verified against Google's docs, 2026-08):**
- Endpoint: `https://sheetsmcp.googleapis.com/mcp/v1`
- Transport: HTTP (streamable-http in MCP terms)
- Auth: OAuth 2.0 — scopes `drive.readonly`, `drive.file`, `spreadsheets.readonly`, `spreadsheets`
- Tools exposed: `get_values`, `get_spreadsheet`, `update_values`, `update_formulas`, `update_spreadsheet`, `insert_dimension` — **no dedicated "append row" tool**, so appending means: read current values with `get_values` to find the next empty row, then write the new row with `update_values`. (This read-before-write also sets us up for the Phase 6 dedup check later — no wasted work.)
- Google's docs frame this for "AI applications like Google Antigravity and Claude" but note generic MCP clients can connect — we're the unverified case here, flagged as a risk below.

## Prerequisites (manual, before/alongside implementation)

1. **Google Sheet** — create the tracking sheet (or use an existing one), note its spreadsheet ID from the URL. Add header row: `company, title, match_reasoning, status, link, date`.
2. **OAuth client in the same dedicated GCP project** used for Gemini (per Phase 0 — keep it billing-free):
   - Enable the Google Sheets API and Google Drive API for the project.
   - Configure the OAuth consent screen (Internal if using a Workspace account, External + test user otherwise).
   - Create an OAuth 2.0 **Desktop app** client ID/secret — the redirect URI for a local script is a loopback address (e.g. `http://localhost:8765/callback`), not `https://claude.ai/...` (that one's Claude-specific).
3. **Risk to confirm early:** verify the installed `mcp` SDK (pinned transitively via `langchain-mcp-adapters==0.3.0`, which requires `mcp>=1.9.2`) actually supports OAuth on `streamable_http` transport end-to-end with a custom Python host — this is the part of the plan most likely to need adjustment once we're in the code, since Google's docs don't document non-Claude/non-Antigravity clients in detail.

## Implementation

### 1. `.env` — new variables
```
GOOGLE_SHEETS_CLIENT_ID=...
GOOGLE_SHEETS_CLIENT_SECRET=...
GOOGLE_SHEET_ID=...          # the tracking spreadsheet's ID
```

### 2. `.gitignore` — add token cache
The OAuth flow caches a refresh token locally after the first browser consent (so subsequent runs don't reprompt). Add e.g. `.mcp_tokens/` to `.gitignore` before it's ever created.

### 3. `tools/sheets_auth.py` (new) — OAuth provider for the MCP client
Builds an `OAuthClientProvider` (from the `mcp` SDK's `mcp.client.auth`) configured with the client ID/secret from `.env`, the loopback redirect URI, the four Sheets/Drive scopes, and a `TokenStorage` implementation that persists to `.mcp_tokens/sheets.json`. First run opens a browser for consent; later runs reuse the cached refresh token silently.

### 4. `tools/mcp_config.py` — register the second server
```python
MCP_SERVERS = {
    "jobspy": { ... unchanged ... },
    "sheets": {
        "transport": "streamable_http",
        "url": "https://sheetsmcp.googleapis.com/mcp/v1",
        "auth": <OAuthClientProvider from tools/sheets_auth.py>,
    },
}
```
No changes needed to `tools/mcp_client.py` — `get_mcp_tools()` already loads from every registered server.

### 5. `tools/sheets_logging.py` (new) — mirrors `tools/jobs_searching.py`'s shape
- `get_sheets_tools()` — filters the shared client's tools down to `sheets_`-prefixed ones (`tool_name_prefix=True` on the client, same as jobspy).
- `log_job_match(spreadsheet_id, row: dict)` — plain async helper (not yet an `@tool`, per the guide's Phase 3 vs Phase 4 split): calls `sheets_get_values` to find the next empty row, then `sheets_update_values` to write `[company, title, match_reasoning, status, link, date]` into it.
- `_smoke_test()` — standalone check: load tools, print names, call `log_job_match` with one fake row, confirm it lands in the real sheet.

**Milestone:** `python -m tools.sheets_logging` — first run pops a browser consent screen once, then a real row appears in the real Google Sheet.

## Explicitly out of scope (this pass)

- Wrapping `log_job_match` as a LangChain `@tool` and binding it to Gemini — Phase 4.
- Dedup-by-link and `DRY_RUN` guardrails — Phase 6 (though the read-before-write shape here is built with that in mind).
- The `gemini-3.5-flash` model id issue in `constants.py`/`main.py` — pre-existing, unrelated.

## Verification

1. `.env` has the three new variables; `.gitignore` has the token cache path.
2. `python -m tools.sheets_logging` from repo root — expect one browser consent prompt on first run, then a JSON/rich-printed confirmation and a real new row in the sheet.
3. Re-run the same command — expect no browser prompt (cached token reused).
