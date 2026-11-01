"""
Twitch OAuth token generator for the Rachel V5 pipeline.

Implements Authorization Code Flow:
  1. Reads TWITCH_CLIENT_ID + TWITCH_CLIENT_SECRET from .env (or prompts)
  2. Opens browser → user authorises in Twitch dashboard
  3. Local HTTP server on port 17563 captures the ?code= redirect
  4. Exchanges code → access + refresh tokens via Twitch API
  5. Writes TWITCH_TOKEN and TWITCH_REFRESH_TOKEN back to .env

Required scopes (used by twitch_listener.py):
  channel:read:subscriptions  — subscribe / subscription.gift events
  bits:read                   — cheer events
  channel:read:redemptions    — channel point redemption events

Usage:
  python twitch-auth.py
"""

import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from dotenv import load_dotenv, set_key

ENV_PATH     = Path(__file__).resolve().parent.parent / ".env"
REDIRECT_URI = "http://localhost:17563"
SCOPES       = "channel:read:subscriptions bits:read channel:read:redemptions"
AUTH_URL     = "https://id.twitch.tv/oauth2/authorize"
TOKEN_URL    = "https://id.twitch.tv/oauth2/token"

load_dotenv(ENV_PATH)
import os
CLIENT_ID     = os.getenv("TWITCH_CLIENT_ID",     "").strip()
CLIENT_SECRET = os.getenv("TWITCH_CLIENT_SECRET", "").strip()


def _prompt_if_missing(value: str, label: str) -> str:
    if value:
        return value
    entered = input(f"  Enter {label}: ").strip()
    if not entered:
        print(f"  ✗ {label} is required.")
        sys.exit(1)
    return entered


# ── Capture redirect ──────────────────────────────────────────────────────────

_captured_code: str | None = None


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress access log

    def do_GET(self):
        global _captured_code
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "error" in params:
            msg = params.get("error_description", [params["error"]])[0]
            body = f"<h1>Authorization denied</h1><p>{msg}</p>".encode()
            self.send_response(400)
        elif "code" in params:
            _captured_code = params["code"][0]
            body = b"<h1>Authorized!</h1><p>You can close this tab.</p>"
            self.send_response(200)
        else:
            body = b"<h1>Unexpected response</h1>"
            self.send_response(400)

        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body)
        # shut down the server after one request
        self.server._done = True


def _wait_for_code() -> str:
    server = HTTPServer(("localhost", 17563), _Handler)
    server._done = False
    print("  Waiting for browser redirect…")
    while not server._done:
        server.handle_request()
    server.server_close()
    if _captured_code is None:
        print("  ✗ Did not receive an authorization code.")
        sys.exit(1)
    return _captured_code


# ── Token exchange ────────────────────────────────────────────────────────────

def _exchange_code(client_id: str, client_secret: str, code: str) -> dict:
    r = requests.post(TOKEN_URL, data={
        "client_id":     client_id,
        "client_secret": client_secret,
        "code":          code,
        "grant_type":    "authorization_code",
        "redirect_uri":  REDIRECT_URI,
    }, timeout=15)
    if r.status_code != 200:
        print(f"  ✗ Token exchange failed ({r.status_code}): {r.text}")
        sys.exit(1)
    return r.json()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global CLIENT_ID, CLIENT_SECRET

    print("\n=== Twitch OAuth Token Generator ===\n")

    CLIENT_ID     = _prompt_if_missing(CLIENT_ID,     "Client ID")
    CLIENT_SECRET = _prompt_if_missing(CLIENT_SECRET, "Client Secret")

    params = urlencode({
        "client_id":     CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "scope":         SCOPES,
        "force_verify":  "true",
    })
    auth_link = f"{AUTH_URL}?{params}"

    print(f"  Opening browser for authorization…")
    print(f"  URL: {auth_link}\n")
    webbrowser.open(auth_link)

    code   = _wait_for_code()
    tokens = _exchange_code(CLIENT_ID, CLIENT_SECRET, code)

    access_token  = tokens.get("access_token",  "")
    refresh_token = tokens.get("refresh_token", "")
    expires_in    = tokens.get("expires_in",    0)

    print(f"\n  ✅ Access token : {access_token}")
    print(f"  ✅ Refresh token: {refresh_token}")
    print(f"     Expires in   : {expires_in} s\n")

    set_key(str(ENV_PATH), "TWITCH_TOKEN",         access_token)
    set_key(str(ENV_PATH), "TWITCH_REFRESH_TOKEN", refresh_token)
    if CLIENT_ID != os.getenv("TWITCH_CLIENT_ID", ""):
        set_key(str(ENV_PATH), "TWITCH_CLIENT_ID", CLIENT_ID)
    if CLIENT_SECRET != os.getenv("TWITCH_CLIENT_SECRET", ""):
        set_key(str(ENV_PATH), "TWITCH_CLIENT_SECRET", CLIENT_SECRET)

    print(f"  ✅ .env updated — TWITCH_TOKEN and TWITCH_REFRESH_TOKEN written.")
    print(f"\n  Run: python twitch_listener.py")


if __name__ == "__main__":
    main()
