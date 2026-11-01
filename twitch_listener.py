"""
twitch_listener.py — Twitch EventSub + stream info poller
──────────────────────────────────────────────────────────
Does NOT handle chat (comes via Streamlabs overlay → obs_bridge).

Two jobs:
  poll_stream_info  — GET /helix/streams every 60 s → POST obs_bridge /twitch-info
  eventsub_listener — WebSocket EventSub → POST obs_bridge /twitch-event

Required env vars:
  TWITCH_CLIENT_ID      App client ID
  TWITCH_TOKEN          User access token (OAuth, no "oauth:" prefix)
  TWITCH_CHANNEL        Channel login name (lowercase, no #)

Setup:
  pip install requests
  set TWITCH_CLIENT_ID=...
  set TWITCH_TOKEN=...
  set TWITCH_CHANNEL=your_channel_name
  python twitch_listener.py
"""

import json
import logging
import os
import threading
import time

import requests
import websocket

from config import OBS_BRIDGE_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [twitch] %(message)s")
log = logging.getLogger(__name__)

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID", "")
TOKEN     = os.getenv("TWITCH_TOKEN",     "")
CHANNEL   = os.getenv("TWITCH_CHANNEL",   "")

HELIX     = "https://api.twitch.tv/helix"
EVENTSUB_WS = "wss://eventsub.wss.twitch.tv/ws"

SUBSCRIPTIONS = [
    "channel.subscribe",
    "channel.subscription.gift",
    "channel.cheer",
    "channel.raid",
    "channel.channel_points_custom_reward_redemption.add",
    "channel.update",
]

# ─── Helpers ──────────────────────────────────────────────

def _headers() -> dict:
    return {
        "Authorization": f"Bearer {TOKEN}",
        "Client-Id":     CLIENT_ID,
    }


def _post_bridge(endpoint: str, payload: dict):
    try:
        requests.post(f"{OBS_BRIDGE_URL}{endpoint}", json=payload, timeout=3)
    except Exception as e:
        log.debug(f"bridge POST {endpoint} failed: {e}")


def _get_broadcaster_id() -> str | None:
    """Resolve channel login name → broadcaster user ID."""
    try:
        r = requests.get(f"{HELIX}/users", params={"login": CHANNEL}, headers=_headers(), timeout=10)
        data = r.json().get("data", [])
        return data[0]["id"] if data else None
    except Exception as e:
        log.error(f"Failed to get broadcaster ID: {e}")
        return None


# ─── Stream info poller ───────────────────────────────────

def poll_stream_info(broadcaster_id: str):
    """Poll /helix/streams every 60 s and push viewer count + game to obs_bridge."""
    while True:
        try:
            r = requests.get(
                f"{HELIX}/streams",
                params={"user_id": broadcaster_id},
                headers=_headers(),
                timeout=10,
            )
            streams = r.json().get("data", [])
            if streams:
                s = streams[0]
                _post_bridge("/twitch-info", {
                    "viewers": s.get("viewer_count", 0),
                    "game":    s.get("game_name", ""),
                })
                log.info(f"Stream: {s.get('game_name')} | {s.get('viewer_count')} viewers")
            else:
                _post_bridge("/twitch-info", {"viewers": 0, "game": ""})
        except Exception as e:
            log.warning(f"poll_stream_info: {e}")
        time.sleep(60)


# ─── EventSub WebSocket listener ─────────────────────────

def eventsub_listener(broadcaster_id: str):
    """Connect to Twitch EventSub WebSocket and relay events to obs_bridge."""
    session_id = None

    def subscribe(sid: str):
        for sub_type in SUBSCRIPTIONS:
            condition = {"broadcaster_user_id": broadcaster_id}
            if sub_type == "channel.raid":
                condition = {"to_broadcaster_user_id": broadcaster_id}
            try:
                r = requests.post(
                    f"{HELIX}/eventsub/subscriptions",
                    headers={**_headers(), "Content-Type": "application/json"},
                    json={
                        "type":      sub_type,
                        "version":   "1",
                        "condition": condition,
                        "transport": {"method": "websocket", "session_id": sid},
                    },
                    timeout=10,
                )
                if r.status_code in (200, 202):
                    log.info(f"Subscribed: {sub_type}")
                else:
                    log.warning(f"Subscribe {sub_type} → {r.status_code}: {r.text[:120]}")
            except Exception as e:
                log.warning(f"Subscribe {sub_type} failed: {e}")

    def on_message(ws, raw):
        try:
            msg      = json.loads(raw)
            msg_type = msg.get("metadata", {}).get("message_type", "")

            if msg_type == "session_welcome":
                nonlocal session_id
                session_id = msg["payload"]["session"]["id"]
                log.info(f"EventSub session: {session_id}")
                subscribe(session_id)

            elif msg_type == "notification":
                sub_type = msg["payload"]["subscription"]["type"]
                event    = msg["payload"]["event"]
                _post_bridge("/twitch-event", {"type": sub_type, "data": event})
                log.info(f"Event: {sub_type} → {list(event.keys())}")

            elif msg_type == "session_reconnect":
                url = msg["payload"]["session"]["reconnect_url"]
                log.info(f"EventSub reconnect → {url}")
                ws.close()

        except Exception as e:
            log.warning(f"on_message: {e}")

    def on_error(ws, err):
        log.warning(f"EventSub WS error: {err}")

    def on_close(ws, code, msg):
        log.info(f"EventSub WS closed ({code}), reconnecting in 5 s...")

    while True:
        try:
            ws = websocket.WebSocketApp(
                EVENTSUB_WS,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            log.warning(f"eventsub_listener crashed: {e}")
        time.sleep(5)


# ─── Entry point ──────────────────────────────────────────

if __name__ == "__main__":
    if not all([CLIENT_ID, TOKEN, CHANNEL]):
        log.error("TWITCH_CLIENT_ID, TWITCH_TOKEN, and TWITCH_CHANNEL must all be set.")
        raise SystemExit(1)

    broadcaster_id = _get_broadcaster_id()
    if not broadcaster_id:
        log.error(f"Could not resolve broadcaster ID for channel '{CHANNEL}'.")
        raise SystemExit(1)

    log.info(f"Channel: {CHANNEL} (ID {broadcaster_id})")

    threading.Thread(
        target=poll_stream_info,
        args=(broadcaster_id,),
        name="stream-info",
        daemon=True,
    ).start()

    eventsub_listener(broadcaster_id)   # blocks; reconnects on drop
