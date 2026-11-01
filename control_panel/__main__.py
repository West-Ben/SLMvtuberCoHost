"""
python -m control_panel [--port 8060] [--no-browser]

Run from vtuberMCP/. Binds 127.0.0.1 only. Ctrl+C stops the panel and every
service it started (services adopted as "external" are left running).
"""

import argparse
import threading
import webbrowser

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="Rachel control panel")
    ap.add_argument("--port", type=int, default=8060)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}/"
    if not a.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"Rachel control panel → {url}")
    uvicorn.run("control_panel.app:app", host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
