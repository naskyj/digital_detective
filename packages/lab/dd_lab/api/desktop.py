"""pywebview desktop shell: run the FastAPI app in a thread, open a native window."""

from __future__ import annotations

import threading
import time

import httpx


def launch(app, host: str, port: int) -> None:
    import uvicorn
    import webview  # raises ImportError if the `desktop` extra isn't installed

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://{host}:{port}"
    for _ in range(100):
        try:
            if httpx.get(f"{base}/health", timeout=0.5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)

    window = webview.create_window("Digital Detective", base, width=1200, height=820)
    webview.start()
    server.should_exit = True
    _ = window
