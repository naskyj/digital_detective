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
    window.expose(_make_pick_file(window))
    window.expose(_make_pick_folder(window))
    webview.start()
    server.should_exit = True


def _make_pick_file(window):
    """Expose a native "open file" dialog to the UI as `pywebview.api.pick_file()`
    — lets the investigator browse to a disk image / memory dump instead of
    typing a path, without ever copying the (potentially huge) file: the
    dialog just returns the real local path, which the existing image-import
    flow already reads directly."""

    def pick_file(file_types: list[str] | None = None) -> str | None:
        import webview

        types = tuple(file_types) if file_types else (
            # pywebview's filter parser only accepts [\w ] in the description
            # (see webview.util.parse_file_type) - no "/" or other punctuation,
            # or it rejects the WHOLE call with "not a valid file filter".
            "Disk and memory images (*.raw;*.dd;*.img;*.001;*.e01;*.E01;*.vmdk;*.vhd;*.vhdx)",
            "All files (*.*)",
        )
        result = window.create_file_dialog(webview.FileDialog.OPEN, file_types=types)
        return result[0] if result else None

    return pick_file


def _make_pick_folder(window):
    """Expose a native "choose folder" dialog as `pywebview.api.pick_folder()`
    — for the image-import path's other two accepted shapes (an OS-root-
    shaped folder, or a plain folder of already-extracted app subfolders),
    where a file picker doesn't apply."""

    def pick_folder() -> str | None:
        import webview

        result = window.create_file_dialog(webview.FileDialog.FOLDER)
        return result[0] if result else None

    return pick_folder
