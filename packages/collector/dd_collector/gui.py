"""Minimal Tkinter GUI for the collector (stdlib only, keeps the exe small).

Mirrors the original Digital Detective Collection Tool: pick applications, an
output location, options, then Start. All work runs on a worker thread so the
log pane stays live.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from dd_collector.acquire import acquire
from dd_collector.discover import ElectronApp, discover_apps


class CollectorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Digital Detective — Collector")
        root.geometry("720x560")
        self._apps: list[ElectronApp] = []
        self._app_vars: list[tuple[tk.BooleanVar, ElectronApp]] = []
        self._log_q: queue.Queue[str] = queue.Queue()
        self._build()
        self._refresh_apps()
        self.root.after(100, self._drain_log)

    # ------------------------------------------------------------------ ui
    def _build(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="both", expand=True)

        ttk.Label(top, text="Applications", font=("", 12, "bold")).pack(anchor="w")
        self.apps_frame = ttk.Frame(top)
        self.apps_frame.pack(fill="x", pady=(2, 8))
        ttk.Button(top, text="Rescan", command=self._refresh_apps).pack(anchor="w")

        opts = ttk.Frame(top)
        opts.pack(fill="x", pady=10)
        self.var_memory = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Capture memory (processes + full RAM)",
                        variable=self.var_memory).pack(anchor="w")

        row = ttk.Frame(top)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text="Operator:").pack(side="left")
        self.var_operator = tk.StringVar()
        ttk.Entry(row, textvariable=self.var_operator, width=32).pack(side="left", padx=6)

        row2 = ttk.Frame(top)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Output:").pack(side="left")
        self.var_output = tk.StringVar()
        ttk.Entry(row2, textvariable=self.var_output).pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row2, text="Browse…", command=self._pick_output).pack(side="left")

        self.start_btn = ttk.Button(top, text="Start Collection", command=self._start)
        self.start_btn.pack(anchor="w", pady=(10, 6))

        ttk.Label(top, text="Log", font=("", 11, "bold")).pack(anchor="w")
        self.log = tk.Text(top, height=14, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    def _refresh_apps(self) -> None:
        for w in self.apps_frame.winfo_children():
            w.destroy()
        self._apps = discover_apps()
        self._app_vars = []
        if not self._apps:
            ttk.Label(self.apps_frame, text="(no Electron applications detected)").pack(anchor="w")
            return
        for app in self._apps:
            var = tk.BooleanVar(value=False)
            label = f"{app.name}" + (f"  v{app.version}" if app.version else "")
            if app.running:
                label += f"  · running (pids {', '.join(map(str, app.pids))})"
            ttk.Checkbutton(self.apps_frame, text=label, variable=var).pack(anchor="w")
            self._app_vars.append((var, app))

    def _pick_output(self) -> None:
        d = filedialog.askdirectory(title="Choose output location (e.g. external drive)")
        if d:
            self.var_output.set(d)

    # ------------------------------------------------------------------ run
    def _start(self) -> None:
        selected = [app for var, app in self._app_vars if var.get()]
        out = self.var_output.get().strip()
        if not selected:
            messagebox.showwarning("Collector", "Select at least one application.")
            return
        if not out or not Path(out).is_dir():
            messagebox.showwarning("Collector", "Choose a valid output directory.")
            return
        self.start_btn.config(state="disabled")
        self._log("Starting…")
        threading.Thread(
            target=self._worker, args=(selected, out, self.var_operator.get(),
                                       self.var_memory.get()), daemon=True,
        ).start()

    def _worker(self, apps, out, operator, memory) -> None:
        try:
            result = acquire(
                apps, out, operator=operator, collect_memory=memory,
                progress=lambda m: self._log_q.put(m),
            )
            self._log_q.put(f"\nDONE — case {result.case_id}")
            self._log_q.put(f"  {result.file_count} files, {result.total_bytes:,} bytes")
            self._log_q.put(f"  fingerprint {result.public_key_fingerprint}")
            for w in result.warnings:
                self._log_q.put(f"  warning: {w}")
        except Exception as e:  # noqa: BLE001
            self._log_q.put(f"ERROR: {e}")
        finally:
            self._log_q.put("__done__")

    def _drain_log(self) -> None:
        try:
            while True:
                msg = self._log_q.get_nowait()
                if msg == "__done__":
                    self.start_btn.config(state="normal")
                else:
                    self._log(msg)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log)

    def _log(self, msg: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")


def main() -> int:
    root = tk.Tk()
    CollectorGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
