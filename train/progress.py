"""Tiny JSONL progress logger shared by the trainers.

Training jobs run as separate processes and append JSON lines to a log file; the
web server tails that file to stream live updates to the browser via SSE.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional


class ProgressLogger:
    def __init__(self, path: Optional[str] = None, echo: bool = True, append: bool = False):
        self.path = path
        self.echo = echo
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            # Truncate at start of a job (unless resuming a crashed one).
            open(path, "a" if append else "w").close()

    def log(self, payload: dict) -> None:
        payload = {"t": time.time(), **payload}
        line = json.dumps(payload)
        if self.echo:
            print(line, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(line + "\n")

    def __call__(self, payload: dict) -> None:
        self.log(payload)
