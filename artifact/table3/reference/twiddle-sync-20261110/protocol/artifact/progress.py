"""Best-effort progress snapshots and a compact, dependency-free terminal display.

Measurements publish one atomic JSON snapshot; the controller owns terminal output.
Telemetry failures must never turn a successful measurement into a failed run.
"""
from contextlib import suppress
import json
import math
import numbers
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
import unicodedata


PROGRESS_FILE_ENV = "THUNDERFFT_PROGRESS_FILE"
MAX_SNAPSHOT_BYTES = 65536
_ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]|\x1b[@-_]")
_TERMINAL = {"completed", "failed", "interrupted", "cancelled", "skipped", "partial", "not_started"}
_ACTIVE = {"running", "measuring", "active"}


def _clean(value, maximum=256):
    if not isinstance(value, str):
        return ""
    value = _ANSI.sub("", value)
    value = "".join(" " if char in "\r\n\t" else char for char in value
                    if char in "\r\n\t" or unicodedata.category(char) not in {"Cc", "Cf", "Cs"})
    return " ".join(value.split())[:maximum]


def _counter(value):
    return (isinstance(value, numbers.Integral) and not isinstance(value, bool)
            and 0 <= value <= 2 ** 63 - 1)


def validate_snapshot(value):
    """Return a bounded, sanitized task or runner snapshot, or ``None``."""
    if not isinstance(value, dict) or type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        return None
    aggregate = "completed" in value
    count_key = "completed" if aggregate else "done"
    count, total = value.get(count_key), value.get("total")
    if not _counter(count) or not _counter(total) or count > total:
        return None
    updated = value.get("updated_at")
    if not isinstance(updated, numbers.Real) or isinstance(updated, bool):
        return None
    try:
        updated = float(updated)
    except (ValueError, OverflowError):
        return None
    if not math.isfinite(updated) or updated < 0:
        return None
    result = {"schema_version": 1, count_key: int(count), "total": int(total),
              "status": _clean(value.get("status", "running"), 48), "updated_at": float(updated)}
    if aggregate:
        detail = value.get("detail")
        if detail is not None:
            if not isinstance(detail, dict) or "completed" in detail:
                return None
            detail = validate_snapshot(detail)
            if detail is None:
                return None
        result.update(evaluation=_clean(value.get("evaluation", ""), 80), detail=detail)
    else:
        result.update(unit=_clean(value.get("unit", "cases"), 48),
                      detail=_clean(value.get("detail", ""), 512))
    return result


def write_snapshot(path, value):
    """Atomically replace a progress file; return False on invalid data or IO failure."""
    temporary = None
    try:
        snapshot = validate_snapshot(value)
        if snapshot is None:
            return False
        data = (json.dumps(snapshot, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
        if len(data) > MAX_SNAPSHOT_BYTES:
            return False
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        os.replace(temporary, path)
        return True
    except (OSError, ValueError, TypeError, OverflowError):
        return False
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink()


def report(done, total, *, unit="cases", detail="", status="measuring"):
    """Publish task progress when the runner supplies a destination in its environment."""
    path = os.environ.get(PROGRESS_FILE_ENV)
    if path:
        write_snapshot(path, {"schema_version": 1, "done": done, "total": total,
                              "unit": unit, "detail": detail, "status": status,
                              "updated_at": time.time()})


def read_snapshot(path):
    """Read at most one bounded JSON snapshot, ignoring incomplete or invalid files."""
    try:
        with Path(path).open("rb") as handle:
            data = handle.read(MAX_SNAPSHOT_BYTES + 1)
        if len(data) > MAX_SNAPSHOT_BYTES:
            return None
        return validate_snapshot(json.loads(data))
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return None


def _cell_width(char):
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _clip(value, width):
    output, used = [], 0
    for char in value:
        cells = _cell_width(char)
        if used + cells > width:
            break
        output.append(char)
        used += cells
    return "".join(output)


class Dashboard:
    """One redrawable row per GPU; redirected output contains only transitions.

    ``update`` changes state, and ``refresh`` renders at most five times a second.
    Call ``close`` in a finally block, or use the display as a context manager.
    """

    def __init__(self, gpus=("A100", "H100", "B200"), *, mode="auto", stream=None, log_path=None):
        if mode not in {"auto", "tty", "plain", "off"}:
            raise ValueError("progress mode must be auto, tty, plain, or off")
        self.stream = stream if stream is not None else sys.stderr
        if mode == "auto":
            try:
                mode = "tty" if self.stream.isatty() and os.environ.get("TERM") != "dumb" else "plain"
            except (AttributeError, OSError):
                mode = "plain"
        self.mode, self.log_path = mode, Path(log_path) if log_path is not None else None
        self.gpus = tuple(dict.fromkeys(_clean(str(gpu), 32) for gpu in gpus))
        self.started = time.monotonic()
        self.rows = {gpu: {"status": "waiting", "progress": None} for gpu in self.gpus}
        self._last_refresh = float("-inf")
        self._plain_states = {}
        self._drawn = 0
        self._cursor_hidden = False
        self._closed = False
        self._failed = False
        self._lock = threading.RLock()

    def update(self, gpu, patch):
        if gpu not in self.rows or not isinstance(patch, dict):
            return
        with self._lock:
            row = self.rows[gpu]
            if "status" in patch:
                row["status"] = _clean(patch["status"], 48)
            if "progress" in patch:
                progress = patch["progress"]
                valid = validate_snapshot(progress) if progress is not None else None
                if progress is None or valid is not None:
                    row["progress"] = valid
            for field in ("detail", "message"):
                if field in patch:
                    row[field] = _clean(patch[field], 256)
            elapsed = patch.get("elapsed_seconds")
            if isinstance(elapsed, numbers.Real) and math.isfinite(elapsed) and elapsed >= 0:
                row["elapsed_seconds"] = float(elapsed)
            if row["status"] in _TERMINAL and "finished_elapsed" not in row:
                row["finished_elapsed"] = time.monotonic() - self.started

    @staticmethod
    def _status(row):
        status = row.get("status", "waiting")
        progress = row.get("progress") or {}
        if status in _ACTIVE:
            status = progress.get("status") or status
            detail = progress.get("detail")
            if status in _ACTIVE and isinstance(detail, dict):
                status = detail.get("status") or status
        return "finishing" if status == "completed" and row.get("status") in _ACTIVE else status

    def _line(self, gpu, width):
        row = self.rows[gpu]
        progress = row.get("progress") or {}
        inner = progress.get("detail") if "completed" in progress else progress
        inner = inner if isinstance(inner, dict) else {}
        status = self._status(row)
        completed, total = progress.get("completed", 0), progress.get("total", 0)
        evaluation = progress.get("evaluation", "")
        evaluation = re.sub(r"^(table|figure)", lambda match: match[0].title(), evaluation)
        inner_text = (f'{inner["done"]}/{inner["total"]} {inner.get("unit") or "cases"}'
                      if "done" in inner else "")
        task = " ".join(value for value in (evaluation, inner_text) if value)
        # Stage audits may measure only GPU execution, excluding acquisition and
        # setup. Keep the live clock on one wall-time basis so it cannot jump
        # backwards when the final audit duration arrives.
        elapsed = row.get("finished_elapsed", time.monotonic() - self.started)
        seconds = int(elapsed)
        clock = f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60:02d}:{seconds % 60:02d}"
        if width < 45:
            # Keep the measured fraction visible even in a narrow terminal.
            fields = [gpu, status, inner_text or (f"{completed}/{total}" if total else "")]
            return _clip(" ".join(field for field in fields if field), width)
        fraction = completed / total if total and "completed" in progress else 0
        if total and inner.get("total") and completed < total:
            fraction = (completed + inner["done"] / inner["total"]) / total
        fraction = 1.0 if row["status"] == "completed" else min(fraction, .99)
        filled = min(10, int(10 * fraction))
        bar = "[" + "#" * filled + "." * (10 - filled) + "]"
        steps = f"{completed}/{total} steps" if total and "completed" in progress else ""
        fields = [f"{gpu:<4} {status}", steps, task, clock]
        # Fractions are more useful than the decorative bar when space is limited.
        if width >= 90:
            fields.insert(1, bar)
        line = " | ".join(field for field in fields if field)
        description = inner.get("detail") or row.get("detail") or row.get("message")
        if description:
            line += " | " + description
        return _clip(line, width)

    def _width(self):
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        return max(1, columns - 1)

    def _write(self, text):
        try:
            self.stream.write(text)
            self.stream.flush()
            return True
        except (OSError, ValueError):
            self._failed = True
            self._restore_cursor()
            return False

    def _restore_cursor(self):
        if self._cursor_hidden:
            with suppress(OSError, ValueError):
                self.stream.write("\x1b[?25h")
                self.stream.flush()
            self._cursor_hidden = False

    def refresh(self, *, force=False):
        with self._lock:
            if self._closed or self._failed or self.mode == "off":
                return
            now = time.monotonic()
            if not force and now - self._last_refresh < .2:
                return
            self._last_refresh = now
            if self.mode == "plain":
                for gpu in self.gpus:
                    row = self.rows[gpu]
                    progress = row.get("progress") or {}
                    state = (row["status"], self._status(row), progress.get("evaluation", ""))
                    if self._plain_states.get(gpu) != state:
                        self._write(self._line(gpu, 200) + "\n")
                        self._plain_states[gpu] = state
                return
            prefix = ""
            if not self._cursor_hidden:
                prefix = "\x1b[?25l"
                self._cursor_hidden = True
            if self._drawn:
                prefix += "\r" + (f"\x1b[{self._drawn - 1}A" if self._drawn > 1 else "")
            lines = ["\r\x1b[2K" + self._line(gpu, self._width()) for gpu in self.gpus]
            if lines and self._write(prefix + "\n".join(lines)):
                self._drawn = len(lines)

    def log(self, message):
        """Keep verbose controller messages out of the live GPU panel."""
        message = _clean(str(message), 8192)
        with self._lock:
            if self.log_path is not None:
                try:
                    self.log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.log_path.open("a", encoding="utf-8") as handle:
                        handle.write(message + "\n")
                except OSError:
                    pass
            elif self.mode == "plain" and not self._closed:
                self._write(message + "\n")

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                self.refresh(force=True)
                if self._drawn and not self._failed:
                    self._write("\n")
            finally:
                self._restore_cursor()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exception):
        self.close()
