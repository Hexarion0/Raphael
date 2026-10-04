"""Terminal captions driven by playback progress, without a separate typing timer."""

from __future__ import annotations

import logging
import re
import shutil
import sys
import threading
import unicodedata
from collections.abc import Callable
from typing import TextIO

from raphael.logging import SecretMaskingFilter

_TERMINAL_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\)|[@-_])"
)


def _plain_text(text: str) -> str:
    """Keep caption text on one line and remove terminal control sequences."""
    text = SecretMaskingFilter._redact(_TERMINAL_ESCAPE.sub("", text))
    return " ".join(
        "".join(char for char in text if char.isspace() or ord(char) >= 32 and ord(char) != 127)
        .split()
    )


def _cell_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _fit_line(text: str, columns: int) -> str:
    """Show the newest caption characters without wrapping the terminal line."""
    available = max(0, columns - 1)
    if sum(_cell_width(char) for char in text) <= available:
        return text
    if available < 2:
        return "" if available == 0 else "…"
    suffix: list[str] = []
    used = 1  # Ellipsis.
    for char in reversed(text):
        width = _cell_width(char)
        if used + width > available:
            break
        suffix.append(char)
        used += width
    tail = "".join(reversed(suffix))
    while tail and unicodedata.combining(tail[0]):
        tail = tail[1:]
    return "…" + tail


class TerminalCaptions:
    """Display cumulative visible prefixes supplied by the audio player.

    ``start(full_text)`` opens a sentence after playback starts. Each ``update``
    supplies its visible prefix; the renderer never advances on its own. A final
    or interrupted update saves that prefix as a completed terminal line. The
    next sentence starts another line. ``close`` saves a currently visible
    partial sentence and resets the renderer, which can be started again.

    TTY progress rewrites one line. Redirected output receives only final or
    interrupted snapshots, with no terminal escapes or intermediate prefixes.
    All output, including caption-aware logs, shares one lock.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        prefix: str = "RAPHAEL: ",
        width: int | None = None,
    ) -> None:
        self.stream = sys.stdout if stream is None else stream
        self.prefix = _plain_text(prefix).rstrip() + " "
        self.width = width
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._lock = threading.RLock()
        self._full_text = ""
        self._visible_text = ""
        self._active = False
        self._drawn = False
        self._last_line: str | None = None

    def start(self, full_text: str) -> None:
        """Start a sentence, preserving a previously unfinished visible prefix."""
        with self._lock:
            if self._active:
                self._finish_locked(interrupted=True)
            self._full_text = _plain_text(full_text)
            self._visible_text = ""
            self._active = bool(self._full_text)
            if self._active and self.is_tty:
                self._draw_locked()

    def update(
        self,
        visible_text: str,
        *,
        finished: bool = False,
        interrupted: bool = False,
    ) -> None:
        """Show a valid prefix; ignore mismatches and never rewind visible words."""
        with self._lock:
            if not self._active:
                return
            visible = _plain_text(visible_text)
            if not self._full_text.startswith(visible):
                if interrupted:
                    # A partially revealed secret may not match its final mask.
                    # Save the last safe prefix without leaving a stale caption.
                    self._finish_locked(interrupted=True)
                return
            if len(visible) >= len(self._visible_text):
                self._visible_text = visible
            if finished or interrupted:
                self._finish_locked(interrupted=interrupted)
            elif self.is_tty:
                self._draw_locked()

    def close(self) -> None:
        """Save any visible unfinished speech and leave a clean terminal line."""
        with self._lock:
            if self._active:
                self._finish_locked(interrupted=True)

    def write_log(self, message: str) -> None:
        """Write a formatted log between caption redraws under the same lock."""
        with self._lock:
            if self.is_tty and self._drawn:
                self._clear_locked()
            self.stream.write(message.rstrip("\r\n") + "\n")
            self.stream.flush()
            if self.is_tty and self._active:
                self._draw_locked()

    def _draw_locked(self, *, interrupted: bool = False) -> None:
        line = self.prefix + self._visible_text
        if interrupted:
            line += " [interrupted]"
        columns = self.width if self.width is not None else shutil.get_terminal_size().columns
        line = _fit_line(line, columns)
        if self._last_line == line and self._drawn:
            return
        self.stream.write("\r\x1b[2K" + line)
        self.stream.flush()
        self._last_line = line
        self._drawn = True

    def _clear_locked(self) -> None:
        self.stream.write("\r\x1b[2K")
        self._drawn = False
        self._last_line = None

    def _finish_locked(self, *, interrupted: bool) -> None:
        if self._visible_text:
            if self.is_tty:
                self._draw_locked(interrupted=interrupted)
                self.stream.write("\n")
            else:
                marker = " [interrupted]" if interrupted else ""
                self.stream.write(self.prefix + self._visible_text + marker + "\n")
        elif self.is_tty and self._drawn:
            self._clear_locked()
        self.stream.flush()
        self._full_text = self._visible_text = ""
        self._active = self._drawn = False
        self._last_line = None


class CaptionLoggingHandler(logging.StreamHandler):
    """Preserve formatted logging while sharing the caption renderer's lock."""

    def __init__(self, captions: TerminalCaptions) -> None:
        super().__init__(captions.stream)
        self.captions = captions

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.captions.write_log(self.format(record))
        except Exception:
            self.handleError(record)


def _replace_handler(
    logger: logging.Logger, original: logging.Handler, replacement: logging.Handler,
) -> bool:
    """Keep handler order, including filters that redact before other handlers."""
    with logging._lock:
        if original not in logger.handlers:
            return False
        logger.handlers[logger.handlers.index(original)] = replacement
        return True


def install_caption_logging(
    captions: TerminalCaptions, logger: logging.Logger | None = None,
) -> Callable[[], None]:
    """Replace matching console handlers and return an idempotent restore hook.

    Existing formatter, level, and filters (including secret redaction) are
    copied. Handlers for other streams or files are left in place. Call
    ``captions.close()`` before the returned hook when leaving the voice loop.
    """
    target = logging.getLogger() if logger is None else logger
    replacements: list[tuple[logging.Handler, logging.Handler]] = []
    for original in list(target.handlers):
        if (
            not isinstance(original, logging.StreamHandler)
            or original.stream is not captions.stream
        ):
            continue
        if isinstance(original, CaptionLoggingHandler):
            continue
        replacement = CaptionLoggingHandler(captions)
        replacement.setLevel(original.level)
        replacement.setFormatter(original.formatter)
        for log_filter in original.filters:
            replacement.addFilter(log_filter)
        if _replace_handler(target, original, replacement):
            replacements.append((original, replacement))

    def restore() -> None:
        for original, replacement in replacements:
            if _replace_handler(target, replacement, original):
                replacement.close()

    return restore
