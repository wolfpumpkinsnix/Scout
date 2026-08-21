import logging
import shutil
import sys
from typing import TextIO

_RESET = "\033[0m"
_CLEAR_LINE = "\r\033[2K"
_COLORS = {
    "blue": "\033[94m",
    "cyan": "\033[96m",
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, fmt: str, color: bool = True) -> None:
        super().__init__(fmt)
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        if not self.color:
            return message
        if record.levelno >= logging.ERROR:
            color = _COLORS["red"]
        elif "stop" in message or "complete" in message or "finished" in message:
            color = _COLORS["green"]
        elif "start" in message or "load-model" in message or "read" in message:
            color = _COLORS["cyan"]
        elif "phase=prepare" in message or "phase=chunk" in message:
            color = _COLORS["yellow"]
        else:
            color = _COLORS["blue"]
        return f"{color}{message}{_RESET}"


class InteractiveConsoleHandler(logging.StreamHandler):
    def __init__(self, stream: TextIO = sys.stderr) -> None:
        super().__init__(stream)
        self.interactive = bool(getattr(stream, "isatty", lambda: False)())
        self._progress = ""

    def emit(self, record: logging.LogRecord) -> None:
        if self._progress:
            self.stream.write(_CLEAR_LINE)
        super().emit(record)
        if self._progress:
            self.stream.write(self._progress)
            self.flush()

    def update_progress(self, label: str, current: int, total: int) -> None:
        if not self.interactive or total < 1:
            return
        current = min(max(current, 0), total)
        width = max(10, min(40, shutil.get_terminal_size((80, 20)).columns - len(label) - 16))
        filled = width * current // total
        percent = current * 100 // total
        self._progress = (
            f"\r{_COLORS['cyan']}{label:<12}{_RESET} "
            f"[{_COLORS['green']}{'█' * filled}{_RESET}{'░' * (width - filled)}] "
            f"{percent:3d}% ({current}/{total})"
        )
        self.stream.write(_CLEAR_LINE + self._progress)
        if current == total:
            self.stream.write("\n")
            self._progress = ""
        self.flush()

    def clear_progress(self) -> None:
        if self._progress:
            self.stream.write(_CLEAR_LINE)
            self.flush()
            self._progress = ""


_CONSOLE: InteractiveConsoleHandler | None = None


def update_progress(label: str, current: int, total: int) -> None:
    if _CONSOLE is not None:
        _CONSOLE.update_progress(label, current, total)


def clear_progress() -> None:
    if _CONSOLE is not None:
        _CONSOLE.clear_progress()


def configure_colored_logging() -> None:
    global _CONSOLE
    _CONSOLE = InteractiveConsoleHandler()
    _CONSOLE.setFormatter(ColoredFormatter(
        "%(asctime)s %(levelname)s %(message)s", color=_CONSOLE.interactive))
    logging.basicConfig(level=logging.INFO, handlers=[_CONSOLE], force=True)
