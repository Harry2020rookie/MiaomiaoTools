from __future__ import annotations

import ctypes
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from blackstream.window import StandaloneWindow


PROJECT_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def main() -> int:
    if sys.platform == "win32" and hasattr(ctypes, "windll"):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ega.miaomiao.tools"
        )
    app = QApplication(sys.argv)
    app.setApplicationName("妙妙工具")
    app.setStyle("Fusion")
    window = StandaloneWindow(PROJECT_ROOT)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
