from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from blackstream.window import StandaloneWindow


def test_manual_palette_exports_drag_text_and_has_resizable_splitter(project_root: Path):
    app = QApplication.instance() or QApplication([])
    window = StandaloneWindow(project_root)
    try:
        assert window.manual_splitter.count() == 2
        assert window.manual_palette.minimumHeight() >= 190
        item = next(
            window.manual_palette.item(index)
            for index in range(window.manual_palette.count())
            if window.manual_palette.item(index).data(Qt.ItemDataRole.UserRole)
        )
        assert window.manual_palette.mimeData([item]).hasText()
    finally:
        window.close()
        app.processEvents()


def test_context_candidate_action_applies_existing_candidate(project_root: Path):
    app = QApplication.instance() or QApplication([])
    window = StandaloneWindow(project_root)
    try:
        cell = (1, 1)
        slot = window.state.slots[cell]
        slot.present = True
        slot.node_type = "未知的凶戾"
        slot.candidates = ["作战", "紧急作战"]
        window._apply_context_candidate(cell, "紧急作战")
        assert window.state.slots[cell].node_type == "紧急作战"
    finally:
        window.close()
        app.processEvents()
