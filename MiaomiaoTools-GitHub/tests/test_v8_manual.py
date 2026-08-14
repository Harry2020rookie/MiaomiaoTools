from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from blackstream.window import StandaloneWindow


def test_manual_template_preview_counts_and_loading(project_root: Path):
    app = QApplication.instance() or QApplication([])
    window = StandaloneWindow(project_root)
    try:
        expected = {1: 3, 2: 10, 3: 10, 4: 10, 5: 10}
        for floor, count in expected.items():
            window.manual_floor_combo.setCurrentIndex(floor - 1)
            assert window.template_list.count() == count
            window._manual_template_clicked(window.template_list.item(0))
            assert window.manual_state.reference_template_id == f"floor_{floor}_template_01"
            assert window.manual_state.start_cell is not None
            assert sum(edge.present for edge in window.manual_state.edges.values()) > 0
    finally:
        window.close()
        app.processEvents()


def test_manual_annotation_and_blank_click_clear_selection(project_root: Path):
    app = QApplication.instance() or QApplication([])
    window = StandaloneWindow(project_root)
    try:
        window._manual_template_clicked(window.template_list.item(0))
        cell = next(iter(window.manual_state.slots))
        window._manual_type_dropped(cell, "作战")
        assert window.manual_state.slots[cell].node_type == "作战"
        window.manual_canvas.nodeSelected.emit(None)
        assert window.manual_selected_cell is None
        assert window.manual_node_title.text() == "未选择节点"
    finally:
        window.close()
        app.processEvents()
