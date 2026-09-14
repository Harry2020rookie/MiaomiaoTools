import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication
from blackstream.window import StandaloneWindow
from blackstream.map_model import FloorMapState


@pytest.fixture
def window(project_root):
    app = QApplication.instance() or QApplication([])
    widget = StandaloneWindow(project_root)
    yield widget
    widget.close()


def test_switch_preserves_map_and_undo(window):
    original = window.state
    original.slots[(1, 0)].present = True
    window._push_history()
    original.slots[(1, 0)].node_type = '作战'
    window.floor_combo.setCurrentIndex(1)
    window.floor_combo.setCurrentIndex(0)
    assert window.state is original
    assert window.state.slots[(1, 0)].node_type == '作战'
    window._undo()
    assert window.state.slots[(1, 0)].node_type is None


def test_auto_import_preserves_previous_floor_and_reset_is_local(window):
    first = window.state
    second = FloorMapState.empty(2, 8, 5)
    window._apply_recognition(second, 'second.png')
    window.floor_combo.setCurrentIndex(0)
    assert window.state is first
    window._reset_map()
    window.floor_combo.setCurrentIndex(1)
    assert window.state is second
    assert window.source_label.text() == 'second.png'


def test_manual_switch_preserves_state(window):
    first = window.manual_state
    first.slots[(1, 0)].present = True
    window.manual_floor_combo.setCurrentIndex(1)
    window.manual_floor_combo.setCurrentIndex(0)
    assert window.manual_state is first
    assert window.manual_state.slots[(1, 0)].present


def test_move_and_paint_current_marker(window):
    cell = (1, 0)
    window.state.slots[cell].present = True
    window.move_button.setChecked(True)
    window._node_double_clicked(cell)
    assert window.state.current_cell == cell
    image = QImage(160, 160, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        window.canvas._paint_current_marker(painter, QPointF(70, 70))
    finally:
        painter.end()
    window._undo()
    assert window.state.current_cell is None
