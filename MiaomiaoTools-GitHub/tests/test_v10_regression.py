from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QApplication

from blackstream.slot_recognizer import SlotRecognizer
from blackstream.window import StandaloneWindow


def test_mobile_badged_encounter_and_ideal_source_glow(project_root: Path):
    recognizer = SlotRecognizer(project_root)
    floor_two = recognizer.recognize_path(
        project_root / "data" / "screen" / "测试2.jpg", 2
    )
    assert floor_two.slots[(0, 1)].node_type == "不期而遇"
    assert floor_two.slots[(1, 1)].node_type == "不期而遇"

    floor_three = recognizer.recognize_path(
        project_root / "data" / "screen" / "测试3.jpg", 3
    )
    assert floor_three.reference_template_id == "floor_3_template_09"
    assert floor_three.slots[(0, 0)].node_type is None
    assert floor_three.slots[(2, 3)].node_type is None


def test_mobile_flow_resident_overlay_wins_over_icon_match(project_root: Path):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data" / "screen" / "测试4.jpg", 4
    )
    flow_cells = {cell for cell, slot in state.slots.items() if slot.flow_resident}
    assert flow_cells == {(0, 2), (3, 2), (6, 2)}
    assert all(state.slots[cell].node_type == "流窜“居民”" for cell in flow_cells)


def test_manual_templates_apply_fixed_node_rules(project_root: Path):
    app = QApplication.instance() or QApplication([])
    window = StandaloneWindow(project_root)
    try:
        window._manual_template_clicked(window.template_list.item(0))
        assert window.manual_state.slots[(3, 1)].node_type == "诡意行商"
        assert all(
            slot.node_type == "作战"
            for slot in window.manual_state.slots.values()
            if slot.present and slot.distance == 1
        )

        window._manual_template_clicked(window.template_list.item(2))
        assert window.manual_state.slots[(1, 0)].node_type == "诡意行商"
        assert window.manual_state.slots[(4, 1)].node_type == "诡意行商"

        window.manual_floor_combo.setCurrentIndex(1)
        window._manual_template_clicked(window.template_list.item(0))
        assert all(
            slot.node_type == "作战"
            for slot in window.manual_state.slots.values()
            if slot.present and slot.distance == 1
        )

        window.manual_floor_combo.setCurrentIndex(4)
        window._manual_template_clicked(window.template_list.item(6))
        assert (window.manual_state.columns, window.manual_state.rows) == (9, 5)
        assert window.template_list.iconSize().width() == 216
    finally:
        window.close()
        app.processEvents()
