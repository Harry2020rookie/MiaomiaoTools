from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QListWidgetItem

from blackstream.map_model import FloorMapState
from blackstream.slot_recognizer import SlotRecognizer
from blackstream.window import StandaloneWindow
from models.grid_geometry import GRID_SHAPES
from models.map_template import MapTemplate
from rules.rule_loader import load_rules


def test_floor_one_starts_as_fifteen_unconnected_slots():
    state = FloorMapState.empty(1, *GRID_SHAPES[1])
    assert (state.columns, state.rows) == (5, 3)
    assert len(state.slots) == 15
    assert all(not slot.present for slot in state.slots.values())
    assert all(not edge.present for edge in state.edges.values())


@pytest.mark.parametrize(
    ("filename", "template_id", "node_count", "edge_count", "start", "end_count", "traders"),
    [
        ("scenario1.png", "floor_1_template_02", 11, 11, (1, 1), 1, {(3, 1)}),
        ("scenario2.png", "floor_1_template_03", 13, 12, (2, 1), 2, {(1, 0), (4, 1)}),
        ("scenario3.png", "floor_1_template_01", 12, 13, (0, 1), 2, {(3, 1)}),
    ],
)
def test_examples_restore_topology_and_fixed_traders(
    project_root: Path,
    filename: str,
    template_id: str,
    node_count: int,
    edge_count: int,
    start: tuple[int, int],
    end_count: int,
    traders: set[tuple[int, int]],
):
    state = SlotRecognizer(project_root).recognize_path(project_root / "data" / filename, 1)
    assert state.reference_template_id == template_id
    assert sum(slot.present for slot in state.slots.values()) == node_count
    assert sum(edge.present for edge in state.edges.values()) == edge_count
    assert state.start_cell == start
    assert sum(slot.fixed_end for slot in state.slots.values()) == end_count
    for cell, slot in state.slots.items():
        if cell in traders and slot.node_type == "未知的诡秘":
            assert slot.candidates == ["诡意行商"]
            assert slot.inferred_type == "诡意行商"
        elif slot.node_type == "未知的诡秘":
            assert "诡意行商" not in slot.candidates


@pytest.fixture()
def window(project_root: Path):
    app = QApplication.instance() or QApplication([])
    result = StandaloneWindow(project_root)
    yield result
    result.close()
    app.processEvents()


def test_candidate_click_applies_type_and_undo_redo(window: StandaloneWindow):
    cell = (1, 1)
    slot = window.state.slots[cell]
    slot.present = True
    slot.node_type = "未知的凶戾"
    slot.candidates = ["作战", "紧急作战"]
    window._select_node(cell)

    window._candidate_clicked(QListWidgetItem("紧急作战"))
    assert slot.node_type == "紧急作战"
    assert slot.inferred_type is None

    window._undo()
    assert window.state.slots[cell].node_type == "未知的凶戾"
    window._redo()
    assert window.state.slots[cell].node_type == "紧急作战"


def test_delete_and_move_keep_start(window: StandaloneWindow):
    start = (0, 1)
    target = (1, 1)
    window.state.start_cell = start
    window.state.current_cell = start
    window.state.slots[start].present = True
    window.state.slots[start].node_type = "起点"
    window.state.slots[target].present = True
    window.state.slots[target].node_type = "不期而遇"

    window.delete_button.setChecked(True)
    window._node_double_clicked(start)
    assert window.state.slots[start].present

    window.move_button.setChecked(True)
    window._node_double_clicked(target)
    assert window.state.current_cell == target
    assert window.state.slots[target].visited
    assert window.state.slots[start].present

    window.delete_button.setChecked(True)
    window._node_double_clicked(target)
    assert not window.state.slots[target].present
    window._undo()
    assert window.state.slots[target].present


@pytest.mark.parametrize(
    ("floor", "filename", "template_id", "start", "nodes", "edges", "ends"),
    [
        (2, "二层.png", "floor_2_template_08", (4, 0), 19, 20, 2),
        (3, "三层.png", "floor_3_template_08", (5, 4), 29, 32, 0),
    ],
)
def test_real_floor_two_and_three_screenshots(
    project_root: Path,
    floor: int,
    filename: str,
    template_id: str,
    start: tuple[int, int],
    nodes: int,
    edges: int,
    ends: int,
):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data/screen" / filename, floor
    )
    assert state.reference_template_id == template_id
    assert state.start_cell == start
    assert state.current_cell == start
    assert sum(slot.present for slot in state.slots.values()) == nodes
    assert sum(edge.present for edge in state.edges.values()) == edges
    assert sum(slot.fixed_end for slot in state.slots.values()) == ends
    assert all(
        slot.candidates
        for slot in state.slots.values()
        if slot.node_type in {"未知的诡秘", "未知的凶戾"}
    )


@pytest.mark.parametrize(
    ("floor", "folder", "mapping"),
    [
        (
            2,
            "二层",
            (4, 5, 6, 7, 8, 9, 10, 1, 2, 3),
        ),
        (
            3,
            "三层",
            (8, 9, 10, 1, 2, 3, 4, 5, 6, 7),
        ),
    ],
)
def test_website_reference_images_map_to_json_templates(
    project_root: Path,
    floor: int,
    folder: str,
    mapping: tuple[int, ...],
):
    recognizer = SlotRecognizer(project_root)
    for image_number, template_number in enumerate(mapping, start=1):
        state = recognizer.recognize_path(
            project_root / "data/template" / folder / f"{image_number}.png", floor
        )
        assert state.reference_template_id == (
            f"floor_{floor}_template_{template_number:02d}"
        )
        assert state.start_cell is not None


def test_manual_annotation_types_are_grouped(window: StandaloneWindow):
    rows = [
        (window.type_combo.itemText(index), window.type_combo.itemData(index))
        for index in range(window.type_combo.count())
    ]
    headings = [text for text, data in rows if data is None]
    assert headings == ["── 固定可见节点 ──", "── 诡秘类节点 ──", "── 凶戾类节点 ──"]
    assert [data for _text, data in rows if data == "紧急作战"] == ["紧急作战"]


def test_floor_selector_opens_one_to_five(window: StandaloneWindow):
    for floor in range(1, 6):
        window.floor_combo.setCurrentIndex(floor - 1)
        assert window.state.floor == floor
        assert (window.state.columns, window.state.rows) == GRID_SHAPES[floor]
        assert window.floor_combo.model().item(floor - 1).isEnabled()


def test_distance_badges_are_optional_and_disabled_by_default(
    window: StandaloneWindow,
):
    assert not window.distance_checkbox.isChecked()
    assert not window.canvas.show_distances
    window.distance_checkbox.setChecked(True)
    assert window.canvas.show_distances
    window.distance_checkbox.setChecked(False)
    assert not window.canvas.show_distances


@pytest.mark.parametrize("floor", (4, 5))
def test_floor_four_and_five_reference_images_match_all_templates(
    project_root: Path, floor: int
):
    recognizer = SlotRecognizer(project_root)
    folder = project_root / "data/template" / f"floor_{floor}"
    for template_number in range(1, 11):
        state = recognizer.recognize_path(
            folder / f"v{template_number}.jpg", floor
        )
        assert state.reference_template_id == (
            f"floor_{floor}_template_{template_number:02d}"
        )
        assert state.start_cell is not None


def test_floor_four_reference_fullscreen(project_root: Path):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data/screen/floor_4_reference.png", 4
    )
    assert state.reference_template_id == "floor_4_template_05"
    assert state.start_cell == (4, 2)
    assert state.current_cell == (4, 2)
    assert sum(slot.present for slot in state.slots.values()) == 34
    assert sum(edge.present for edge in state.edges.values()) == 34
    assert all(
        slot.candidates
        for slot in state.slots.values()
        if slot.node_type in {"未知的诡秘", "未知的凶戾"}
    )


def test_floor_five_special_base_uses_nine_columns(project_root: Path):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data/template/floor_5/v7.jpg", 5
    )
    assert state.reference_template_id == "floor_5_template_07"
    assert (state.columns, state.rows) == (9, 5)
    assert len(state.slots) == 45


def test_floor_four_and_five_templates_validate_and_report_review_state(
    project_root: Path,
):
    templates = [
        MapTemplate.load(path)
        for floor in (4, 5)
        for path in sorted(
            (project_root / "data/templates").glob(
                f"floor_{floor}_template_*.json"
            )
        )
    ]
    assert len(templates) == 20
    assert any(template.metadata.get("requires_manual_review") for template in templates)
    assert any(not template.metadata.get("requires_manual_review") for template in templates)


def test_floor_one_four_visible_combats_do_not_force_emergency(project_root: Path):
    state = FloorMapState.empty(1, *GRID_SHAPES[1])
    state.start_cell = (0, 0)
    state.slots[(0, 0)].present = True
    state.slots[(0, 0)].node_type = "起点"
    for cell in ((1, 0), (2, 0), (0, 1), (0, 2)):
        state.slots[cell].present = True
        state.slots[cell].node_type = "作战"
    target = (3, 0)
    state.slots[target].present = True
    state.slots[target].node_type = "未知的凶戾"
    for first, second in (
        ((0, 0), (1, 0)),
        ((1, 0), (2, 0)),
        ((2, 0), (3, 0)),
    ):
        state.edges[tuple(sorted((first, second)))].present = True
    state.recompute(load_rules(project_root / "data/rules"))
    assert state.slots[target].candidates == ["作战", "紧急作战"]
    assert state.slots[target].inferred_type is None


def test_floor_two_ideal_source_forces_emergency(project_root: Path):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data/screen/v4-regression/floor2.png", 2
    )
    slot = state.slots[(0, 1)]
    assert slot.node_type == "未知的凶戾"
    assert slot.ideal_source
    assert slot.candidates == ["紧急作战"]
    assert slot.inferred_type == "紧急作战"
    assert slot.label == "紧急作战 · 理想源"


def test_floor_three_live_hud_does_not_shift_grid(project_root: Path):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data/screen/v4-regression/floor3.png", 3
    )
    assert state.reference_template_id == "floor_3_template_06"
    assert state.start_cell == (4, 1)
    assert state.current_cell == (4, 1)
    assert sum(slot.present for slot in state.slots.values()) == 29
    assert sum(edge.present for edge in state.edges.values()) == 32
    y_levels = sorted({round(slot.screen_center[1]) for slot in state.slots.values()})
    assert y_levels == [320, 539, 758, 977, 1196]


def test_floor_three_tall_danger_enemy_variant_is_recognized(project_root: Path):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data/screen/v7-regression/floor3-danger-enemy.png", 3
    )
    slot = state.slots[(0, 4)]
    assert slot.present
    assert slot.node_type == "险路恶敌"
    assert slot.confidence >= 0.95


def test_floor_four_flow_residents_reverse_infer_hidden_base(project_root: Path):
    recognizer = SlotRecognizer(project_root)
    state = recognizer.recognize_path(
        project_root / "data/screen/v5-regression/floor4-residents.png", 4
    )
    flow_cells = {
        cell for cell, slot in state.slots.items() if slot.flow_resident
    }
    assert flow_cells == {(1, 0), (1, 2)}
    assert state.slots[(4, 3)].node_type == "诡意行商"
    assert not state.slots[(4, 3)].flow_resident
    assert state.slots[(0, 4)].node_type == "“居民”据点"
    assert not any(slot.settlement_candidate for slot in state.slots.values())

    state.slots[(0, 4)].node_type = "未知的凶戾"
    state.slots[(0, 4)].confidence = 0.90
    state.recompute(recognizer.rules)
    candidates = {
        cell: slot.settlement_distance
        for cell, slot in state.slots.items()
        if slot.settlement_candidate
    }
    assert candidates == {(3, 0): 4, (0, 4): 5}
    assert all(
        state.slots[cell].node_type == "未知的凶戾" for cell in candidates
    )
    assert not any(
        slot.settlement_candidate and slot.node_type is None
        for slot in state.slots.values()
    )

    candidate = state.slots[(3, 0)]
    assert candidate.label == "未知的凶戾 · 据点可能位置"


def test_only_one_resident_settlement_can_exist(project_root: Path):
    recognizer = SlotRecognizer(project_root)
    state = recognizer.recognize_path(
        project_root / "data/screen/v5-regression/floor4-residents.png", 4
    )
    duplicate = state.slots[(3, 0)]
    duplicate.present = True
    duplicate.node_type = "“居民”据点"
    duplicate.confidence = 0.50
    state.recompute(recognizer.rules)

    settlements = [
        cell
        for cell, slot in state.slots.items()
        if slot.node_type == "“居民”据点"
    ]
    assert settlements == [(0, 4)]
    assert duplicate.node_type == "未知的凶戾"


def test_floor_five_hud_mask_and_trader_candidate_regression(project_root: Path):
    recognizer = SlotRecognizer(project_root)
    hidden = recognizer.recognize_path(
        project_root / "data/screen/v6-regression/floor5-a.png", 5
    )
    assert {
        cell for cell, slot in hidden.slots.items() if slot.flow_resident
    } == {(4, 0), (2, 2), (2, 3)}
    assert not hidden.slots[(9, 0)].flow_resident
    assert hidden.slots[(5, 4)].node_type == "险路恶敌"
    assert hidden.slots[(6, 4)].distance == 10
    assert "诡意行商" in hidden.slots[(6, 4)].candidates

    revealed = recognizer.recognize_path(
        project_root / "data/screen/v6-regression/floor5-b.png", 5
    )
    assert revealed.slots[(6, 4)].node_type == "诡意行商"


def test_annotation_palette_is_grouped_and_uses_icons(window: StandaloneWindow):
    entries = [
        (
            window.annotation_palette.item(index).text(),
            window.annotation_palette.item(index).data(Qt.ItemDataRole.UserRole),
            window.annotation_palette.item(index).icon().isNull(),
        )
        for index in range(window.annotation_palette.count())
    ]
    headings = [text for text, value, _is_null in entries if value is None]
    assert headings == ["固定可见节点", "诡秘类节点", "凶戾类节点"]
    icon_state = {value: is_null for _text, value, is_null in entries if value}
    assert not icon_state["起点"]
    assert not icon_state["“居民”据点"]
    assert not icon_state["流窜“居民”"]


def test_annotation_palette_applies_selected_type(window: StandaloneWindow):
    cell = (1, 1)
    window.state.slots[cell].present = True
    window._select_node(cell)
    item = next(
        window.annotation_palette.item(index)
        for index in range(window.annotation_palette.count())
        if window.annotation_palette.item(index).data(Qt.ItemDataRole.UserRole)
        == "流窜“居民”"
    )
    window._annotation_palette_clicked(item)
    assert window.state.slots[cell].node_type == "流窜“居民”"
    assert window.state.slots[cell].flow_resident


def test_candidate_filter_previews_every_matching_node(window: StandaloneWindow):
    first, second, other = (1, 0), (2, 0), (3, 0)
    window.state.slots[first].candidates = ["不期而遇", "误入奇境"]
    window.state.slots[second].candidates = ["秘境行商", "误入奇境"]
    window.state.slots[other].candidates = ["秘境行商"]
    window.canvas.candidate_phase = 0

    index = window.filter_combo.findData("误入奇境")
    window.filter_combo.setCurrentIndex(index)
    assert window.canvas.candidate_filter == "误入奇境"
    assert window.canvas.display_type_for_slot(window.state.slots[first]) == (
        "误入奇境",
        True,
    )
    assert window.canvas.display_type_for_slot(window.state.slots[second]) == (
        "误入奇境",
        True,
    )
    assert not window.canvas.display_type_for_slot(window.state.slots[other])[1]

    window.filter_combo.setCurrentIndex(0)
    assert window.canvas.candidate_filter is None
    assert window.canvas.display_type_for_slot(window.state.slots[first]) == (
        "不期而遇",
        False,
    )


def test_v7_theme_background_and_split_palettes(window: StandaloneWindow, project_root: Path):
    assert window.theme == "light"
    assert window.annotation_palette.isHidden()
    window.annotation_toggle.setChecked(True)
    assert not window.annotation_palette.isHidden()

    filter_values = {
        window.filter_palette.item(index).data(Qt.ItemDataRole.UserRole)
        for index in range(window.filter_palette.count())
    }
    assert {"__all__", "误入奇境", "紧急作战", "“居民”据点"} <= filter_values

    assert window.canvas.set_background_image(str(project_root / "data/scenario1.png"))
    assert not window.canvas.background_pixmap.isNull()
    window.theme_button.setChecked(True)
    assert window.theme == "dark"
    assert window.canvas.theme == "dark"


def test_settlement_filter_keeps_unknown_ferocity_appearance(window: StandaloneWindow):
    slot = window.state.slots[(1, 1)]
    slot.present = True
    slot.node_type = "未知的凶戾"
    slot.settlement_candidate = True
    index = window.filter_combo.findData("“居民”据点")
    window.filter_combo.setCurrentIndex(index)

    assert window.canvas.display_type_for_slot(slot) == ("未知的凶戾", True)
