from __future__ import annotations

from pathlib import Path

import pytest

from blackstream.slot_recognizer import SlotRecognizer


@pytest.mark.parametrize(
    ("floor", "template_id", "start_cell"),
    (
        (1, "floor_1_template_02", (1, 1)),
        (2, "floor_2_template_05", (2, 0)),
        (3, "floor_3_template_09", (6, 2)),
        (4, "floor_4_template_04", (4, 3)),
    ),
)
def test_mobile_fullscreen_maps_use_relative_map_region(
    project_root: Path,
    floor: int,
    template_id: str,
    start_cell: tuple[int, int],
):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data" / "screen" / f"测试{floor}.jpg",
        floor,
    )
    assert state.reference_template_id == template_id
    assert state.start_cell == start_cell
    assert state.map_region is not None
    left, top, right, bottom = state.map_region
    assert right > left and bottom > top


def test_mobile_gallery_bytes_use_same_pipeline(project_root: Path):
    image_path = project_root / "data" / "screen" / "测试2.jpg"
    state = SlotRecognizer(project_root).recognize_bytes(
        image_path.read_bytes(), 2, source_name="content://gallery/test2"
    )

    assert state.reference_template_id == "floor_2_template_05"
    assert state.source_path == "content://gallery/test2"
    assert state.map_region is not None
