from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blackstream.slot_recognizer import SlotRecognizer
from blackstream.window import StandaloneWindow
from vision.domain_detector import DomainDetector, DomainHeader
from vision.image_io import read_image


@pytest.mark.parametrize("name,idea,policy", [
    ("macos-auto/IMG_3035.PNG", "idea_3", "policy_12"),
    ("macos-auto/IMG_3034.PNG", None, None),
    ("测试4.jpg", "idea_6", "policy_12"),
    ("v4-regression/floor2.png", "idea_1", "policy_13"),
])
@pytest.mark.parametrize("scale", [1., .5])
def test_real_hud_icons(project_root, name, idea, policy, scale):
    detector = DomainDetector(project_root)
    image = read_image(project_root / "data/screen" / name)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    header = detector.detect_header(image)
    assert (header.idea, header.policy) == (idea, policy)


def test_catalog_complete_and_no_phase_guessed(project_root):
    detector = DomainDetector(project_root)
    assert sum(e["kind"] == "idea" for e in detector.catalog) == 10
    assert sum(e["kind"] == "policy" for e in detector.catalog) == 4
    assert detector.by_id["idea_8"]["removable"] is False
    assert len(detector.by_id["idea_3"]["effects"]) == 3


def test_third_floor_source_and_fog(project_root):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data/screen/macos-auto/IMG_3035.PNG", 3)
    assert {c for c, s in state.slots.items() if s.ideal_source} == {(5, 3)}
    assert state.slots[(5, 3)].domain_affected
    assert state.slots[(5, 3)].node_type == "未知的凶戾"
    assert state.slots[(5, 0)].domain_affected
    assert state.slots[(2, 3)].domain_affected
    assert not state.slots[(0, 2)].domain_affected


def test_no_hud_does_not_add_fog(project_root):
    state = SlotRecognizer(project_root).recognize_path(
        project_root / "data/screen/macos-auto/IMG_3034.PNG", 2)
    assert state.domain_idea is None
    assert not any(s.domain_affected or s.ideal_source for s in state.slots.values())


def test_positive_domain_does_not_force_ideal_source(project_root, monkeypatch):
    recognizer = SlotRecognizer(project_root)
    monkeypatch.setattr(recognizer.domain_detector, "detect_header",
                        lambda image: DomainHeader("idea_8", "policy_14", 1., 1.))
    state = recognizer.recognize_path(project_root / "data/screen/macos-auto/IMG_3035.PNG", 3)
    assert not state.domain_removable
    assert not any(s.ideal_source for s in state.slots.values())


def test_disconnected_fog_regions_are_preserved():
    hsv = np.full((400, 800, 3), (20, 230, 35), dtype=np.uint8)
    cv2.circle(hsv, (200, 200), 60, (20, 100, 55), -1)
    cv2.circle(hsv, (580, 200), 60, (20, 100, 55), -1)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    mask = DomainDetector.segment_fog(image, (80, 80, 720, 320), 80)
    assert DomainDetector.fog_coverage(mask, (200, 200), 80) > .8
    assert DomainDetector.fog_coverage(mask, (580, 200), 80) > .8
    assert DomainDetector.fog_coverage(mask, (390, 200), 80) == 0


def test_domain_corrections_support_undo(project_root):
    app = QApplication.instance() or QApplication([])
    window = StandaloneWindow(project_root)
    try:
        cell = (1, 1)
        window.state.slots[cell].present = True
        window.state.slots[cell].node_type = "未知的凶戾"
        window._toggle_domain_flag(cell, "ideal_source")
        assert window.state.slots[cell].ideal_source
        window._undo()
        assert not window.state.slots[cell].ideal_source
        window._redo()
        assert window.state.slots[cell].ideal_source
        window._set_domain("idea_8", "policy_14")
        assert not window.state.slots[cell].ideal_source
        window._undo()
        assert window.state.slots[cell].ideal_source
    finally:
        window.close()
        app.processEvents()
