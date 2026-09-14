from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QFileDialog

from blackstream.window import StandaloneWindow
from vision.floor_detector import detect_floor
from vision.image_io import read_image


@pytest.mark.parametrize("name,floor", [
    ("测试1.jpg", 1), ("测试2.jpg", 2), ("测试3.jpg", 3), ("测试4.jpg", 4),
    ("v5-regression/floor4-residents.png", 4),
    ("v6-regression/floor5-a.png", 5),
    ("macos-auto/IMG_3034.PNG", 2), ("macos-auto/IMG_3035.PNG", 3),
])
@pytest.mark.parametrize("scale", [1, .5])
def test_header_floor_on_real_screenshots(project_root, name, floor, scale):
    image = read_image(project_root / "data/screen" / name)
    if scale != 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    assert detect_floor(image) == floor


def test_missing_header_does_not_guess(project_root):
    image = read_image(project_root / "data/screen/macos-auto/IMG_3034.PNG")
    image[:round(image.shape[0] * .14)] = 0
    assert detect_floor(image) is None
    assert detect_floor(np.zeros((800, 1600, 3), dtype=np.uint8)) is None


def test_consecutive_imports_switch_floor_without_clearing(project_root, monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = StandaloneWindow(project_root)
    try:
        for name, floor in [("IMG_3034.PNG", 2), ("IMG_3035.PNG", 3)]:
            path = project_root / "data/screen/macos-auto" / name
            monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(path), ""))
            window._choose_image()
            assert window.state.floor == floor
            assert window.floor_combo.currentData() == floor
            assert any(slot.present for slot in window.state.slots.values())
            assert window.canvas.state is window.state
            assert window.source_label.text() == name
            assert f"自动识别为第 {floor} 层" in window.status_bar.text()
    finally:
        window.close()
        app.processEvents()


def test_manual_override_and_unknown_header(project_root, tmp_path):
    from blackstream.slot_recognizer import SlotRecognizer

    recognizer = SlotRecognizer(project_root)
    path = project_root / "data/screen/macos-auto/IMG_3034.PNG"
    assert recognizer.recognize_path(path, 1, auto_floor=False).floor == 1
    image = read_image(path)
    image[:round(image.shape[0] * .14)] = 0
    target = tmp_path / "no-header.png"
    cv2.imwrite(str(target), image)
    state = recognizer.recognize_path(target, 2, auto_floor=True)
    assert state.floor == 2
    assert "未能确定截图层数" in state.status
