from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from blackstream import window as window_module
from blackstream.window import StandaloneWindow


@pytest.fixture
def mac_window(monkeypatch, project_root):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(window_module.sys, "platform", "darwin")
    window = StandaloneWindow(project_root)
    yield window
    window.close()
    app.processEvents()


def test_mac_has_no_screen_capture(mac_window, monkeypatch):
    assert mac_window.capture_button.isHidden()
    monkeypatch.setattr(mac_window, "showMinimized", lambda: pytest.fail("screen capture started"))
    mac_window._capture_screen()
    mac_window._finish_screen_capture()


def test_cancel_preserves_map(mac_window, monkeypatch):
    original = mac_window.state
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    mac_window._choose_image()
    assert mac_window.state is original
    assert mac_window.upload_button.isEnabled()


def test_import_unicode_path_and_remember_folder(mac_window, monkeypatch, project_root, tmp_path):
    image = tmp_path / "中文 截图.png"
    image.write_bytes((project_root / "data/scenario1.png").read_bytes())
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(image), ""))
    mac_window._choose_image()
    assert mac_window.source_label.text() == image.name
    assert mac_window.image_directory == str(tmp_path)
    assert any(slot.present for slot in mac_window.state.slots.values())
    assert mac_window.upload_button.isEnabled()
    assert mac_window.floor_combo.isEnabled()
    assert mac_window.open_image_action.isEnabled()


def test_invalid_image_preserves_map_and_recovers(mac_window, monkeypatch, tmp_path):
    image = tmp_path / "invalid.png"
    image.write_bytes(b"not an image")
    original = mac_window.state
    errors = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(image), ""))
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args))
    mac_window._choose_image()
    assert errors
    assert mac_window.state is original
    assert mac_window.upload_button.isEnabled()
    assert mac_window.open_image_action.isEnabled()
