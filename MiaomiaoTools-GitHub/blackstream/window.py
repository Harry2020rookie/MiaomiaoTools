from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSize, QMimeData, QStandardPaths, QSignalBlocker
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from models.grid_geometry import GRID_SHAPES
from models.map_template import MapTemplate
from models.grid_geometry import locate_template_points

from .map_canvas import MapCanvas
from .map_model import Cell, FloorMapState
from .slot_recognizer import SlotRecognitionError, SlotRecognizer


class NodeTypePalette(QListWidget):
    """Expose the selected node type as plain text for canvas drops."""

    def mimeData(self, items):  # type: ignore[override]
        mime = QMimeData()
        if items:
            value = items[0].data(Qt.ItemDataRole.UserRole)
            if value:
                mime.setText(str(value))
        return mime


class StandaloneWindow(QMainWindow):
    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.project_root = project_root
        self.manual_capture_only = sys.platform == "darwin"
        self.image_directory = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.PicturesLocation
        ) or str(Path.home())
        self.recognizer = SlotRecognizer(project_root)
        self.state = FloorMapState.empty(1, *GRID_SHAPES[1])
        self.selected_cell: Cell | None = None
        self.undo_stack: list[FloorMapState] = []
        self.redo_stack: list[FloorMapState] = []
        self.theme = "light"
        self.manual_state = FloorMapState.empty(1, *GRID_SHAPES[1])
        self.manual_selected_cell: Cell | None = None
        self.manual_undo_stack: list[FloorMapState] = []
        self.manual_redo_stack: list[FloorMapState] = []
        self.setWindowTitle("妙妙工具 · 实托邦识别版")
        self.setWindowIcon(QIcon(str(project_root / "data/icons/wrong-turn.png")))
        self.resize(1320, 820)
        self.setMinimumSize(1080, 680)
        self._build_ui()
        self._apply_style()
        self._bind_shortcuts()
        self._refresh_summary()
        self._update_history_buttons()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_topbar())

        self.canvas = MapCanvas(self.project_root)
        self.canvas.set_theme(self.theme)
        self.canvas.set_state(self.state)
        self.canvas.nodeSelected.connect(self._select_node)
        self.canvas.nodeDoubleClicked.connect(self._node_double_clicked)
        self.canvas.edgeToggled.connect(self._toggle_edge)
        self.canvas.nodeContextRequested.connect(self._show_node_context_menu)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(16, 16, 16, 16)
        body_layout.setSpacing(14)
        body_layout.addWidget(self._build_sidebar())

        center_frame = QFrame()
        center_frame.setObjectName("canvasFrame")
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(self.canvas)
        body_layout.addWidget(center_frame, 1)
        body_layout.addWidget(self._build_inspector())
        self.auto_page = body

        self.manual_canvas = MapCanvas(self.project_root)
        self.manual_canvas.set_theme(self.theme)
        self.manual_canvas.set_state(self.manual_state)
        self.manual_canvas.nodeSelected.connect(self._manual_select_node)
        self.manual_canvas.nodeTypeDropped.connect(self._manual_type_dropped)
        self.manual_canvas.nodeContextRequested.connect(
            self._show_manual_node_context_menu
        )
        self.manual_page = self._build_manual_page()

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self.auto_page)
        self.mode_stack.addWidget(self.manual_page)
        root_layout.addWidget(self.mode_stack, 1)

        self.status_bar = QLabel("选择完整游戏截图，默认自动识别层数；图片仅在本机处理。")
        self.status_bar.setObjectName("statusBar")
        root_layout.addWidget(self.status_bar)
        self.setCentralWidget(root)

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topbar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(18, 10, 18, 10)
        brand = QLabel()
        brand.setObjectName("brandIcon")
        brand.setPixmap(
            QPixmap(str(self.project_root / "data/icons/wrong-turn.png")).scaled(
                34,
                34,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        title_box = QVBoxLayout()
        title = QLabel("妙妙工具")
        title.setObjectName("brandTitle")
        subtitle = QLabel("INTEGRATED STRATEGY MAP TOOL")
        subtitle.setObjectName("brandSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addWidget(brand)
        layout.addLayout(title_box)
        self.theme_button = QPushButton("夜间模式")
        self.theme_button.setObjectName("quietButton")
        self.theme_button.setCheckable(True)
        self.theme_button.toggled.connect(self._theme_changed)
        layout.addWidget(self.theme_button)
        layout.addStretch(1)
        self.auto_mode_button = QPushButton("自动识别")
        self.auto_mode_button.setCheckable(True)
        self.auto_mode_button.setChecked(True)
        self.manual_mode_button = QPushButton("手动编辑")
        self.manual_mode_button.setCheckable(True)
        self.auto_mode_button.clicked.connect(lambda: self._set_mode(0))
        self.manual_mode_button.clicked.connect(lambda: self._set_mode(1))
        layout.addWidget(self.auto_mode_button)
        layout.addWidget(self.manual_mode_button)
        test_tag = QLabel("第 1-5 层 · 桌面版")
        test_tag.setObjectName("testTag")
        layout.addWidget(test_tag)
        return bar

    def _set_mode(self, index: int) -> None:
        if not hasattr(self, "mode_stack"):
            return
        self.mode_stack.setCurrentIndex(index)
        self.auto_mode_button.setChecked(index == 0)
        self.manual_mode_button.setChecked(index == 1)
        self.status_bar.setText("自动识别模式" if index == 0 else "手动编辑模式：选择基底或拖拽节点类型到地图")

    def _build_manual_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        side = QFrame()
        side.setObjectName("sidePanel")
        side.setFixedWidth(300)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(8)
        title = QLabel("手动编辑")
        title.setObjectName("sectionTitle")
        side_layout.addWidget(title)
        side_layout.addWidget(QLabel("当前层数"))
        self.manual_floor_combo = QComboBox()
        for floor in range(1, 6):
            self.manual_floor_combo.addItem(f"第 {floor} 层", floor)
        self.manual_floor_combo.currentIndexChanged.connect(self._manual_floor_changed)
        side_layout.addWidget(self.manual_floor_combo)
        self.manual_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_panel = QFrame()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)
        preview_layout.addWidget(QLabel("选择基底预览"))
        self.template_list = QListWidget()
        self.template_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.template_list.setIconSize(QSize(216, 132))
        self.template_list.setGridSize(QSize(250, 164))
        self.template_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.template_list.setMovement(QListWidget.Movement.Static)
        self.template_list.setMinimumHeight(170)
        self.template_list.itemClicked.connect(self._manual_template_clicked)
        preview_layout.addWidget(self.template_list)
        self.manual_splitter.addWidget(preview_panel)

        type_panel = QFrame()
        type_layout = QVBoxLayout(type_panel)
        type_layout.setContentsMargins(0, 0, 0, 0)
        type_layout.setSpacing(6)
        annotate_title = QLabel("节点类型（可拖拽到地图）")
        annotate_title.setObjectName("sectionTitle")
        type_layout.addWidget(annotate_title)
        self.manual_palette = NodeTypePalette()
        self.manual_palette.setObjectName("typePalette")
        self.manual_palette.setIconSize(QSize(32, 32))
        self.manual_palette.setMinimumHeight(190)
        self.manual_palette.setDragEnabled(True)
        self.manual_palette.itemClicked.connect(self._manual_palette_clicked)
        self._populate_manual_palette()
        type_layout.addWidget(self.manual_palette)
        self.manual_splitter.addWidget(type_panel)
        self.manual_splitter.setStretchFactor(0, 1)
        self.manual_splitter.setStretchFactor(1, 1)
        self.manual_splitter.setSizes([260, 270])
        side_layout.addWidget(self.manual_splitter, 1)
        self.manual_reset_button = QPushButton("清空手动地图")
        self.manual_reset_button.clicked.connect(self._manual_reset)
        side_layout.addWidget(self.manual_reset_button)
        self.manual_distance_checkbox = QCheckBox("显示节点距离")
        self.manual_distance_checkbox.toggled.connect(self.manual_canvas.set_show_distances)
        side_layout.addWidget(self.manual_distance_checkbox)
        history = QHBoxLayout()
        self.manual_undo_button = QPushButton("撤销")
        self.manual_redo_button = QPushButton("恢复")
        self.manual_undo_button.clicked.connect(self._manual_undo)
        self.manual_redo_button.clicked.connect(self._manual_redo)
        history.addWidget(self.manual_undo_button)
        history.addWidget(self.manual_redo_button)
        side_layout.addLayout(history)
        layout.addWidget(side)

        frame = QFrame()
        frame.setObjectName("canvasFrame")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.manual_canvas)
        layout.addWidget(frame, 1)

        inspector = QFrame()
        inspector.setObjectName("sidePanel")
        inspector.setFixedWidth(260)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(14, 14, 14, 14)
        inspector_layout.setSpacing(8)
        heading = QLabel("手动节点详情")
        heading.setObjectName("sectionTitle")
        inspector_layout.addWidget(heading)
        self.manual_node_title = QLabel("未选择节点")
        self.manual_node_title.setObjectName("nodeTitle")
        inspector_layout.addWidget(self.manual_node_title)
        self.manual_node_meta = QLabel("点击地图节点查看状态")
        self.manual_node_meta.setObjectName("hint")
        self.manual_node_meta.setWordWrap(True)
        inspector_layout.addWidget(self.manual_node_meta)
        inspector_layout.addWidget(QLabel("点击左侧类型即可标注"))
        inspector_layout.addStretch(1)
        layout.addWidget(inspector)
        self._populate_template_list(1)
        return page

    def _populate_manual_palette(self) -> None:
        groups = (
            ("固定节点", ("起点", "曲折密道", "羽瞰点", "险路尽头", "命运所指", "险路恶敌")),
            ("诡秘类", ("险路小径", "不期而遇", "失与得", "先行一步", "得偿所愿", "狭路相逢", "诡意行商", "秘境行商", "安全的角落", "应急助力", "误入奇境", "未知的诡秘")),
            ("凶戾类", ("作战", "紧急作战", "“居民”据点", "流窜“居民”", "未知的凶戾")),
        )
        for group, values in groups:
            header = QListWidgetItem(group)
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            self.manual_palette.addItem(header)
            for value in values:
                item = QListWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, value)
                item.setIcon(self._node_type_icon(value))
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsDragEnabled)
                self.manual_palette.addItem(item)

    def _template_preview_path(self, floor: int, index: int) -> Path | None:
        if floor == 1:
            image_number = {1: 3, 2: 1, 3: 2}[index]
            path = self.project_root / "data" / f"scenario{image_number}.png"
        elif floor in {2, 3}:
            folder = "二层" if floor == 2 else "三层"
            image_number = (
                {1: 8, 2: 9, 3: 10, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7}[index]
                if floor == 2
                else {1: 4, 2: 5, 3: 6, 4: 7, 5: 8, 6: 9, 7: 10, 8: 1, 9: 2, 10: 3}[index]
            )
            path = self.project_root / "data" / "template" / folder / f"{image_number}.png"
        else:
            path = self.project_root / "data" / "template" / f"floor_{floor}" / f"v{index}.jpg"
        return path if path.exists() else None

    def _populate_template_list(self, floor: int) -> None:
        self.template_list.clear()
        count = 3 if floor == 1 else 10
        for index in range(1, count + 1):
            item = QListWidgetItem(f"基底 {index:02d}")
            path = self._template_preview_path(floor, index)
            if path:
                item.setIcon(QIcon(str(path)))
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.template_list.addItem(item)

    def _manual_floor_changed(self) -> None:
        floor = int(self.manual_floor_combo.currentData())
        self._populate_template_list(floor)
        self.manual_state = FloorMapState.empty(floor, *GRID_SHAPES[floor])
        self.manual_canvas.set_state(self.manual_state)
        self.manual_selected_cell = None
        self._clear_manual_inspector()
        self.status_bar.setText(f"已切换手动编辑至第 {floor} 层，请选择基底。")

    def _manual_template_clicked(self, item: QListWidgetItem) -> None:
        floor = int(self.manual_floor_combo.currentData())
        index = int(item.data(Qt.ItemDataRole.UserRole))
        template_path = self.project_root / "data" / "templates" / f"floor_{floor}_template_{index:02d}.json"
        try:
            template = MapTemplate.load(template_path)
            cells = locate_template_points(
                ((node.id, node.x, node.y) for node in template.nodes),
                columns=GRID_SHAPES[floor][0], rows=GRID_SHAPES[floor][1],
            )
        except Exception as exc:
            QMessageBox.warning(self, "基底载入失败", str(exc))
            return
        state = FloorMapState.empty(floor, *GRID_SHAPES[floor])
        state.reference_template_id = template.template_id
        for node in template.nodes:
            cell = cells[node.id]
            slot = state.slots[cell]
            slot.present = True
            if node.is_start:
                state.start_cell = cell
                state.current_cell = cell
                slot.node_type = "起点"
                slot.confidence = 1.0
            elif node.is_end:
                slot.node_type = "险路尽头"
                slot.fixed_end = True
                slot.confidence = 1.0
            else:
                # The topology templates do not encode hidden node categories;
                # leave those cells present but explicitly awaiting annotation.
                slot.node_type = None
                slot.confidence = 0.0
        for first_id, second_id in template.edges:
            key = (cells[first_id], cells[second_id])
            edge = state.edges.get(tuple(sorted(key)))
            if edge:
                edge.present = True
                edge.confidence = 1.0
        if template.template_id == "floor_5_template_07":
            state.trim_trailing_columns(9)
        state.recompute(self.recognizer.rules)
        if floor in {1, 2}:
            for slot in state.slots.values():
                if slot.present and slot.distance == 1:
                    slot.node_type = "作战"
                    slot.confidence = 1.0
        if floor == 1:
            trader_cells = {
                1: {(3, 1)},
                2: {(3, 1)},
                3: {(1, 0), (4, 1)},
            }[index]
            for cell in trader_cells:
                slot = state.slots[cell]
                slot.present = True
                slot.node_type = "诡意行商"
                slot.confidence = 1.0
        state.recompute(self.recognizer.rules)
        self.manual_state = state
        self.manual_canvas.set_state(state)
        self.manual_undo_stack.clear()
        self.manual_redo_stack.clear()
        self.manual_selected_cell = None
        self._clear_manual_inspector()
        self.status_bar.setText(f"已载入{template.template_id}，可继续拖拽或选择节点标注。")
        self._update_manual_history_buttons()

    def _manual_palette_clicked(self, item: QListWidgetItem) -> None:
        value = item.data(Qt.ItemDataRole.UserRole)
        if value and self.manual_selected_cell is not None:
            self._apply_manual_type(self.manual_selected_cell, str(value))

    def _manual_type_dropped(self, cell: Cell, value: str) -> None:
        if value in {"固定节点", "诡秘类", "凶戾类"}:
            return
        self._apply_manual_type(cell, value)

    def _apply_manual_type(self, cell: Cell, value: str) -> None:
        if value == "起点" and self.manual_state.start_cell not in {None, cell}:
            self.manual_state.slots[self.manual_state.start_cell].node_type = "未知的诡秘"
        self.manual_undo_stack.append(copy.deepcopy(self.manual_state))
        self.manual_redo_stack.clear()
        slot = self.manual_state.slots[cell]
        slot.present = True
        slot.node_type = value
        slot.confidence = 1.0
        slot.flow_resident = value == "流窜“居民”"
        if value == "起点":
            self.manual_state.start_cell = cell
            self.manual_state.current_cell = cell
        elif self.manual_state.start_cell == cell:
            self.manual_state.start_cell = None
        if value == "“居民”据点":
            self.manual_state._enforce_single_settlement()
        self.manual_state.recompute(self.recognizer.rules)
        self.manual_canvas.set_state(self.manual_state)
        self.manual_selected_cell = cell
        self.manual_canvas.select_cell(cell)
        self._manual_select_node(cell)
        self._update_manual_history_buttons()

    def _manual_select_node(self, cell: Cell | None) -> None:
        if cell is None:
            self.manual_selected_cell = None
            self._clear_manual_inspector()
            return
        self.manual_selected_cell = cell
        slot = self.manual_state.slots[cell]
        self.manual_node_title.setText(f"第 {cell[1] + 1} 行 · 第 {cell[0] + 1} 列")
        distance = "不可达" if slot.distance is None else str(slot.distance)
        self.manual_node_meta.setText(f"状态：{slot.label}\nBFS 距离：{distance}\n点击类型或拖拽类型进行标注")

    def _clear_manual_inspector(self) -> None:
        self.manual_node_title.setText("未选择节点")
        self.manual_node_meta.setText("点击地图节点查看状态")

    def _manual_reset(self) -> None:
        floor = int(self.manual_floor_combo.currentData())
        self.manual_state = FloorMapState.empty(floor, *GRID_SHAPES[floor])
        self.manual_canvas.set_state(self.manual_state)
        self.manual_selected_cell = None
        self._clear_manual_inspector()
        self._update_manual_history_buttons()

    def _update_manual_history_buttons(self) -> None:
        self.manual_undo_button.setEnabled(bool(self.manual_undo_stack))
        self.manual_redo_button.setEnabled(bool(self.manual_redo_stack))

    def _manual_undo(self) -> None:
        if self.manual_undo_stack:
            self.manual_redo_stack.append(copy.deepcopy(self.manual_state))
            self.manual_state = self.manual_undo_stack.pop()
            self.manual_canvas.set_state(self.manual_state)
            self._manual_select_node(self.manual_selected_cell)
            self._update_manual_history_buttons()

    def _manual_redo(self) -> None:
        if self.manual_redo_stack:
            self.manual_undo_stack.append(copy.deepcopy(self.manual_state))
            self.manual_state = self.manual_redo_stack.pop()
            self.manual_canvas.set_state(self.manual_state)
            self._manual_select_node(self.manual_selected_cell)
            self._update_manual_history_buttons()

    def _build_sidebar(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 16, 14, 16)
        layout.setSpacing(10)
        section = QLabel("地图输入")
        section.setObjectName("sectionTitle")
        layout.addWidget(section)
        layout.addWidget(QLabel("当前层数"))
        self.floor_combo = QComboBox()
        for floor in range(1, 6):
            self.floor_combo.addItem(f"第 {floor} 层", floor)
        self.floor_combo.currentIndexChanged.connect(self._floor_changed)
        layout.addWidget(self.floor_combo)
        self.auto_floor_checkbox = QCheckBox("自动识别层数")
        self.auto_floor_checkbox.setChecked(True)
        self.auto_floor_checkbox.setToolTip("读取截图顶部层数；无法确定时使用下拉框层数。取消勾选可手动指定。")
        layout.addWidget(self.auto_floor_checkbox)
        self.upload_button = QPushButton("选择截图并识别")
        self.upload_button.setObjectName("primaryButton")
        self.upload_button.clicked.connect(self._choose_image)
        layout.addWidget(self.upload_button)
        self.capture_button = QPushButton("实时截图并识别")
        self.capture_button.clicked.connect(self._capture_screen)
        layout.addWidget(self.capture_button)
        self.capture_button.setVisible(not self.manual_capture_only)
        self.reset_button = QPushButton("清空当前地图")
        self.reset_button.clicked.connect(self._reset_map)
        layout.addWidget(self.reset_button)
        self.distance_checkbox = QCheckBox("显示节点距离")
        self.distance_checkbox.setChecked(False)
        self.distance_checkbox.toggled.connect(self._distance_visibility_changed)
        layout.addWidget(self.distance_checkbox)
        background_row = QHBoxLayout()
        self.background_button = QPushButton("选择底图")
        self.background_button.clicked.connect(self._choose_background)
        self.clear_background_button = QPushButton("清除底图")
        self.clear_background_button.clicked.connect(self._clear_background)
        background_row.addWidget(self.background_button)
        background_row.addWidget(self.clear_background_button)
        layout.addLayout(background_row)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setObjectName("divider")
        layout.addWidget(divider)
        edit_title = QLabel("校正工具")
        edit_title.setObjectName("sectionTitle")
        layout.addWidget(edit_title)
        self.edge_button = QPushButton("编辑连线")
        self.edge_button.setCheckable(True)
        self.edge_button.toggled.connect(self._edge_mode_changed)
        layout.addWidget(self.edge_button)
        self.delete_button = QPushButton("删除节点")
        self.delete_button.setCheckable(True)
        layout.addWidget(self.delete_button)
        self.move_button = QPushButton("移动当前位置")
        self.move_button.setCheckable(True)
        layout.addWidget(self.move_button)
        self.edit_mode_group = QButtonGroup(self)
        self.edit_mode_group.setExclusive(False)
        for button in (self.edge_button, self.delete_button, self.move_button):
            self.edit_mode_group.addButton(button)
        self.delete_button.toggled.connect(self._tool_mode_changed)
        self.move_button.toggled.connect(self._tool_mode_changed)
        self.annotation_toggle = QPushButton("节点标注")
        self.annotation_toggle.setCheckable(True)
        self.annotation_toggle.toggled.connect(self._annotation_visibility_changed)
        layout.addWidget(self.annotation_toggle)
        self.annotation_palette = NodeTypePalette()
        self.annotation_palette.setObjectName("typePalette")
        self.annotation_palette.setIconSize(QSize(28, 28))
        self.annotation_palette.setEnabled(False)
        self.annotation_palette.setMinimumHeight(260)
        self.annotation_palette.setMaximumHeight(260)
        self.annotation_palette.setVisible(False)
        self.annotation_palette.itemClicked.connect(self._annotation_palette_clicked)
        self._populate_annotation_palette()
        layout.addWidget(self.annotation_palette)
        history_row = QHBoxLayout()
        self.undo_button = QPushButton("撤销")
        self.redo_button = QPushButton("恢复")
        self.undo_button.clicked.connect(self._undo)
        self.redo_button.clicked.connect(self._redo)
        history_row.addWidget(self.undo_button)
        history_row.addWidget(self.redo_button)
        layout.addLayout(history_row)
        hint = QLabel("连线模式单击线段；删除或移动模式双击节点。起点不会被删除。")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addSpacerItem(QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        self.source_label = QLabel("尚未载入截图")
        self.source_label.setObjectName("sourceLabel")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        scroll = QScrollArea()
        scroll.setObjectName("sideScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setFixedWidth(240)
        scroll.setWidget(panel)
        return scroll

    def _build_inspector(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sidePanel")
        panel.setFixedWidth(274)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        self.domain_button = QPushButton("实托邦：未检出\n查看效果 / 校正")
        self.domain_button.clicked.connect(self._edit_domain)
        layout.addWidget(self.domain_button)
        legend = QLabel("淡紫框：疑似雾区 · 紫环：理想源")
        legend.setObjectName("hint")
        layout.addWidget(legend)
        self.prediction_button = QPushButton("居民移动次数：未知 · 设置")
        self.prediction_button.clicked.connect(self._edit_prediction_context)
        layout.addWidget(self.prediction_button)
        title = QLabel("节点详情")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.node_title = QLabel("未选择节点")
        self.node_title.setObjectName("nodeTitle")
        layout.addWidget(self.node_title)
        self.node_meta = QLabel("点击地图中的槽位查看详情")
        self.node_meta.setObjectName("hint")
        self.node_meta.setWordWrap(True)
        layout.addWidget(self.node_meta)
        self.type_combo = QComboBox()
        self.type_combo.setEnabled(False)
        self._populate_types()
        self.type_combo.currentIndexChanged.connect(self._manual_type_changed)
        self.type_combo.setVisible(False)
        title = QLabel("候选筛选")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.filter_combo = QComboBox()
        self._populate_filter_types()
        self.filter_combo.currentIndexChanged.connect(self._filter_changed)
        self.filter_combo.setVisible(False)
        self.filter_palette = QListWidget()
        self.filter_palette.setObjectName("typePalette")
        self.filter_palette.setIconSize(QSize(25, 25))
        self.filter_palette.setMaximumHeight(182)
        self.filter_palette.itemClicked.connect(self._filter_palette_clicked)
        self._populate_filter_palette()
        layout.addWidget(self.filter_palette)

        title = QLabel("当前节点候选")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        self.candidate_list = QListWidget()
        self.candidate_list.setObjectName("typePalette")
        self.candidate_list.setIconSize(QSize(28, 28))
        self.candidate_list.itemClicked.connect(self._candidate_clicked)
        layout.addWidget(self.candidate_list, 1)
        summary_title = QLabel("识别摘要")
        summary_title.setObjectName("sectionTitle")
        layout.addWidget(summary_title)
        self.summary = QLabel()
        self.summary.setObjectName("summaryBox")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        return panel

    def _edit_domain(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("实托邦效果与校正")
        dialog.resize(560, 420)
        layout = QVBoxLayout(dialog)
        idea, policy = QComboBox(), QComboBox()
        idea.addItem("未识别 / 无", None)
        policy.addItem("未识别 / 无", None)
        detector = self.recognizer.domain_detector
        for entry in detector.catalog:
            combo = idea if entry["kind"] == "idea" else policy
            combo.addItem(QIcon(str(self.project_root / entry["icon"])), entry["name"], entry["id"])
        idea.setCurrentIndex(max(0, idea.findData(self.state.domain_idea)))
        policy.setCurrentIndex(max(0, policy.findData(self.state.domain_policy)))
        layout.addWidget(QLabel("理念（大图标）"))
        layout.addWidget(idea)
        layout.addWidget(QLabel("方针（右下角小图标）"))
        layout.addWidget(policy)
        effects = QLabel()
        effects.setWordWrap(True)
        effects.setMinimumHeight(140)
        layout.addWidget(effects)

        def refresh_effects():
            lines = []
            for combo in (idea, policy):
                entry = detector.by_id.get(combo.currentData())
                if not entry:
                    continue
                lines.append(entry["name"])
                if entry["kind"] == "idea":
                    lines.extend(f"{phase}：{text}" for phase, text in zip(
                        ("早期", "中期", "晚期"), entry["effects"]))
                else:
                    lines.extend(entry["effects"])
                if entry.get("note"):
                    lines.append(entry["note"])
            effects.setText("\n".join(lines) or "未识别到实托邦，可在此手动选择。")

        idea.currentIndexChanged.connect(refresh_effects)
        policy.currentIndexChanged.connect(refresh_effects)
        refresh_effects()
        hint = QLabel("强度未从截图判断，以上列出各阶段效果。\n雾区为视觉估计，非精确边界；右键节点可校正雾区及理想源。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        source = QLabel('<a href="https://prts.wiki/w/沉沦者的黑流树海/黑流数据库">资料：PRTS 黑流数据库</a>')
        source.setOpenExternalLinks(True)
        layout.addWidget(source)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._set_domain(idea.currentData(), policy.currentData())

    def _set_domain(self, idea: str | None, policy: str | None) -> None:
        if (idea, policy) == (self.state.domain_idea, self.state.domain_policy):
            return
        self._push_history()
        self.state.domain_idea, self.state.domain_policy = idea, policy
        self.state.domain_confidence = 1.0 if idea else 0.
        self.state.domain_policy_confidence = 1.0 if policy else 0.
        entry = self.recognizer.domain_detector.by_id.get(idea)
        self.state.domain_removable = entry["removable"] if entry else True
        if not self.state.domain_removable:
            for slot in self.state.slots.values():
                slot.ideal_source = False
        self.state.recompute(self.recognizer.rules)
        self._restore_history_state("已更新实托邦信息；雾区范围可逐节点校正。")

    def _toggle_domain_flag(self, cell: Cell, field: str) -> None:
        if field not in {"ideal_source", "domain_affected"}:
            return
        slot = self.state.slots[cell]
        if not slot.present:
            return
        if field == "ideal_source" and not self.state.domain_removable:
            self.status_bar.setText("当前理念不可消除，不标记理想源。")
            return
        self._push_history()
        setattr(slot, field, not getattr(slot, field))
        if field == "ideal_source" and slot.ideal_source:
            slot.domain_affected = True
            if slot.node_type not in {"未知的凶戾", "紧急作战"}:
                slot.node_type = "未知的凶戾"
        if field == "domain_affected" and not slot.domain_affected:
            slot.ideal_source = False
        slot.domain_confidence = 1.0 if slot.domain_affected else 0.
        self.state.recompute(self.recognizer.rules)
        self._restore_history_state("已校正实托邦标记。")

    def _populate_types(self) -> None:
        self.type_combo.addItem("未生成槽位", "__absent__")
        self.type_combo.addItem("空连接点", "__empty__")
        sections = (
            (
                "固定可见节点",
                ("起点", "曲折密道", "羽瞰点", "险路尽头", "命运所指", "险路恶敌"),
            ),
            (
                "诡秘类节点",
                (
                    "险路小径", "不期而遇", "失与得", "先行一步", "得偿所愿",
                    "狭路相逢", "诡意行商", "秘境行商", "安全的角落", "应急助力",
                    "误入奇境", "未知的诡秘",
                ),
            ),
            ("凶戾类节点", ("作战", "紧急作战", "“居民”据点", "流窜“居民”", "未知的凶戾")),
        )
        model = self.type_combo.model()
        for title, node_types in sections:
            self.type_combo.addItem(f"── {title} ──", None)
            heading = model.item(self.type_combo.count() - 1)
            heading.setEnabled(False)
            for node_type in node_types:
                self.type_combo.addItem(f"  {node_type}", node_type)

    def _populate_annotation_palette(self) -> None:
        groups = (
            ("固定可见节点", ("起点", "曲折密道", "羽瞰点", "险路尽头", "命运所指", "险路恶敌")),
            ("诡秘类节点", ("险路小径", "不期而遇", "失与得", "先行一步", "得偿所愿", "狭路相逢", "诡意行商", "秘境行商", "安全的角落", "应急助力", "误入奇境", "未知的诡秘")),
            ("凶戾类节点", ("作战", "紧急作战", "“居民”据点", "流窜“居民”", "未知的凶戾")),
        )
        for title, node_types in groups:
            heading = QListWidgetItem(title)
            heading.setData(Qt.ItemDataRole.UserRole, None)
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            self.annotation_palette.addItem(heading)
            for node_type in node_types:
                item = QListWidgetItem(node_type)
                item.setData(Qt.ItemDataRole.UserRole, node_type)
                item.setIcon(self._node_type_icon(node_type))
                self.annotation_palette.addItem(item)

    def _populate_filter_types(self) -> None:
        self.filter_combo.addItem("不过滤候选", "__all__")
        groups = (
            ("诡秘类候选", ("险路小径", "不期而遇", "失与得", "先行一步", "得偿所愿", "狭路相逢", "诡意行商", "秘境行商", "安全的角落", "应急助力", "误入奇境")),
            ("凶戾类候选", ("作战", "紧急作战")),
            ("特殊推断", ("“居民”据点",)),
        )
        model = self.filter_combo.model()
        for title, node_types in groups:
            self.filter_combo.addItem(title, None)
            model.item(self.filter_combo.count() - 1).setEnabled(False)
            for node_type in node_types:
                self.filter_combo.addItem(node_type, node_type)
                self.filter_combo.setItemIcon(
                    self.filter_combo.count() - 1,
                    self._node_type_icon(node_type),
                )

    def _populate_filter_palette(self) -> None:
        reset = QListWidgetItem("显示全部候选")
        reset.setData(Qt.ItemDataRole.UserRole, "__all__")
        self.filter_palette.addItem(reset)
        groups = (
            ("诡秘类候选", ("险路小径", "不期而遇", "失与得", "先行一步", "得偿所愿", "狭路相逢", "诡意行商", "秘境行商", "安全的角落", "应急助力", "误入奇境")),
            ("凶戾类候选", ("作战", "紧急作战")),
            ("特殊推断", ("“居民”据点",)),
        )
        for title, node_types in groups:
            heading = QListWidgetItem(title)
            heading.setData(Qt.ItemDataRole.UserRole, None)
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            self.filter_palette.addItem(heading)
            for node_type in node_types:
                item = QListWidgetItem(node_type)
                item.setData(Qt.ItemDataRole.UserRole, node_type)
                item.setIcon(self._node_type_icon(node_type))
                self.filter_palette.addItem(item)

    def _node_type_icon(self, node_type: str) -> QIcon:
        if node_type == "流窜“居民”":
            marker = QPixmap(30, 30)
            marker.fill(Qt.GlobalColor.transparent)
            painter = QPainter(marker)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QColor("#0891b2"))
            painter.setBrush(QColor("#ecfeff"))
            painter.drawRect(6, 6, 18, 18)
            painter.drawLine(10, 15, 20, 15)
            painter.end()
            return QIcon(marker)
        pixmap = self.canvas.icons.get(node_type)
        if pixmap is not None and not pixmap.isNull():
            return QIcon(pixmap)
        if node_type == "起点":
            marker = QPixmap(30, 30)
            marker.fill(Qt.GlobalColor.transparent)
            painter = QPainter(marker)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#facc15"))
            painter.drawEllipse(8, 3, 14, 24)
            painter.end()
            return QIcon(marker)
        return QIcon()

    def _annotation_palette_clicked(self, item: QListWidgetItem) -> None:
        value = item.data(Qt.ItemDataRole.UserRole)
        if value is None or self.selected_cell is None:
            return
        index = self.type_combo.findData(value)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

    def _sync_annotation_palette(self, value: str) -> None:
        for index in range(self.annotation_palette.count()):
            item = self.annotation_palette.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == value:
                self.annotation_palette.setCurrentRow(index)
                return

    def _apply_style(self) -> None:
        if self.theme == "dark":
            colors = {
                "window": "#111318", "panel": "#191d24", "surface": "#12161c",
                "border": "#303743", "text": "#e7edf5", "muted": "#929dab",
                "accent": "#22b5e8", "accent_soft": "#123342", "hover": "#242a33",
            }
        else:
            colors = {
                "window": "#eef2f6", "panel": "#ffffff", "surface": "#f7f9fb",
                "border": "#d5dce5", "text": "#172033", "muted": "#667085",
                "accent": "#0284c7", "accent_soft": "#e0f2fe", "hover": "#edf3f8",
            }
        self.setStyleSheet(
            f"""
            QMainWindow, QWidget {{ background: {colors['window']}; color: {colors['text']}; font-family: {'PingFang SC' if self.manual_capture_only else 'Microsoft YaHei UI'}; font-size: 12px; }}
            QLabel, QCheckBox {{ background: transparent; }}
            #topbar {{ background: {colors['panel']}; border-bottom: 1px solid {colors['border']}; }}
            #brandIcon {{ min-width: 36px; max-width: 36px; min-height: 36px; max-height: 36px; border-radius: 5px; background: {colors['accent_soft']}; border: 1px solid {colors['border']}; qproperty-alignment: AlignCenter; }}
            #brandTitle {{ font-size: 15px; font-weight: 700; color: {colors['text']}; }}
            #brandSubtitle {{ font-size: 8px; color: {colors['muted']}; }}
            #testTag {{ color: {colors['accent']}; background: {colors['accent_soft']}; border: 1px solid {colors['border']}; border-radius: 4px; padding: 5px 8px; font-size: 10px; }}
            #sidePanel, #canvasFrame {{ background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 6px; }}
            #sideScroll {{ background: transparent; }}
            #sectionTitle {{ color: {colors['text']}; font-size: 12px; font-weight: 700; padding: 4px 0 2px 0; border-bottom: 1px solid {colors['border']}; }}
            #nodeTitle {{ color: {colors['accent']}; font-size: 16px; font-weight: 700; padding: 3px 0; }}
            #hint, #sourceLabel {{ color: {colors['muted']}; font-size: 10px; }}
            #summaryBox {{ background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 5px; padding: 8px; color: {colors['muted']}; }}
            #statusBar {{ background: {colors['panel']}; border-top: 1px solid {colors['border']}; color: {colors['muted']}; padding: 7px 16px; }}
            QPushButton {{ background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 4px; padding: 7px 9px; color: {colors['text']}; }}
            QPushButton:hover {{ background: {colors['hover']}; border-color: {colors['accent']}; }}
            QPushButton:checked {{ color: {colors['accent']}; border-color: {colors['accent']}; background: {colors['accent_soft']}; }}
            QPushButton:disabled {{ color: {colors['muted']}; background: {colors['surface']}; }}
            #primaryButton {{ background: {colors['accent']}; border-color: {colors['accent']}; color: white; font-weight: 700; }}
            #quietButton {{ padding: 5px 9px; }}
            QComboBox {{ background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 4px; padding: 6px 9px; min-height: 19px; }}
            QComboBox:focus {{ border-color: {colors['accent']}; }}
            QComboBox QAbstractItemView {{ background: {colors['panel']}; color: {colors['text']}; selection-background-color: {colors['accent_soft']}; border: 1px solid {colors['border']}; }}
            QListWidget {{ background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 5px; padding: 3px; outline: none; }}
            QListWidget::item {{ min-height: 28px; padding: 3px 6px; border-bottom: 1px solid {colors['border']}; }}
            QListWidget::item:selected {{ background: {colors['accent_soft']}; color: {colors['accent']}; }}
            #typePalette::item {{ min-height: 31px; }}
            #divider {{ color: {colors['border']}; }}
            """
        )

    def _bind_shortcuts(self) -> None:
        action = QAction(self)
        action.setShortcut(QKeySequence.StandardKey.Open)
        action.triggered.connect(self._choose_image)
        self.addAction(action)
        self.open_image_action = action
        undo_action = QAction(self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self._undo)
        self.addAction(undo_action)
        redo_action = QAction(self)
        redo_action.setShortcuts([QKeySequence.StandardKey.Redo, QKeySequence("Ctrl+Y")])
        redo_action.triggered.connect(self._redo)
        self.addAction(redo_action)

    def _floor_changed(self) -> None:
        self._reset_map()

    def _distance_visibility_changed(self, enabled: bool) -> None:
        self.canvas.set_show_distances(enabled)

    def _theme_changed(self, dark: bool) -> None:
        self.theme = "dark" if dark else "light"
        self.theme_button.setText("浅色模式" if dark else "夜间模式")
        self.canvas.set_theme(self.theme)
        if hasattr(self, "manual_canvas"):
            self.manual_canvas.set_theme(self.theme)
        self._apply_style()

    def _choose_background(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择地图底图",
            str(self.project_root / "data"),
            "图片 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return
        if not self.canvas.set_background_image(path):
            QMessageBox.warning(self, "底图载入失败", "无法读取所选图片。")
            return
        self.status_bar.setText(f"已应用地图底图：{Path(path).name}")

    def _clear_background(self) -> None:
        self.canvas.set_background_image(None)
        self.status_bar.setText("地图底图已清除。")

    def _annotation_visibility_changed(self, visible: bool) -> None:
        self.annotation_palette.setVisible(visible)
        self.annotation_toggle.setText("收起节点标注" if visible else "节点标注")

    def _filter_palette_clicked(self, item: QListWidgetItem) -> None:
        value = item.data(Qt.ItemDataRole.UserRole)
        if value is None:
            return
        index = self.filter_combo.findData(value)
        if index >= 0:
            self.filter_combo.setCurrentIndex(index)

    def _filter_changed(self) -> None:
        value = self.filter_combo.currentData()
        node_type = None if value in {None, "__all__"} else str(value)
        self.canvas.set_candidate_filter(node_type)
        for index in range(self.filter_palette.count()):
            item = self.filter_palette.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == (node_type or "__all__"):
                self.filter_palette.setCurrentRow(index)
                break
        if node_type:
            self.status_bar.setText(f"已筛选候选：{node_type}")
        elif hasattr(self, "state"):
            self.status_bar.setText(self.state.status)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择完整游戏截图",
            self.image_directory,
            "图片 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)",
        )
        if not path:
            return
        self.image_directory = str(Path(path).parent)
        try:
            self._set_recognition_busy(True)
            state = self.recognizer.recognize_path(
                path, int(self.floor_combo.currentData()),
                auto_floor=self.auto_floor_checkbox.isChecked(),
            )
        except (SlotRecognitionError, ValueError) as exc:
            QMessageBox.warning(self, "识别失败", str(exc))
            self.status_bar.setText(str(exc))
            return
        finally:
            self._set_recognition_busy(False)
        self._apply_recognition(state, Path(path).name)

    def _capture_screen(self) -> None:
        if self.manual_capture_only:
            return
        self._set_recognition_busy(True)
        self.status_bar.setText("窗口已最小化，正在截取主屏幕...")
        self.showMinimized()
        QTimer.singleShot(800, self._finish_screen_capture)

    def _finish_screen_capture(self) -> None:
        if self.manual_capture_only:
            return
        try:
            import mss

            with mss.mss() as capture:
                shot = capture.grab(capture.monitors[1])
                image = np.asarray(shot)[:, :, :3].copy()
            self.showNormal()
            self.raise_()
            self.activateWindow()
            QApplication.processEvents()
            state = self.recognizer.recognize(image, int(self.floor_combo.currentData()))
        except (SlotRecognitionError, ValueError, OSError) as exc:
            self.showNormal()
            QMessageBox.warning(self, "截图识别失败", str(exc))
            self.status_bar.setText(str(exc))
            return
        finally:
            self._set_recognition_busy(False)
        self._apply_recognition(state, "实时截图")

    def _set_recognition_busy(self, busy: bool) -> None:
        self.upload_button.setEnabled(not busy)
        self.capture_button.setEnabled(not busy)
        self.floor_combo.setEnabled(not busy)
        self.auto_floor_checkbox.setEnabled(not busy)
        self.open_image_action.setEnabled(not busy)
        if busy:
            self.status_bar.setText("正在识别固定槽位、节点图标和连线...")
        QApplication.processEvents()

    def _apply_recognition(self, state: FloorMapState, source: str) -> None:
        # Updating the display must not fire _floor_changed and clear the map.
        with QSignalBlocker(self.floor_combo):
            self.floor_combo.setCurrentIndex(self.floor_combo.findData(state.floor))
        self.state = state
        self.canvas.set_state(state)
        self.filter_combo.setCurrentIndex(0)
        self.source_label.setText(source)
        self.status_bar.setText(state.status)
        self._clear_history()
        self._clear_inspector()
        self._refresh_summary()

    def _reset_map(self) -> None:
        floor = int(self.floor_combo.currentData())
        self.state = FloorMapState.empty(floor, *GRID_SHAPES[floor])
        self.canvas.set_state(self.state)
        self.filter_combo.setCurrentIndex(0)
        self.source_label.setText("尚未载入截图")
        self.status_bar.setText("地图已清空。")
        self._clear_history()
        self._clear_inspector()
        self._refresh_summary()

    def _edge_mode_changed(self, enabled: bool) -> None:
        if enabled:
            self.delete_button.setChecked(False)
            self.move_button.setChecked(False)
        self.canvas.set_edge_edit_mode(enabled)
        self.status_bar.setText("连线编辑已开启，点击线段进行切换。" if enabled else self.state.status)

    def _tool_mode_changed(self, enabled: bool) -> None:
        if not enabled:
            if not any(
                button.isChecked()
                for button in (self.edge_button, self.delete_button, self.move_button)
            ):
                self.status_bar.setText(self.state.status)
            return
        active_button = self.sender()
        for button in (self.edge_button, self.delete_button, self.move_button):
            if button is not active_button:
                button.setChecked(False)
        if active_button is self.delete_button:
            self.status_bar.setText("删除模式：双击非起点节点进行删除。")
        elif active_button is self.move_button:
            self.status_bar.setText("移动模式：双击目标节点，节点事件将消失。")

    def _select_node(self, cell: Cell | None) -> None:
        if cell is None:
            self._clear_inspector()
            self.canvas.select_cell(None)
            return
        self.selected_cell = cell
        slot = self.state.slots[cell]
        self.node_title.setText(f"第 {cell[1] + 1} 行 · 第 {cell[0] + 1} 列")
        distance = "不可达" if slot.distance is None else str(slot.distance)
        self.node_meta.setText(f"状态：{slot.label}\nBFS 距离：{distance}")
        if slot.domain_affected:
            self.node_meta.setText(self.node_meta.text() + "\n实托邦范围：已标记（可右键校正）")
        target = "__absent__" if not slot.present else (slot.node_type or "__empty__")
        index = self.type_combo.findData(target)
        self.type_combo.blockSignals(True)
        self.type_combo.setCurrentIndex(max(0, index))
        self.type_combo.blockSignals(False)
        self.type_combo.setEnabled(True)
        self.annotation_palette.setEnabled(True)
        self._sync_annotation_palette(target)
        self._populate_candidate_list(slot)

    @staticmethod
    def _context_candidates(slot) -> list[str]:
        values = list(slot.candidates)
        if slot.settlement_candidate and "“居民”据点" not in values:
            values.append("“居民”据点")
        return values

    def _show_node_context_menu(self, cell: Cell, global_pos) -> None:
        self._select_node(cell)
        slot = self.state.slots[cell]
        menu = QMenu(self)
        for field, text in (("domain_affected", "实托邦范围节点"), ("ideal_source", "理想源")):
            mark = menu.addAction(text)
            mark.setCheckable(True)
            mark.setChecked(getattr(slot, field))
            mark.setEnabled(slot.present and (field != "ideal_source" or self.state.domain_removable))
            mark.triggered.connect(lambda _checked=False, key=field: self._toggle_domain_flag(cell, key))
        menu.addSeparator()
        candidates_menu = menu.addMenu("当前节点候选")
        candidates = self._context_candidates(slot)
        if not candidates:
            empty = candidates_menu.addAction("当前没有可用候选")
            empty.setEnabled(False)
        else:
            for value in candidates:
                action = candidates_menu.addAction(self._node_type_icon(value), value)
                action.triggered.connect(
                    lambda _checked=False, candidate=value, target=cell:
                    self._apply_context_candidate(target, candidate)
                )

        editor_menu = menu.addMenu("编辑节点")
        empty_action = editor_menu.addAction("空连接点")
        empty_action.triggered.connect(
            lambda _checked=False, target=cell: self._apply_context_type(
                target, "__empty__"
            )
        )
        for index in range(self.type_combo.count()):
            value = self.type_combo.itemData(index)
            if not value or value in {"__absent__", "__empty__"}:
                continue
            action = editor_menu.addAction(self._node_type_icon(str(value)), str(value))
            action.triggered.connect(
                lambda _checked=False, candidate=str(value), target=cell:
                self._apply_context_type(target, candidate)
            )
        menu.exec(global_pos)

    def _apply_context_candidate(self, cell: Cell, candidate: str) -> None:
        self._select_node(cell)
        item = QListWidgetItem(candidate)
        item.setData(Qt.ItemDataRole.UserRole, candidate)
        self._candidate_clicked(item)

    def _apply_context_type(self, cell: Cell, value: str) -> None:
        self._select_node(cell)
        index = self.type_combo.findData(value)
        if index >= 0:
            self.type_combo.setCurrentIndex(index)

    def _show_manual_node_context_menu(self, cell: Cell, global_pos) -> None:
        self._manual_select_node(cell)
        menu = QMenu(self)
        candidates_menu = menu.addMenu("当前节点候选")
        candidates = self._context_candidates(self.manual_state.slots[cell])
        if not candidates:
            empty = candidates_menu.addAction("手动基底暂无推断候选")
            empty.setEnabled(False)
        else:
            for value in candidates:
                action = candidates_menu.addAction(self._node_type_icon(value), value)
                action.triggered.connect(
                    lambda _checked=False, candidate=value, target=cell:
                    self._apply_manual_type(target, candidate)
                )
        editor_menu = menu.addMenu("编辑节点")
        for index in range(self.manual_palette.count()):
            item = self.manual_palette.item(index)
            value = item.data(Qt.ItemDataRole.UserRole)
            if not value:
                continue
            action = editor_menu.addAction(self._node_type_icon(str(value)), str(value))
            action.triggered.connect(
                lambda _checked=False, candidate=str(value), target=cell:
                self._apply_manual_type(target, candidate)
            )
        menu.exec(global_pos)

    def _populate_candidate_list(self, slot) -> None:
        self.candidate_list.clear()
        groups = (
            ("诡秘类候选", set(self.recognizer.rules.categories["unknown_mystery"])),
            ("凶戾类候选", {"作战", "紧急作战"}),
            ("特殊推断", {"“居民”据点"}),
        )
        if slot.candidates or slot.settlement_candidate:
            available = list(slot.candidates)
            if slot.settlement_candidate and "“居民”据点" not in available:
                available.append("“居民”据点")
            for title, node_types in groups:
                members = [candidate for candidate in available if candidate in node_types]
                if not members:
                    continue
                heading = QListWidgetItem(title)
                heading.setData(Qt.ItemDataRole.UserRole, None)
                heading.setFlags(Qt.ItemFlag.NoItemFlags)
                self.candidate_list.addItem(heading)
                for candidate in members:
                    item = QListWidgetItem(candidate)
                    item.setData(Qt.ItemDataRole.UserRole, candidate)
                    item.setIcon(self._node_type_icon(candidate))
                    self.candidate_list.addItem(item)
            return
        message = (
            "当前距离下没有合法候选"
            if slot.node_type in {"未知的诡秘", "未知的凶戾"}
            else "该节点不需要推断"
        )
        item = QListWidgetItem(message)
        item.setData(Qt.ItemDataRole.UserRole, None)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self.candidate_list.addItem(item)

    def _manual_type_changed(self) -> None:
        if self.selected_cell is None or not self.type_combo.isEnabled():
            return
        value = str(self.type_combo.currentData())
        slot = self.state.slots[self.selected_cell]
        current = "__absent__" if not slot.present else (slot.node_type or "__empty__")
        if value == current:
            return
        if self.selected_cell == self.state.start_cell and value == "__absent__":
            self.status_bar.setText("起点始终保留，无法删除。")
            self._select_node(self.selected_cell)
            return
        self._push_history()
        if value == "__absent__":
            slot.present = False
            slot.node_type = None
            if self.state.start_cell == self.selected_cell:
                self.state.start_cell = None
            for edge in self.state.edges.values():
                if self.selected_cell in (edge.first, edge.second):
                    edge.present = False
        elif value == "__empty__":
            slot.present = True
            slot.node_type = None
            if self.state.start_cell == self.selected_cell:
                self.state.start_cell = None
        elif value == "起点":
            if self.state.start_cell is not None:
                self.state.slots[self.state.start_cell].node_type = None
            self.state.start_cell = self.selected_cell
            slot.present = True
            slot.node_type = "起点"
        else:
            slot.present = True
            slot.node_type = value
            if self.state.start_cell == self.selected_cell:
                self.state.start_cell = None
        if value == "“居民”据点":
            self._demote_other_settlements(self.selected_cell)
        slot.flow_resident = value == "流窜“居民”"
        if value not in {"未知的凶戾", "紧急作战"}:
            slot.ideal_source = False
        slot.confidence = 1.0
        self.state.recompute(self.recognizer.rules)
        self.canvas.update()
        self._select_node(self.selected_cell)
        self._refresh_summary()

    def _candidate_clicked(self, item) -> None:
        if self.selected_cell is None:
            return
        candidate = item.data(Qt.ItemDataRole.UserRole)
        candidate = item.text() if candidate is None else str(candidate)
        slot = self.state.slots[self.selected_cell]
        if candidate not in slot.candidates and not (
            candidate == "“居民”据点" and slot.settlement_candidate
        ):
            return
        self._push_history()
        slot.present = True
        slot.node_type = candidate
        if candidate == "“居民”据点":
            self._demote_other_settlements(self.selected_cell)
        slot.flow_resident = candidate == "流窜“居民”"
        slot.inferred_type = None
        slot.confidence = 1.0
        self.state.recompute(self.recognizer.rules)
        self.canvas.update()
        self._select_node(self.selected_cell)
        self._refresh_summary()
        self.status_bar.setText(f"已将节点标注为：{candidate}")

    def _demote_other_settlements(self, keep_cell: Cell) -> None:
        for cell, other in self.state.slots.items():
            if cell != keep_cell and other.node_type == "“居民”据点":
                other.node_type = "未知的凶戾"
                other.inferred_type = None
                other.confidence = min(other.confidence, 0.90)

    def _toggle_edge(self, key: object) -> None:
        self._push_history()
        edge = self.state.edges[key]  # type: ignore[index]
        edge.present = not edge.present
        if edge.present:
            self.state.slots[edge.first].present = True
            self.state.slots[edge.second].present = True
        edge.confidence = 1.0
        self.state.recompute(self.recognizer.rules)
        self.canvas.update()
        if self.selected_cell is not None:
            self._select_node(self.selected_cell)
        self._refresh_summary()

    def _node_double_clicked(self, cell: Cell) -> None:
        slot = self.state.slots[cell]
        if self.delete_button.isChecked():
            if cell == self.state.start_cell:
                self.status_bar.setText("起点始终保留，无法删除。")
                return
            if not slot.present:
                return
            self._push_history()
            slot.present = False
            slot.node_type = None
            slot.inferred_type = None
            slot.visited = False
            slot.flow_resident = False
            slot.ideal_source = False
            for edge in self.state.edges.values():
                if cell in (edge.first, edge.second):
                    edge.present = False
            if self.state.current_cell == cell:
                self.state.current_cell = self.state.start_cell
            self._after_mutation("节点已删除。")
        elif self.move_button.isChecked():
            if not slot.present:
                return
            self._push_history()
            if self.state.current_cell is not None and self.state.current_cell != self.state.start_cell:
                self.state.slots[self.state.current_cell].visited = True
            self.state.current_cell = cell
            if cell != self.state.start_cell:
                slot.visited = True
            self._after_mutation("当前位置已更新。")

    def _edit_prediction_context(self) -> None:
        moves, accepted = QInputDialog.getInt(
            self, "当前截图的预测条件",
            "进入本层后、拍摄此截图前的移动次数：\n"
            "-1 = 未知；0 = 刚进入本层。填写移动次数，不是行动力消耗。\n"
            "每张新截图会重置为未知。",
            -1 if self.state.resident_moves is None else self.state.resident_moves,
            -1, 9999,
        )
        if accepted:
            self._push_history()
            self.state.resident_moves = None if moves < 0 else moves
            self._after_mutation("已更新居民据点预测条件；候选不代表必定存在据点。")

    def _after_mutation(self, message: str) -> None:
        self.state.recompute(self.recognizer.rules)
        self.canvas.update()
        if self.selected_cell is not None:
            self._select_node(self.selected_cell)
        self._refresh_summary()
        self.status_bar.setText(message)

    def _push_history(self) -> None:
        self.undo_stack.append(copy.deepcopy(self.state))
        self.redo_stack.clear()
        self._update_history_buttons()

    def _undo(self) -> None:
        if not self.undo_stack:
            return
        self.redo_stack.append(copy.deepcopy(self.state))
        self.state = self.undo_stack.pop()
        self._restore_history_state("已撤销上一步操作。")

    def _redo(self) -> None:
        if not self.redo_stack:
            return
        self.undo_stack.append(copy.deepcopy(self.state))
        self.state = self.redo_stack.pop()
        self._restore_history_state("已恢复上一步操作。")

    def _restore_history_state(self, message: str) -> None:
        self.canvas.set_state(self.state)
        if self.selected_cell is not None:
            self.canvas.select_cell(self.selected_cell)
            self._select_node(self.selected_cell)
        self._refresh_summary()
        self._update_history_buttons()
        self.status_bar.setText(message)

    def _clear_history(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()
        self._update_history_buttons()

    def _update_history_buttons(self) -> None:
        if hasattr(self, "undo_button"):
            self.undo_button.setEnabled(bool(self.undo_stack))
            self.redo_button.setEnabled(bool(self.redo_stack))

    def _clear_inspector(self) -> None:
        self.selected_cell = None
        self.node_title.setText("未选择节点")
        self.node_meta.setText("点击地图中的槽位查看详情")
        self.type_combo.setEnabled(False)
        self.annotation_palette.setEnabled(False)
        self.annotation_palette.setCurrentRow(-1)
        self.candidate_list.clear()

    def _refresh_summary(self) -> None:
        catalog = self.recognizer.domain_detector.by_id
        idea = catalog.get(self.state.domain_idea, {}).get("name", "未检出")
        policy = catalog.get(self.state.domain_policy, {}).get("name", "方针未检出")
        self.domain_button.setText(f"理念：{idea}\n方针：{policy} · 查看/校正")
        moves = self.state.resident_moves
        self.prediction_button.setText(f"居民移动次数：{'未知' if moves is None else moves} · 设置")
        present = sum(slot.present for slot in self.state.slots.values())
        edges = sum(edge.present for edge in self.state.edges.values())
        unknown = sum(
            slot.node_type in {"未知的诡秘", "未知的凶戾"}
            for slot in self.state.slots.values()
        )
        fixed_ends = sum(slot.fixed_end for slot in self.state.slots.values())
        ideal_sources = sum(slot.ideal_source for slot in self.state.slots.values())
        flow_residents = sum(slot.flow_resident for slot in self.state.slots.values())
        settlement_candidates = sum(
            slot.settlement_candidate for slot in self.state.slots.values()
        )
        self.summary.setText(
            f"图节点  {present}\n连线  {edges}\n待推断节点  {unknown}\n"
            f"理想源  {ideal_sources}\n流窜居民  {flow_residents}\n"
            f"居民据点候选  {settlement_candidates}\n固定险路尽头  {fixed_ends}"
            + ("\n\n" + "\n".join(self.state.prediction_warnings) if self.state.prediction_warnings else "")
        )
