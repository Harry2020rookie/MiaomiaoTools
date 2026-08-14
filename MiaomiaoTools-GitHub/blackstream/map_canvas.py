from __future__ import annotations

import json
import math
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from .map_model import Cell, EdgeKey, FloorMapState


class MapCanvas(QWidget):
    nodeSelected = Signal(object)
    nodeDoubleClicked = Signal(object)
    edgeToggled = Signal(object)
    nodeTypeDropped = Signal(object, str)
    nodeContextRequested = Signal(object, object)

    def __init__(self, project_root: Path) -> None:
        super().__init__()
        self.setMinimumSize(700, 500)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.state = FloorMapState.empty(1, 5, 3)
        self.selected_cell: Cell | None = None
        self.edge_edit_mode = False
        self.show_distances = False
        self.candidate_filter: str | None = None
        self.candidate_phase = 0
        self.theme = "light"
        self.background_pixmap = QPixmap()
        self._node_points: dict[Cell, QPointF] = {}
        self._edge_paths: dict[EdgeKey, tuple[QPointF, QPointF]] = {}
        mapping = json.loads(
            (project_root / "data/rules/icon_node_types.json").read_text(encoding="utf-8")
        )
        self.icons = {
            node_type: QPixmap(str(project_root / "data/icons" / filename))
            for filename, node_type in mapping.items()
        }
        self.candidate_timer = QTimer(self)
        self.candidate_timer.setInterval(1000)
        self.candidate_timer.timeout.connect(self._advance_candidate_phase)
        self.candidate_timer.start()

    def _advance_candidate_phase(self) -> None:
        self.candidate_phase += 1
        if any(len(slot.candidates) > 1 for slot in self.state.slots.values()):
            self.update()

    def set_state(self, state: FloorMapState) -> None:
        self.state = state
        self.selected_cell = None
        self.update()

    def set_edge_edit_mode(self, enabled: bool) -> None:
        self.edge_edit_mode = enabled
        self.update()

    def set_show_distances(self, enabled: bool) -> None:
        self.show_distances = enabled
        self.update()

    def set_candidate_filter(self, node_type: str | None) -> None:
        self.candidate_filter = node_type
        self.update()

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        self.update()

    def set_background_image(self, path: str | None) -> bool:
        pixmap = QPixmap(path) if path else QPixmap()
        if path and pixmap.isNull():
            return False
        self.background_pixmap = pixmap
        self.update()
        return True

    def select_cell(self, cell: Cell | None) -> None:
        self.selected_cell = cell
        self.update()

    def display_type_for_slot(self, slot) -> tuple[str | None, bool]:
        filter_match = (
            self.candidate_filter is not None
            and (
                self.candidate_filter in slot.candidates
                or (
                    self.candidate_filter == "“居民”据点"
                    and slot.settlement_candidate
                )
            )
        )
        if filter_match:
            if self.candidate_filter == "“居民”据点":
                return slot.display_type, True
            return self.candidate_filter, True
        if len(slot.candidates) > 1:
            return slot.candidates[self.candidate_phase % len(slot.candidates)], False
        return slot.display_type, False

    def _layout_points(self) -> dict[Cell, QPointF]:
        margin_x = max(80.0, self.width() * 0.10)
        margin_y = max(90.0, self.height() * 0.18)
        usable_width = max(1.0, self.width() - margin_x * 2)
        usable_height = max(1.0, self.height() - margin_y * 2)
        return {
            (column, row): QPointF(
                margin_x + usable_width * column / max(1, self.state.columns - 1),
                margin_y + usable_height * row / max(1, self.state.rows - 1),
            )
            for row in range(self.state.rows)
            for column in range(self.state.columns)
        }

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        base = QColor("#f4f7fa") if self.theme == "light" else QColor("#0c0d10")
        painter.fillRect(self.rect(), base)
        self._paint_background_image(painter)
        self._paint_grid_texture(painter)
        self._node_points = self._layout_points()
        self._edge_paths = {}
        self._paint_edges(painter)
        self._paint_nodes(painter)

    def _paint_background_image(self, painter: QPainter) -> None:
        if self.background_pixmap.isNull():
            return
        scaled = self.background_pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.save()
        painter.setOpacity(0.46 if self.theme == "light" else 0.38)
        painter.drawPixmap(x, y, scaled)
        painter.setOpacity(1.0)
        overlay = QColor(255, 255, 255, 92) if self.theme == "light" else QColor(3, 7, 12, 112)
        painter.fillRect(self.rect(), overlay)
        painter.restore()

    def _paint_grid_texture(self, painter: QPainter) -> None:
        painter.save()
        grid = QColor(15, 23, 42, 13) if self.theme == "light" else QColor(255, 255, 255, 7)
        painter.setPen(QPen(grid, 1))
        spacing = 32
        for x in range(0, self.width(), spacing):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), spacing):
            painter.drawLine(0, y, self.width(), y)
        painter.restore()

    def _paint_edges(self, painter: QPainter) -> None:
        for key, edge in self.state.edges.items():
            first = self._node_points[edge.first]
            second = self._node_points[edge.second]
            self._edge_paths[key] = (first, second)
            if not edge.present and not self.edge_edit_mode:
                continue
            painter.save()
            if edge.present:
                color = QColor("#64748b") if self.theme == "light" else QColor("#64727c")
                color.setAlpha(235)
                painter.setPen(QPen(color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(first, second)
                highlight = QColor("#cbd5e1") if self.theme == "light" else QColor("#a4b2bb")
                painter.setPen(QPen(highlight, 1.4, Qt.PenStyle.SolidLine))
            else:
                painter.setPen(QPen(QColor(125, 139, 148, 80), 2, Qt.PenStyle.DashLine))
            painter.drawLine(first, second)
            painter.restore()

    def _paint_nodes(self, painter: QPainter) -> None:
        compact = self.state.columns >= 7 or self.state.rows >= 5
        icon_radius = 40 if compact else 52
        selection_radius = 46 if compact else 61
        badge_offset = 36 if compact else 48
        label_y = 44 if compact else 58
        for cell, slot in self.state.slots.items():
            point = self._node_points[cell]
            selected = cell == self.selected_cell
            painter.save()
            if selected:
                painter.setPen(QPen(QColor(56, 189, 248, 145), 2))
                painter.setBrush(QColor(14, 165, 233, 18))
                painter.drawEllipse(point, selection_radius, selection_radius)
            if not slot.present:
                outline = QColor(100, 116, 139, 145) if self.theme == "light" else QColor(113, 122, 130, 115)
                fill = QColor(248, 250, 252, 205) if self.theme == "light" else QColor(24, 24, 27, 140)
                painter.setPen(QPen(outline, 1.4, Qt.PenStyle.DashLine))
                painter.setBrush(fill)
                painter.drawEllipse(point, 12, 12)
                painter.setPen(QColor("#64748b") if self.theme == "light" else QColor(115, 123, 130))
                painter.setFont(QFont("Microsoft YaHei UI", 8))
                painter.drawText(
                    QRectF(point.x() - 28, point.y() + 18, 56, 18),
                    Qt.AlignmentFlag.AlignCenter,
                    f"{cell[1] + 1}-{cell[0] + 1}",
                )
                painter.restore()
                continue

            if slot.visited and cell != self.state.start_cell:
                if cell == self.state.current_cell:
                    self._paint_current_marker(painter, point)
                painter.restore()
                continue

            display_type, filter_match = self.display_type_for_slot(slot)
            if filter_match:
                painter.setPen(QPen(QColor("#fbbf24"), 3, Qt.PenStyle.DashLine))
                painter.setBrush(QColor(251, 191, 36, 22))
                painter.drawEllipse(point, selection_radius + 5, selection_radius + 5)

            if slot.node_type == "起点":
                painter.setPen(QPen(QColor("#facc15"), 3))
                painter.setBrush(QColor("#f8fafc"))
                start_radius = 28 if compact else 34
                painter.drawEllipse(point, start_radius, start_radius)
                painter.setPen(QColor("#111318"))
                painter.setFont(
                    QFont("Microsoft YaHei UI", 13 if compact else 16, QFont.Weight.Bold)
                )
                painter.drawText(QRectF(point.x() - 30, point.y() - 30, 60, 60), Qt.AlignmentFlag.AlignCenter, "起")
            elif display_type or slot.settlement_candidate:
                if slot.flow_resident:
                    self._paint_flow_resident_marker(painter, point, compact)
                else:
                    icon = self.icons.get(display_type or "“居民”据点")
                if not slot.flow_resident and icon is not None and not icon.isNull():
                    draw_radius = icon_radius
                    if slot.settlement_candidate and not slot.node_type:
                        draw_radius = 30 if compact else 38
                    target = QRectF(
                        point.x() - draw_radius,
                        point.y() - draw_radius,
                        draw_radius * 2,
                        draw_radius * 2,
                    )
                    if slot.settlement_candidate and not slot.node_type:
                        painter.setOpacity(0.42)
                    painter.drawPixmap(target.toRect(), icon)
                    painter.setOpacity(1.0)
            else:
                painter.setPen(QPen(QColor("#98a5ad"), 3))
                painter.setBrush(QColor("#f8fafc") if self.theme == "light" else QColor("#101216"))
                painter.drawEllipse(point, 10, 10)
                painter.setPen(QPen(QColor("#39434a"), 2))
                painter.drawEllipse(point, 5, 5)

            if self.show_distances and slot.distance is not None:
                badge = QPointF(
                    point.x() + badge_offset,
                    point.y() - (34 if compact else 45),
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#0ea5e9"))
                painter.drawEllipse(badge, 11, 11)
                painter.setPen(QColor("#ffffff"))
                painter.setFont(QFont("Inter", 8, QFont.Weight.Bold))
                painter.drawText(QRectF(badge.x() - 10, badge.y() - 10, 20, 20), Qt.AlignmentFlag.AlignCenter, str(slot.distance))

            label_font_size = 8 if self.state.columns >= 7 else 9
            label_width = 92 if compact else 156
            painter.setFont(
                QFont("Microsoft YaHei UI", label_font_size, QFont.Weight.DemiBold)
            )
            painter.setPen(QColor("#172033") if self.theme == "light" else QColor("#e4e4e7"))
            label = slot.label
            if len(slot.candidates) > 1 and display_type:
                label = slot.node_type or "候选"
            if slot.fixed_end and slot.confidence < 0.99:
                label += " · 固定"
            if (
                slot.node_type is not None
                or slot.inferred_type is not None
                or slot.settlement_candidate
            ):
                painter.drawText(
                    QRectF(
                        point.x() - label_width / 2,
                        point.y() + label_y,
                        label_width,
                        34,
                    ),
                    Qt.AlignmentFlag.AlignHCenter
                    | Qt.AlignmentFlag.AlignTop
                    | Qt.TextFlag.TextWordWrap,
                    label,
                )
            painter.restore()

        if (
            self.state.current_cell is not None
            and self.state.current_cell != self.state.start_cell
        ):
            current_slot = self.state.slots[self.state.current_cell]
            if not current_slot.visited or self.state.current_cell == self.state.start_cell:
                painter.save()
                self._paint_current_marker(painter, self._node_points[self.state.current_cell])
                painter.restore()

    def _paint_flow_resident_marker(
        self,
        painter: QPainter,
        point: QPointF,
        compact: bool,
    ) -> None:
        radius = 14 if compact else 18
        rect = QRectF(
            point.x() - radius,
            point.y() - radius,
            radius * 2,
            radius * 2,
        )
        painter.setPen(QPen(QColor("#0891b2"), 2.5))
        painter.setBrush(QColor("#ecfeff") if self.theme == "light" else QColor("#083344"))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor("#22d3ee"), 2))
        painter.drawLine(
            QPointF(point.x() - radius * 0.48, point.y()),
            QPointF(point.x() + radius * 0.48, point.y()),
        )

    @staticmethod
    def _paint_current_marker(painter: QPainter, point: QPointF) -> None:
        painter.setPen(QPen(QColor("#38bdf8"), 3))
        painter.setBrush(QColor(14, 165, 233, 55))
        painter.drawEllipse(point, 18, 18)
        painter.setPen(QColor("#e0f2fe"))
        painter.setFont(QFont("Microsoft YaHei UI", 8, QFont.Weight.Bold))
        painter.drawText(
            QRectF(point.x() - 38, point.y() + 22, 76, 18),
            Qt.AlignmentFlag.AlignCenter,
            "当前位置",
        )

    @staticmethod
    def _type_color(node_type: str) -> QColor:
        if node_type == "未知的诡秘":
            return QColor("#22d3ee")
        if node_type == "未知的凶戾":
            return QColor("#e879f9")
        if node_type == "险路尽头":
            return QColor("#84cc16")
        if "作战" in node_type:
            return QColor("#f472b6")
        return QColor("#a1a1aa")

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            cell = self._nearest_node(event.position(), 64)
            if cell is not None:
                self.selected_cell = cell
                self.nodeContextRequested.emit(
                    cell, event.globalPosition().toPoint()
                )
                self.update()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = event.position()
        if self.edge_edit_mode:
            nearest = self._nearest_edge(point)
            if nearest is not None:
                self.edgeToggled.emit(nearest)
                return
        if not self._node_points:
            return
        cell = self._nearest_node(point, 48)
        if cell is not None:
            self.selected_cell = cell
            self.nodeSelected.emit(cell)
            self.update()
        else:
            self.selected_cell = None
            self.nodeSelected.emit(None)
            self.update()

    def _nearest_node(self, point: QPointF, radius: float) -> Cell | None:
        if not self._node_points:
            return None
        cell, center = min(
            self._node_points.items(),
            key=lambda item: math.dist(
                (point.x(), point.y()), (item[1].x(), item[1].y())
            ),
        )
        distance = math.dist(
            (point.x(), point.y()), (center.x(), center.y())
        )
        return cell if distance <= radius else None

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        if not event.mimeData().hasText() or not self._node_points:
            event.ignore()
            return
        point = event.position()
        cell, center = min(
            self._node_points.items(),
            key=lambda item: math.dist((point.x(), point.y()), (item[1].x(), item[1].y())),
        )
        if math.dist((point.x(), point.y()), (center.x(), center.y())) <= 72:
            self.nodeTypeDropped.emit(cell, event.mimeData().text())
            event.acceptProposedAction()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton or not self._node_points:
            return
        point = event.position()
        cell, center = min(
            self._node_points.items(),
            key=lambda item: math.dist((point.x(), point.y()), (item[1].x(), item[1].y())),
        )
        if math.dist((point.x(), point.y()), (center.x(), center.y())) <= 64:
            self.nodeDoubleClicked.emit(cell)

    def _nearest_edge(self, point: QPointF) -> EdgeKey | None:
        choices: list[tuple[float, EdgeKey]] = []
        for key, (first, second) in self._edge_paths.items():
            dx = second.x() - first.x()
            dy = second.y() - first.y()
            length_sq = dx * dx + dy * dy
            if length_sq <= 0:
                continue
            ratio = max(0.0, min(1.0, ((point.x() - first.x()) * dx + (point.y() - first.y()) * dy) / length_sq))
            projected = (first.x() + ratio * dx, first.y() + ratio * dy)
            distance = math.dist((point.x(), point.y()), projected)
            if 0.12 <= ratio <= 0.88:
                choices.append((distance, key))
        if not choices:
            return None
        distance, key = min(choices)
        return key if distance <= 16 else None
