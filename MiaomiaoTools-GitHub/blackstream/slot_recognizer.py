from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from models.grid_geometry import GRID_SHAPES
from models.grid_geometry import locate_template_points
from models.map_template import MapTemplate
from rules.rule_loader import load_rules
from vision.image_io import read_image, read_image_bytes
from vision.floor_detector import detect_floor
from vision.domain_detector import DomainDetector
from vision.ideal_source_detector import IdealSourceDetector
from vision.flow_resident_detector import FlowResidentDetector
from vision.node_grid_detector import NodeGridDetector
from vision.node_type_detector import NodeTypeDetector
from vision.start_detector import StartDetector

from .map_model import EdgeState, FloorMapState, SlotState


class SlotRecognitionError(RuntimeError):
    pass


class SlotRecognizer:
    """Recognize a fixed floor lattice without selecting a preset graph."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self.node_detector = NodeTypeDetector(self.project_root)
        self.grid_detector = NodeGridDetector()
        self.domain_detector = DomainDetector(self.project_root)
        self.rules = load_rules(self.project_root / "data/rules")

    def recognize_path(
        self, path: str | Path, floor: int, *, auto_floor: bool = False
    ) -> FloorMapState:
        image = read_image(path)
        if image is None:
            raise SlotRecognitionError(f"无法读取图片：{path}")
        detected_floor = detect_floor(image) if auto_floor else None
        state = self.recognize(image, detected_floor if detected_floor is not None else floor)
        if auto_floor:
            prefix = (f"自动识别为第 {detected_floor} 层。" if detected_floor is not None
                      else f"未能确定截图层数，已按手选第 {floor} 层识别，请核对。")
            state.status = prefix + state.status
        state.source_path = str(path)
        return state

    def recognize_bytes(
        self, data: bytes, floor: int, source_name: str = ""
    ) -> FloorMapState:
        """Recognize a gallery/import stream without requiring a local path."""

        image = read_image_bytes(data)
        if image is None:
            raise SlotRecognitionError("无法读取导入的图片")
        state = self.recognize(image, floor)
        if source_name:
            state.source_path = source_name
        return state

    def recognize(self, image: np.ndarray, floor: int) -> FloorMapState:
        state = self._recognize_map(image, floor)
        header = self.domain_detector.detect_header(image)
        state.domain_idea, state.domain_policy = header.idea, header.policy
        state.domain_confidence = header.confidence
        state.domain_policy_confidence = header.policy_confidence
        if header.idea:
            state.domain_removable = self.domain_detector.by_id[header.idea]["removable"]
        offset_x, offset_y = state.map_region[:2] if state.map_region else (0, 0)
        first = state.slots[(0, 0)].screen_center
        second = state.slots[(1, 1)].screen_center
        step = min(abs(second[0]-first[0]), abs(second[1]-first[1]))
        last = state.slots[(state.columns-1, state.rows-1)].screen_center
        bounds = (first[0]+offset_x-step, first[1]+offset_y-step,
                  last[0]+offset_x+step, last[1]+offset_y+step)
        fog = self.domain_detector.segment_fog(image, bounds, step) if header.idea else None
        for slot in state.slots.values():
            if not slot.present or slot.screen_center is None:
                continue
            center = (slot.screen_center[0]+offset_x, slot.screen_center[1]+offset_y)
            if fog is not None:
                slot.domain_confidence = self.domain_detector.fog_coverage(fog, center, step)
                slot.domain_affected = slot.domain_confidence >= .5
            # Mesh can obscure the central icon, so include unresolved nodes.
            # A purple combat icon plus radial mesh is required to avoid UI,
            # merchants and ordinary glowing encounters becoming ideal sources.
            if (state.domain_removable and
                    slot.node_type in {None, "未知的凶戾", "紧急作战"} and
                    IdealSourceDetector.has_purple_combat_icon(image, center, step) and
                    IdealSourceDetector.detect(image, center, step)):
                slot.ideal_source = True
                slot.domain_affected = True
                slot.domain_confidence = max(slot.domain_confidence, .85)
                if slot.node_type is None:
                    slot.node_type = "未知的凶戾"
        state.recompute(self.rules)
        return state

    def _recognize_map(self, image: np.ndarray, floor: int) -> FloorMapState:
        if floor not in GRID_SHAPES:
            raise SlotRecognitionError(f"不支持的层数：{floor}")
        if not self._uses_relative_map_region(image):
            return self._recognize_region(image, floor)
        region = self._locate_map_region(image, floor)
        if region is None:
            return self._recognize_region(image, floor)

        left, top, right, bottom = region
        map_image = image[top:bottom, left:right]
        try:
            state = self._recognize_region(map_image, floor)
        except SlotRecognitionError:
            # The full-frame pipeline remains available for unusual screenshots
            # where the preliminary grid is correct but a nearby UI element
            # needs to remain in the local recognition crop.
            return self._recognize_region(image, floor)
        state.map_region = region
        state.status = (
            f"识别完成（已按地图区域归一化）：{sum(slot.present for slot in state.slots.values())} 个图节点，"
            f"{sum(edge.present for edge in state.edges.values())} 条连线。"
        )
        return state

    @staticmethod
    def _uses_relative_map_region(image: np.ndarray) -> bool:
        """Keep the established PC pipeline for standard game aspect ratios.

        The existing detector is already resolution-independent for 16:10 and
        16:9 desktop captures.  A separate crop is needed when phone capture
        padding creates a much wider or narrower full image.
        """

        height, width = image.shape[:2]
        aspect_ratio = width / max(1, height)
        return aspect_ratio >= 1.95 or aspect_ratio <= 1.42

    def _locate_map_region(
        self, image: np.ndarray, floor: int
    ) -> tuple[int, int, int, int] | None:
        """Find the map lattice in the full screenshot and return a padded crop.

        The first pass intentionally uses only durable structural signals.  The
        second pass then works in this crop's relative coordinate system, so
        phone HUDs and non-16:10 screen margins do not affect icon regions or
        line scoring.
        """

        columns, rows = GRID_SHAPES[floor]
        global_types = self.node_detector.global_detections(image)
        start_hit = StartDetector.detect_live_graph_center(image)
        extra_points = tuple(item[0] for item in global_types)
        if start_hit is not None:
            extra_points += (start_hit[0],)
        inspection = self.grid_detector.inspect(
            image,
            grid_size=(columns, rows),
            extra_points=extra_points,
        )
        frame = inspection.frame
        if frame is None:
            return None
        height, width = image.shape[:2]
        margin_x = max(36.0, frame.step_x * 1.0)
        margin_y = max(36.0, frame.step_y * 1.0)
        left = max(0, round(frame.left - margin_x))
        top = max(0, round(frame.top - margin_y))
        right = min(width, round(frame.right + margin_x))
        bottom = min(height, round(frame.bottom + margin_y))
        if right - left < 120 or bottom - top < 120:
            return None
        return left, top, right, bottom

    def _recognize_region(self, image: np.ndarray, floor: int) -> FloorMapState:
        columns, rows = GRID_SHAPES[floor]
        global_types = self.node_detector.global_detections(image)
        start_hit = StartDetector.detect_live_graph_center(image)
        extra_points = tuple(item[0] for item in global_types)
        if start_hit is not None:
            extra_points += (start_hit[0],)
        inspection = self.grid_detector.inspect(
            image,
            grid_size=(columns, rows),
            extra_points=extra_points,
        )
        if inspection.frame is None or not inspection.x_levels or not inspection.y_levels:
            raise SlotRecognitionError("未能定位固定节点矩阵，请确认上传的是完整地图截图。")

        state = FloorMapState.empty(floor, columns, rows)
        positions: dict[str, tuple[float, float]] = {}
        for row, y in enumerate(inspection.y_levels):
            for column, x in enumerate(inspection.x_levels):
                cell = (column, row)
                state.slots[cell].screen_center = (x, y)
                positions[self._node_id(cell)] = (x, y)

        local_types = self.node_detector.detect(image, positions)
        occupied = set(inspection.occupied_cells)
        for node_id, detection in local_types.items():
            occupied.add(self._cell_from_id(node_id))

        live_cell = None
        if start_hit is not None:
            live_cell = self._nearest_cell(start_hit[0], state)
            occupied.add(live_cell)
            state.current_cell = live_cell

        for cell, slot in state.slots.items():
            slot.present = cell in occupied
            node_id = self._node_id(cell)
            detection = local_types.get(node_id)
            if detection is not None and slot.present:
                slot.node_type = detection.node_type
                if floor in {2, 4, 5} and detection.node_type == "流窜“居民”":
                    slot.flow_resident = True
                slot.confidence = detection.confidence

        if floor in {2, 4, 5} and inspection.frame is not None:
            grid_step = min(inspection.frame.step_x, inspection.frame.step_y)
            for cell, slot in state.slots.items():
                if (
                    slot.present
                    and slot.screen_center is not None
                    and slot.node_type in {None, "流窜“居民”"}
                    and not self._resident_marker_blocked_by_hud(floor, cell, state)
                    and FlowResidentDetector.detect(
                        image, slot.screen_center, grid_step
                    )
                ):
                    # Confirm unresolved or SIFT-matched resident cells with
                    # the dedicated ring geometry detector. Other concrete
                    # labels remain authoritative to avoid nearby-ring bleed.
                    slot.present = True
                    slot.node_type = "流窜“居民”"
                    slot.flow_resident = True
                    slot.confidence = 0.92

        for edge in state.edges.values():
            edge.score, edge.confidence = self._edge_score(image, state, edge)
            edge.present = (
                state.slots[edge.first].present
                and state.slots[edge.second].present
                and edge.score >= 0.50
            )

        self._reconcile_reference_topology(state)
        if state.reference_template_id == "floor_5_template_07":
            state.trim_trailing_columns(9)
            if live_cell not in state.slots:
                live_cell = None

        if state.start_cell is not None:
            start = state.slots[state.start_cell]
            start.present = True
            start.node_type = "起点"
            start.confidence = max(
                start.confidence,
                float(start_hit[1]) if live_cell == state.start_cell and start_hit else 0.96,
            )
        if live_cell is not None and live_cell != state.start_cell:
            state.slots[live_cell].visited = True

        state.recompute(self.rules)
        state.status = (
            f"识别完成：{sum(slot.present for slot in state.slots.values())} 个图节点，"
            f"{sum(edge.present for edge in state.edges.values())} 条连线。"
        )
        return state

    @staticmethod
    def _resident_marker_blocked_by_hud(
        floor: int,
        cell: tuple[int, int],
        state: FloorMapState,
    ) -> bool:
        """Ignore the fifth-floor slot covered by the action-point HUD."""

        return floor == 5 and cell == (state.columns - 1, 0)

    def _reconcile_reference_topology(self, state: FloorMapState) -> None:
        """Use known fixed bases to resolve weak line and empty-ring evidence.

        Slot/icon recognition remains the primary signal. A reference topology
        is accepted only when its occupied cells strongly agree with the
        independently observed lattice. This also restores fixed, initially
        hidden danger-end positions that have no unique screenshot icon.
        """

        observed = {cell for cell, slot in state.slots.items() if slot.present}
        observed_edges = {key for key, edge in state.edges.items() if edge.present}
        ranked: list[
            tuple[
                float,
                float,
                float,
                MapTemplate,
                dict[str, tuple[int, int]],
                set[tuple[int, int]],
            ]
        ] = []
        pattern = f"floor_{state.floor}_template_*.json"
        for path in sorted((self.project_root / "data/templates").glob(pattern)):
            template = MapTemplate.load(path)
            cells = locate_template_points(
                ((node.id, node.x, node.y) for node in template.nodes),
                columns=state.columns,
                rows=state.rows,
            )
            template_cells = set(cells.values())
            intersection = len(observed & template_cells)
            precision = intersection / max(1, len(observed))
            recall = intersection / max(1, len(template_cells))
            occupancy_score = 2 * precision * recall / max(1e-9, precision + recall)
            reference_edges = {
                tuple(sorted((cells[first], cells[second])))
                for first, second in template.edges
            }
            edge_recall = len(observed_edges & reference_edges) / max(1, len(reference_edges))
            score = 0.70 * occupancy_score + 0.30 * edge_recall
            ranked.append(
                (score, occupancy_score, edge_recall, template, cells, template_cells)
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked or ranked[0][1] < 0.88 or ranked[0][2] < 0.55:
            return
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 0.02:
            return
        _score, _occupancy, _edge_recall, template, cells, template_cells = ranked[0]
        state.reference_template_id = template.template_id
        state.reference_template_needs_review = bool(
            template.metadata.get("requires_manual_review", False)
        )
        state.start_cell = cells[template.start_node_id]
        for cell in template_cells:
            state.slots[cell].present = True
        reference_edges = {
            tuple(sorted((cells[first], cells[second])))
            for first, second in template.edges
        }
        for key, edge in state.edges.items():
            edge.present = key in reference_edges
            if edge.present and edge.score < 0.50:
                edge.confidence = max(edge.confidence, 0.76)
        end_cells = {cells[node_id] for node_id in template.end_node_ids}
        for cell, slot in state.slots.items():
            if slot.node_type == "险路尽头" and cell not in end_cells:
                slot.node_type = None
                slot.confidence = 0.0
        for node_id in template.end_node_ids:
            cell = cells[node_id]
            slot = state.slots[cell]
            slot.present = True
            slot.fixed_end = True
            slot.node_type = "险路尽头"
            slot.confidence = max(slot.confidence, 0.96)

    @staticmethod
    def _node_id(cell: tuple[int, int]) -> str:
        return f"C{cell[0] + 1}R{cell[1] + 1}"

    @staticmethod
    def _cell_from_id(node_id: str) -> tuple[int, int]:
        column, row = node_id[1:].split("R", maxsplit=1)
        return int(column) - 1, int(row) - 1

    @staticmethod
    def _nearest_cell(
        point: tuple[float, float], state: FloorMapState
    ) -> tuple[int, int]:
        return min(
            state.slots,
            key=lambda cell: np.hypot(
                state.slots[cell].screen_center[0] - point[0],  # type: ignore[index]
                state.slots[cell].screen_center[1] - point[1],  # type: ignore[index]
            ),
        )

    @staticmethod
    def _edge_score(
        image: np.ndarray, state: FloorMapState, edge: EdgeState
    ) -> tuple[float, float]:
        first = state.slots[edge.first].screen_center
        second = state.slots[edge.second].screen_center
        if first is None or second is None:
            return 0.0, 0.0
        horizontal = edge.first[1] == edge.second[1]
        if horizontal:
            x0, x1 = sorted((first[0], second[0]))
            margin = (x1 - x0) * 0.22
            x0, x1 = x0 + margin, x1 - margin
            center = (first[1] + second[1]) / 2
            y0, y1 = center - 27, center + 27
        else:
            y0, y1 = sorted((first[1], second[1]))
            margin = (y1 - y0) * 0.22
            y0, y1 = y0 + margin, y1 - margin
            center = (first[0] + second[0]) / 2
            x0, x1 = center - 27, center + 27
        height, width = image.shape[:2]
        left, right = max(0, round(x0)), min(width, round(x1))
        top, bottom = max(0, round(y0)), min(height, round(y1))
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return 0.0, 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(2.0, (4, 4)).apply(gray)
        neutral = ((hsv[:, :, 1] < 105) & (hsv[:, :, 2] > 30)).astype(np.uint8)
        ridges = cv2.morphologyEx(
            neutral,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        if horizontal:
            center_band = ridges[max(0, ridges.shape[0] // 2 - 11) : ridges.shape[0] // 2 + 12]
            coverage = float(np.mean(np.max(center_band, axis=0)))
            brightness = float(np.mean(gray[center_band.shape[0] // 3 : -center_band.shape[0] // 3 or None])) / 255.0
        else:
            center_band = ridges[:, max(0, ridges.shape[1] // 2 - 11) : ridges.shape[1] // 2 + 12]
            coverage = float(np.mean(np.max(center_band, axis=1)))
            brightness = float(np.mean(gray[:, gray.shape[1] // 3 : -gray.shape[1] // 3 or None])) / 255.0
        score = min(1.0, 0.78 * coverage + 0.32 * brightness)
        confidence = min(1.0, abs(score - 0.50) / 0.35)
        return float(score), float(confidence)
