from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from models.grid_geometry import GRID_SHAPES, fit_template_axis
from models.map_template import MapTemplate


@dataclass(slots=True)
class StructuralMatch:
    template: MapTemplate
    score: float
    homography: np.ndarray
    node_positions: dict[str, tuple[float, float]]
    matched_nodes: int
    coverage: float
    scale_x: float
    scale_y: float
    method: str = "grid-first"
    inspection: GridInspection | None = None


@dataclass(frozen=True, slots=True)
class CoordinateFrame:
    left: float
    top: float
    right: float
    bottom: float
    step_x: float
    step_y: float


@dataclass(frozen=True, slots=True)
class GridInspection:
    points: tuple[tuple[float, float], ...]
    frame: CoordinateFrame | None
    columns: int | None = None
    rows: int | None = None
    x_levels: tuple[float, ...] = ()
    y_levels: tuple[float, ...] = ()
    occupied_cells: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _TemplateGeometry:
    node_x: np.ndarray
    node_y: np.ndarray
    x_levels: np.ndarray
    y_levels: np.ndarray
    cells: frozenset[tuple[int, int]]
    source_x: np.ndarray
    source_y: np.ndarray
    source_points: np.ndarray


class NodeGridDetector:
    """Match a map from its node layout when the live background is unsuitable.

    Resource screenshots and live gameplay can have different backgrounds,
    labels and node states.  The stable signal is the orthogonal graph layout:
    small black rings, large circular icons, and the start ring.  Detection is
    performed on a bounded-size copy so it remains fast on 4K screenshots.
    """

    target_width = 1280

    def __init__(self) -> None:
        self._template_geometry: dict[int, _TemplateGeometry] = {}

    def prepare_template(self, template: MapTemplate) -> None:
        """Precompute immutable grid geometry once for a loaded template."""

        key = id(template)
        if key in self._template_geometry:
            return
        grid_size = GRID_SHAPES.get(template.floor)
        if grid_size is None or not template.image_width or not template.image_height:
            return
        columns, rows = grid_size
        node_x = np.asarray([node.x for node in template.nodes], dtype=np.float64)
        node_y = np.asarray([node.y for node in template.nodes], dtype=np.float64)
        x_levels = self._template_levels(node_x, columns)
        y_levels = self._template_levels(node_y, rows)
        if x_levels is None or y_levels is None:
            return
        cells = frozenset(
            (
                int(np.argmin(np.abs(x_levels - node.x))),
                int(np.argmin(np.abs(y_levels - node.y))),
            )
            for node in template.nodes
        )
        width = float(template.image_width)
        height = float(template.image_height)
        self._template_geometry[key] = _TemplateGeometry(
            node_x=node_x,
            node_y=node_y,
            x_levels=x_levels,
            y_levels=y_levels,
            cells=cells,
            source_x=x_levels * width,
            source_y=y_levels * height,
            source_points=np.column_stack((node_x * width, node_y * height)),
        )

    def _geometry(self, template: MapTemplate) -> _TemplateGeometry | None:
        self.prepare_template(template)
        return self._template_geometry.get(id(template))

    @staticmethod
    def _deduplicate(points: list[tuple[float, float]], radius: float = 5.0) -> np.ndarray:
        kept: list[tuple[float, float]] = []
        for point in points:
            if any(np.hypot(point[0] - old[0], point[1] - old[1]) < radius for old in kept):
                continue
            kept.append(point)
        return np.asarray(kept, dtype=np.float64).reshape(-1, 2)

    @classmethod
    def _consolidate_node_circles(
        cls,
        circles: list[tuple[float, float, float]],
        *,
        labeled_node_points: list[tuple[float, float]],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Collapse one labeled icon's concentric details into one node point.

        Exposed connector nodes use small rings and are deliberately retained
        independently.  Named nodes have a large icon plus a text label below;
        SIFT supplies their semantic icon centers as ``labeled_node_points``.
        Around each such center all Hough responses are replaced by exactly one
        point.  Large icon circles without a semantic match are also clustered,
        so an unfamiliar/partly covered icon cannot contribute several fake
        nodes to the coordinate lattice.
        """

        labeled = [
            tuple(point)
            for point in cls._deduplicate(labeled_node_points, radius=12.0)
        ]
        small = [item for item in circles if item[2] <= 22.0]
        large = sorted(
            (item for item in circles if item[2] > 22.0),
            key=lambda item: item[2],
            reverse=True,
        )

        # A fixed icon detected by SIFT is stronger than any circle response.
        # The radius is comfortably below the smallest grid step, so a nearby
        # exposed connector remains independent.
        labeled_radius = max(24.0, min(width, height) * 0.045)
        icon_representatives: list[tuple[float, float, float]] = [
            # Store an effective radius so the common 0.78 suppression factor
            # below covers the whole labeled-icon neighborhood.
            (point[0], point[1], labeled_radius / 0.78) for point in labeled
        ]

        unmatched_large = [
            item
            for item in large
            if not any(
                np.hypot(item[0] - point[0], item[1] - point[1])
                <= max(labeled_radius, item[2] * 0.85)
                for point in labeled
            )
        ]
        for candidate in unmatched_large:
            # Candidates are radius-sorted, so the retained response is the
            # outer icon ring rather than a smaller internal circular detail.
            if any(
                np.hypot(candidate[0] - old[0], candidate[1] - old[1])
                <= max(18.0, min(candidate[2], old[2]) * 0.90)
                for old in icon_representatives
            ):
                continue
            icon_representatives.append(candidate)

        retained_small = [
            (item[0], item[1])
            for item in small
            if not any(
                np.hypot(item[0] - icon[0], item[1] - icon[1])
                <= max(18.0, icon[2] * 0.78)
                for icon in icon_representatives
            )
        ]
        points = [(item[0], item[1]) for item in icon_representatives]
        points.extend(retained_small)
        return cls._deduplicate(points, radius=7.0)

    @staticmethod
    def _cluster_axis(values: np.ndarray, tolerance: float) -> list[tuple[float, int]]:
        groups: list[list[float]] = []
        for value in sorted(float(item) for item in values):
            if not groups or value - float(np.median(groups[-1])) > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)
        return [(float(np.median(group)), len(group)) for group in groups]

    @classmethod
    def _exclude_dense_edge_bands(
        cls,
        points: np.ndarray,
        *,
        working_height: float,
        columns: int,
    ) -> tuple[np.ndarray, tuple[float, ...]]:
        """Remove dense top/bottom HUD rows that cannot be map-node rows.

        A real fixed-grid row contains at most ``columns`` nodes. Desktop and
        game HUD strips can contain many unrelated circular controls on one
        horizontal band. Only over-capacity bands near a screen edge are
        removed, so legitimate maps extending close to an edge are retained.
        """

        if len(points) == 0:
            return points, ()
        tolerance = working_height * 0.025
        excluded = tuple(
            center
            for center, weight in cls._cluster_axis(points[:, 1], tolerance)
            if weight > columns + 2
            and (center < working_height * 0.13 or center > working_height * 0.82)
        )
        if not excluded:
            return points, ()
        keep = np.ones(len(points), dtype=bool)
        for center in excluded:
            keep &= np.abs(points[:, 1] - center) > tolerance
        return points[keep], excluded

    @staticmethod
    def _exclude_bands(
        points: np.ndarray,
        centers: tuple[float, ...],
        *,
        tolerance: float,
    ) -> np.ndarray:
        if len(points) == 0 or not centers:
            return points
        keep = np.ones(len(points), dtype=bool)
        for center in centers:
            keep &= np.abs(points[:, 1] - center) > tolerance
        return points[keep]

    @staticmethod
    def _fixed_lattice_candidates(
        bands: list[tuple[float, int]],
        count: int,
        min_step: float,
        max_step: float,
        axis_extent: float,
        *,
        limit: int = 12,
    ) -> list[tuple[np.ndarray, float, float]]:
        """Rank distinct fixed-axis fits instead of committing to one axis early."""

        if len(bands) < 2 or count < 2:
            return []
        candidates: list[tuple[float, np.ndarray, float]] = []
        for left_value, _left_weight in bands:
            for right_value, _right_weight in bands:
                if right_value <= left_value:
                    continue
                for left_index in range(count):
                    for right_index in range(left_index + 1, count):
                        step = (right_value - left_value) / (right_index - left_index)
                        if not min_step <= step <= max_step:
                            continue
                        origin = left_value - left_index * step
                        levels = origin + np.arange(count, dtype=np.float64) * step
                        tolerance = max(7.0, step * 0.16)
                        matched_indices: list[int] = []
                        weight = 0
                        residuals: list[float] = []
                        for level in levels:
                            distances = [abs(center - level) for center, _ in bands]
                            nearest = int(np.argmin(distances))
                            if distances[nearest] <= tolerance:
                                matched_indices.append(nearest)
                                weight += bands[nearest][1]
                                residuals.append(distances[nearest] / tolerance)
                        coverage = len(set(matched_indices)) / count
                        if coverage < 0.55:
                            continue
                        outside = max(0.0, -levels[0]) + max(
                            0.0, levels[-1] - axis_extent
                        )
                        outside_penalty = outside / max(1.0, axis_extent * 0.1)
                        center_penalty = abs(float(np.mean(levels)) / axis_extent - 0.5)
                        score = (
                            2.5 * coverage
                            # A large icon or Windows UI control can produce
                            # several concentric Hough circles at one place.
                            # Treat that multiplicity as weak evidence only;
                            # otherwise a taskbar row can outweigh the four
                            # actual map rows.
                            + 0.002 * weight
                            - 0.12 * (float(np.mean(residuals)) if residuals else 1.0)
                            # The fixed map lattice occupies the central game
                            # canvas.  This also rejects title/taskbar bands
                            # when a retained debug image came from Photos.
                            - 0.10 * center_penalty
                            - 0.20 * outside_penalty
                        )
                        candidates.append((score, levels, step))

        candidates.sort(key=lambda item: item[0], reverse=True)
        distinct: list[tuple[np.ndarray, float, float]] = []
        for score, levels, step in candidates:
            # The same lattice is usually proposed by many different pairs
            # of noisy coordinate bands.  Keep only geometrically distinct
            # alternatives so joint 2-D scoring stays fast.
            if any(
                float(np.max(np.abs(levels - old_levels)))
                <= max(4.0, min(step, old_step) * 0.08)
                for old_levels, old_step, _old_score in distinct
            ):
                continue
            distinct.append((levels, step, score))
            if len(distinct) >= limit:
                break
        return distinct

    @classmethod
    def _fixed_lattice(
        cls,
        bands: list[tuple[float, int]],
        count: int,
        min_step: float,
        max_step: float,
        axis_extent: float,
    ) -> tuple[np.ndarray, float] | None:
        """Fit exactly ``count`` equally spaced grid coordinates to noisy bands."""

        candidates = cls._fixed_lattice_candidates(
            bands, count, min_step, max_step, axis_extent, limit=1
        )
        if not candidates:
            return None
        levels, step, _score = candidates[0]
        return levels, step

    def _fixed_coordinate_frame(
        self,
        points: np.ndarray,
        working_width: float,
        working_height: float,
        columns: int,
        rows: int,
        template_cell_sets: tuple[set[tuple[int, int]], ...] = (),
        supplemental_points: np.ndarray | None = None,
    ) -> tuple[CoordinateFrame, np.ndarray, np.ndarray] | None:
        # Keep nearby HUD circles separate from the first/last map row.  Four
        # percent merged the floor-1 top row with controls only ~50 px away.
        y_bands = self._cluster_axis(points[:, 1], working_height * 0.025)
        y_candidates = self._fixed_lattice_candidates(
            y_bands,
            rows,
            working_height * 0.06,
            working_height * 0.35,
            working_height,
            limit=8,
        )
        if not y_candidates:
            return None
        best: tuple[float, np.ndarray, np.ndarray, float, float] | None = None
        for y_levels, step_y, y_axis_score in y_candidates:
            # A sparse HUD emblem can imitate a complete extra grid row. Live
            # maps never begin in the top six percent of a full game frame.
            if y_levels[0] < working_height * 0.06:
                continue
            row_tolerance = max(8.0, step_y * 0.22)
            on_grid_rows = np.asarray(
                [
                    point
                    for point in points
                    if float(np.min(np.abs(y_levels - point[1]))) <= row_tolerance
                ],
                dtype=np.float64,
            )
            if len(on_grid_rows) < max(4, columns // 2):
                continue
            x_bands = self._cluster_axis(on_grid_rows[:, 0], working_width * 0.028)
            fitted_x = self._fixed_lattice(
                x_bands,
                columns,
                working_width * 0.04,
                working_width * 0.30,
                working_width,
            )
            if fitted_x is None:
                continue
            x_levels, step_x = fitted_x
            occupied = self._occupied_cells(
                points, x_levels, y_levels, step_x, step_y
            )
            occupied_rows = len({row for _column, row in occupied})
            occupied_columns = len({column for column, _row in occupied})
            if occupied_rows < 2 or occupied_columns < 2:
                continue

            # An axis can look convincing by combining one dense HUD row with
            # unrelated map circles.  Real map nodes must agree on both axes,
            # so the number of occupied 2-D intersections is the dominant
            # signal.  Axis scores only break close ties.
            scoring_occupied = occupied
            if supplemental_points is not None and len(supplemental_points):
                nearest_x = np.min(
                    np.abs(supplemental_points[:, None, 0] - x_levels[None, :]),
                    axis=1,
                )
                nearest_y = np.min(
                    np.abs(supplemental_points[:, None, 1] - y_levels[None, :]),
                    axis=1,
                )
                close_to_intersection = np.sqrt(
                    (nearest_x / max(7.0, step_x * 0.18)) ** 2
                    + (nearest_y / max(7.0, step_y * 0.18)) ** 2
                ) <= 1.0
                if bool(np.any(close_to_intersection)):
                    scoring_occupied = self._occupied_cells(
                        np.vstack((points, supplemental_points[close_to_intersection])),
                        x_levels,
                        y_levels,
                        step_x,
                        step_y,
                    )
            template_fit = max(
                (
                    self._occupancy_similarity(scoring_occupied, template_cells)[0]
                    for template_cells in template_cell_sets
                ),
                default=0.0,
            )
            joint_score = (
                template_fit
                + len(occupied) / (columns * rows)
                + 0.18 * occupied_rows / rows
                + 0.18 * occupied_columns / columns
                + 0.025 * y_axis_score
            )
            candidate = (joint_score, x_levels, y_levels, step_x, step_y)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            return None
        _joint_score, x_levels, y_levels, step_x, step_y = best
        frame = CoordinateFrame(
            left=float(x_levels[0]),
            top=float(y_levels[0]),
            right=float(x_levels[-1]),
            bottom=float(y_levels[-1]),
            step_x=float(step_x),
            step_y=float(step_y),
        )
        return frame, x_levels, y_levels

    @staticmethod
    def _occupied_cells(
        points: np.ndarray,
        x_levels: np.ndarray,
        y_levels: np.ndarray,
        step_x: float,
        step_y: float,
    ) -> set[tuple[int, int]]:
        occupied: set[tuple[int, int]] = set()
        for row, y in enumerate(y_levels):
            for column, x in enumerate(x_levels):
                normalized = np.sqrt(
                    ((points[:, 0] - x) / max(1.0, step_x * 0.28)) ** 2
                    + ((points[:, 1] - y) / max(1.0, step_y * 0.28)) ** 2
                )
                if float(np.min(normalized)) <= 1.0:
                    occupied.add((column, row))
        return occupied

    @staticmethod
    def _occupancy_similarity(
        observed_cells: set[tuple[int, int]],
        template_cells: set[tuple[int, int]],
    ) -> tuple[float, int, float, float]:
        intersection = len(observed_cells & template_cells)
        recall = intersection / max(1, len(template_cells))
        precision = intersection / max(1, len(observed_cells))
        count_score = float(
            np.exp(
                -abs(len(observed_cells) - len(template_cells))
                / max(2.0, len(template_cells) * 0.30)
            )
        )
        # A visible node in a template-empty cell is stronger evidence than a
        # template node missed by circle detection.
        score = 0.38 * recall + 0.54 * precision + 0.08 * count_score
        return float(score), intersection, float(recall), float(precision)

    @staticmethod
    def _dominant_coordinate(values: np.ndarray) -> float:
        """Mode of a 1-D coordinate array, binned to whole pixels.

        Per-node circle snapping leaves a straight game row slightly jittery in
        the overlay.  The stable row/column line is the coordinate most detected
        nodes agree on; outliers get pulled onto it.  When no pixel repeats (a
        tiny or noisy group) the median is used instead.
        """

        if values.size == 0:
            return float("nan")
        if values.size == 1:
            return float(values[0])
        rounded = np.rint(values).astype(np.int64)
        unique, counts = np.unique(rounded, return_counts=True)
        modal_count = int(counts.max())
        if modal_count <= 1:
            return float(np.median(values))
        modal_bin = int(unique[int(np.argmax(counts))])
        return float(np.median(values[rounded == modal_bin]))

    @staticmethod
    def _collinearize_positions(
        positions: np.ndarray,
        observed_x: np.ndarray,
        observed_y: np.ndarray,
    ) -> np.ndarray:
        """Force every node onto its row's dominant Y and column's dominant X.

        Final stabilizing pass once the fixed lattice is known.  Each node is
        assigned to its nearest column/row level, then all nodes in one row
        share a single Y and all nodes in one column share a single X.  Overlay
        boxes therefore line up even when individual circle detections jitter.
        """

        if positions.size == 0 or observed_x.size == 0 or observed_y.size == 0:
            return positions
        aligned = positions.copy()
        columns = np.argmin(np.abs(aligned[:, 0:1] - observed_x[None, :]), axis=1)
        rows = np.argmin(np.abs(aligned[:, 1:2] - observed_y[None, :]), axis=1)
        for column in np.unique(columns):
            members = np.where(columns == column)[0]
            if members.size:
                aligned[members, 0] = NodeGridDetector._dominant_coordinate(
                    positions[members, 0]
                )
        for row in np.unique(rows):
            members = np.where(rows == row)[0]
            if members.size:
                aligned[members, 1] = NodeGridDetector._dominant_coordinate(
                    positions[members, 1]
                )
        return aligned

    @staticmethod
    def _template_levels(values: np.ndarray, count: int) -> np.ndarray | None:
        """Recover a template's complete axis, including a wholly empty edge column."""

        levels = fit_template_axis(values, count)
        return np.asarray(levels, dtype=np.float64) if levels is not None else None

    def detect_points(
        self,
        image: np.ndarray,
        extra_points: tuple[tuple[float, float], ...] = (),
    ) -> tuple[np.ndarray, float]:
        original_height, original_width = image.shape[:2]
        resize_scale = min(1.0, self.target_width / original_width)
        if resize_scale < 1.0:
            working = cv2.resize(
                image,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = image
        height, width = working.shape[:2]
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

        raw_circles: list[tuple[float, float, float]] = []
        settings = (
            # Exposed graph rings and small high-contrast details in icons.
            (24, 5, 20, 3, 16),
            # Large semantic node icons.
            (40, 35, 105, 7, 55),
        )
        for threshold, min_radius, max_radius, blur_size, min_distance in settings:
            blurred = cv2.GaussianBlur(
                gray, (blur_size, blur_size), 1 if blur_size > 3 else 0
            )
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                1.2,
                min_distance,
                param1=100,
                param2=threshold,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is None:
                continue
            for x, y, radius in circles[0]:
                # Skip the title/HUD strips, but retain generous margins for
                # asymmetric layouts. Extra circles are tolerated by matching.
                if width * 0.02 < x < width * 0.98 and height * 0.035 < y < height * 0.95:
                    raw_circles.append((float(x), float(y), float(radius)))

        scaled_extra_points = [
            (float(point[0] * resize_scale), float(point[1] * resize_scale))
            for point in extra_points
        ]
        points = self._consolidate_node_circles(
            raw_circles,
            labeled_node_points=scaled_extra_points,
            width=width,
            height=height,
        )

        return points, resize_scale

    def _weak_circle_points(
        self, image: np.ndarray, resize_scale: float
    ) -> np.ndarray:
        """Return low-threshold circles for occupancy refinement only.

        These points are deliberately kept out of coordinate-frame fitting:
        star fields and desktop chrome contain many weak circles.  Once the
        fixed floor lattice is known, however, a weak circle very close to a
        lattice intersection is useful evidence for a dark/covered node.
        """

        if resize_scale < 1.0:
            working = cv2.resize(
                image,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = image
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            1.2,
            20,
            param1=100,
            param2=24,
            minRadius=5,
            maxRadius=55,
        )
        if circles is None:
            return np.empty((0, 2), dtype=np.float64)
        height, width = working.shape[:2]
        return self._deduplicate(
            [
                (float(x), float(y))
                for x, y, _radius in circles[0]
                if width * 0.02 < x < width * 0.98
                and height * 0.035 < y < height * 0.95
            ],
            radius=5.0,
        )

    def inspect(
        self,
        image: np.ndarray,
        grid_size: tuple[int, int] | None = None,
        extra_points: tuple[tuple[float, float], ...] = (),
    ) -> GridInspection:
        """Return full-resolution diagnostic points and the inferred map frame."""

        points, resize_scale = self.detect_points(image, extra_points)
        x_levels = np.asarray([], dtype=np.float64)
        y_levels = np.asarray([], dtype=np.float64)
        occupied: set[tuple[int, int]] = set()
        if grid_size is not None:
            columns, rows = grid_size
            frame_points, _excluded = self._exclude_dense_edge_bands(
                points,
                working_height=image.shape[0] * resize_scale,
                columns=columns,
            )
            fixed = self._fixed_coordinate_frame(
                frame_points,
                image.shape[1] * resize_scale,
                image.shape[0] * resize_scale,
                columns,
                rows,
            )
            if fixed is None:
                frame = None
            else:
                frame, x_levels, y_levels = fixed
                occupied = self._occupied_cells(
                    frame_points, x_levels, y_levels, frame.step_x, frame.step_y
                )
        else:
            columns = rows = None
            frame = None
        original_points = tuple(
            (float(point[0] / resize_scale), float(point[1] / resize_scale))
            for point in points
        )
        if frame is not None:
            frame = CoordinateFrame(
                left=frame.left / resize_scale,
                top=frame.top / resize_scale,
                right=frame.right / resize_scale,
                bottom=frame.bottom / resize_scale,
                step_x=frame.step_x / resize_scale,
                step_y=frame.step_y / resize_scale,
            )
        return GridInspection(
            original_points,
            frame,
            columns,
            rows,
            tuple(float(value / resize_scale) for value in x_levels),
            tuple(float(value / resize_scale) for value in y_levels),
            tuple(sorted(occupied)),
        )

    def floor_occupancy_counts(
        self,
        image: np.ndarray,
        extra_points: tuple[tuple[float, float], ...] = (),
        *,
        detected_points: tuple[np.ndarray, float] | None = None,
    ) -> dict[int, int]:
        """Count node intersections explained by every fixed floor grid."""

        if detected_points is None:
            points, resize_scale = self.detect_points(image, extra_points)
        else:
            points, resize_scale = detected_points
        if len(points) < 5 or len(points) > 240:
            return {}
        working_width = image.shape[1] * resize_scale
        working_height = image.shape[0] * resize_scale
        counts: dict[int, int] = {}
        for floor, (columns, rows) in GRID_SHAPES.items():
            frame_points, _excluded = self._exclude_dense_edge_bands(
                points, working_height=working_height, columns=columns
            )
            fixed = self._fixed_coordinate_frame(
                frame_points,
                working_width,
                working_height,
                columns,
                rows,
            )
            if fixed is None:
                continue
            frame, x_levels, y_levels = fixed
            occupied = self._occupied_cells(
                frame_points, x_levels, y_levels, frame.step_x, frame.step_y
            )
            if len(occupied) < max(3, round(columns * rows * 0.18)):
                continue
            counts[floor] = len(occupied)
        return counts

    def _grid_first_matches(
        self,
        image: np.ndarray,
        templates: list[MapTemplate],
        points: np.ndarray,
        resize_scale: float,
        weak_points: np.ndarray | None = None,
    ) -> list[StructuralMatch]:
        """Identify templates from fixed-size node occupancy matrices first."""

        working_height = image.shape[0] * resize_scale
        working_width = image.shape[1] * resize_scale
        results: list[StructuralMatch] = []
        if weak_points is None:
            weak_points = self._weak_circle_points(image, resize_scale)

        for floor in sorted({template.floor for template in templates}):
            grid_size = GRID_SHAPES.get(floor)
            if grid_size is None:
                continue
            columns, rows = grid_size
            frame_points, excluded_bands = self._exclude_dense_edge_bands(
                points, working_height=working_height, columns=columns
            )
            floor_weak_points = self._exclude_bands(
                weak_points,
                excluded_bands,
                tolerance=working_height * 0.025,
            )
            template_cell_sets: list[set[tuple[int, int]]] = []
            for template in templates:
                if template.floor != floor:
                    continue
                geometry = self._geometry(template)
                if geometry is None:
                    continue
                template_cell_sets.append(set(geometry.cells))
            fixed = self._fixed_coordinate_frame(
                frame_points,
                working_width,
                working_height,
                columns,
                rows,
                tuple(template_cell_sets),
                floor_weak_points,
            )
            if fixed is None:
                continue
            frame, observed_x, observed_y = fixed
            occupancy_points = frame_points
            if len(floor_weak_points):
                nearest_x = np.min(
                    np.abs(floor_weak_points[:, None, 0] - observed_x[None, :]), axis=1
                )
                nearest_y = np.min(
                    np.abs(floor_weak_points[:, None, 1] - observed_y[None, :]), axis=1
                )
                close_to_intersection = np.sqrt(
                    (nearest_x / max(7.0, frame.step_x * 0.18)) ** 2
                    + (nearest_y / max(7.0, frame.step_y * 0.18)) ** 2
                ) <= 1.0
                if bool(np.any(close_to_intersection)):
                    occupancy_points = np.vstack(
                        (frame_points, floor_weak_points[close_to_intersection])
                    )
            observed_cells = self._occupied_cells(
                occupancy_points,
                observed_x,
                observed_y,
                frame.step_x,
                frame.step_y,
            )
            if len(observed_cells) < max(3, round(columns * rows * 0.18)):
                continue

            original_frame = CoordinateFrame(
                left=frame.left / resize_scale,
                top=frame.top / resize_scale,
                right=frame.right / resize_scale,
                bottom=frame.bottom / resize_scale,
                step_x=frame.step_x / resize_scale,
                step_y=frame.step_y / resize_scale,
            )
            inspection = GridInspection(
                points=tuple(
                    (float(point[0] / resize_scale), float(point[1] / resize_scale))
                    for point in points
                ),
                frame=original_frame,
                columns=columns,
                rows=rows,
                x_levels=tuple(float(value / resize_scale) for value in observed_x),
                y_levels=tuple(float(value / resize_scale) for value in observed_y),
                occupied_cells=tuple(sorted(observed_cells)),
            )

            for template in templates:
                if template.floor != floor or not template.image_width or not template.image_height:
                    continue
                geometry = self._geometry(template)
                if geometry is None:
                    continue
                score, intersection, recall, _precision = self._occupancy_similarity(
                    observed_cells, set(geometry.cells)
                )

                scale_x = (observed_x[-1] - observed_x[0]) / (
                    geometry.source_x[-1] - geometry.source_x[0]
                )
                scale_y = (observed_y[-1] - observed_y[0]) / (
                    geometry.source_y[-1] - geometry.source_y[0]
                )
                translate_x = observed_x[0] - geometry.source_x[0] * scale_x
                translate_y = observed_y[0] - geometry.source_y[0] * scale_y
                projected = np.column_stack(
                    (
                        geometry.source_points[:, 0] * scale_x + translate_x,
                        geometry.source_points[:, 1] * scale_y + translate_y,
                    )
                )

                # The fitted lattice supplies stable global geometry. Snap
                # every template node to a nearby detected node circle; the
                # live actor position is intentionally not consulted.
                dx = np.abs(projected[:, None, 0] - occupancy_points[None, :, 0])
                dy = np.abs(projected[:, None, 1] - occupancy_points[None, :, 1])
                normalized = np.sqrt(
                    (dx / max(9.0, frame.step_x * 0.18)) ** 2
                    + (dy / max(9.0, frame.step_y * 0.18)) ** 2
                )
                nearest_indices = np.argmin(normalized, axis=1)
                safe_to_snap = normalized[
                    np.arange(len(template.nodes)), nearest_indices
                ] <= 1.0
                snapped = projected.copy()
                snapped[safe_to_snap] = occupancy_points[
                    nearest_indices[safe_to_snap]
                ]
                # Final stabilizing pass: pull every node onto its row's dominant
                # Y and column's dominant X so transient circle-detection jitter
                # does not skew the overlay boxes.
                snapped = self._collinearize_positions(snapped, observed_x, observed_y)

                homography = np.asarray(
                    [
                        [scale_x / resize_scale, 0.0, translate_x / resize_scale],
                        [0.0, scale_y / resize_scale, translate_y / resize_scale],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                node_positions = {
                    node.id: (
                        float(point[0] / resize_scale),
                        float(point[1] / resize_scale),
                    )
                    for node, point in zip(template.nodes, snapped)
                }
                results.append(
                    StructuralMatch(
                        template=template,
                        score=float(score),
                        homography=homography,
                        node_positions=node_positions,
                        matched_nodes=intersection,
                        coverage=recall,
                        scale_x=float(scale_x / resize_scale),
                        scale_y=float(scale_y / resize_scale),
                        method="grid-first",
                        inspection=inspection,
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)

    def match(
        self,
        image: np.ndarray,
        templates: list[MapTemplate],
        extra_points: tuple[tuple[float, float], ...] = (),
        *,
        detected_points: tuple[np.ndarray, float] | None = None,
        weak_points: np.ndarray | None = None,
    ) -> list[StructuralMatch]:
        if detected_points is None:
            points, resize_scale = self.detect_points(image, extra_points)
        else:
            points, resize_scale = detected_points
        # Dense noise can make Hough return a circle almost everywhere, which
        # would trivially cover every template node. Real maps remain sparse.
        if len(points) < 5 or len(points) > 240:
            return []
        return self._grid_first_matches(
            image,
            templates,
            points,
            resize_scale,
            weak_points,
        )
