from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np


GRID_SHAPES: dict[int, tuple[int, int]] = {
    1: (5, 3),
    2: (5, 4),
    3: (7, 5),
    4: (8, 5),
    5: (10, 5),
}


class GridGeometryError(ValueError):
    """Raised when normalized template points cannot be placed on their grid."""


def fit_template_axis(values: Iterable[float], count: int) -> tuple[float, ...] | None:
    """Recover a template's complete normalized axis.

    Some maps do not occupy every row or column.  Trying to rank only the
    coordinates that happen to contain nodes would collapse such a gap and
    collapse an empty grid gap, so this fits exactly ``count`` evenly
    spaced levels to the normalized template coordinates.
    """

    raw_values = tuple(float(value) for value in values)
    return _fit_template_axis_cached(raw_values, count)


@lru_cache(maxsize=512)
def _fit_template_axis_cached(
    raw_values: tuple[float, ...], count: int
) -> tuple[float, ...] | None:
    """Cached implementation shared by validation, matching and distance lookup."""

    unique = tuple(sorted({round(value, 5) for value in raw_values}))
    if len(unique) < 2 or count < 2:
        return None

    candidates: list[tuple[float, float]] = []
    evaluated: set[tuple[float, float]] = set()
    for left_value in unique:
        for right_value in unique:
            if right_value <= left_value:
                continue
            for left_index in range(count):
                for right_index in range(left_index + 1, count):
                    step = (right_value - left_value) / (right_index - left_index)
                    if not 0.04 <= step <= 0.40:
                        continue
                    origin = left_value - left_index * step
                    geometry_key = (round(origin, 12), round(step, 12))
                    if geometry_key in evaluated:
                        continue
                    evaluated.add(geometry_key)
                    candidates.append((origin, step))
    if not candidates:
        return None

    raw = np.asarray(raw_values, dtype=np.float64)
    level_indices = np.arange(count, dtype=np.float64)
    best_score = -1.0
    best_levels: np.ndarray | None = None
    # Vectorize the expensive residual scoring while batching to keep memory
    # bounded for large templates. Candidate order is retained, so tie
    # behavior stays identical to the original exhaustive implementation.
    for start in range(0, len(candidates), 2048):
        batch = np.asarray(candidates[start : start + 2048], dtype=np.float64)
        origins = batch[:, 0]
        steps = batch[:, 1]
        levels = origins[:, None] + steps[:, None] * level_indices[None, :]
        distances = np.min(
            np.abs(raw[None, :, None] - levels[:, None, :]), axis=2
        ) / steps[:, None]
        support = np.mean(distances <= 0.28, axis=1)
        quality = np.mean(np.clip(1.0 - distances / 0.35, 0.0, 1.0), axis=1)
        center_score = np.exp(-(((np.mean(levels, axis=1) - 0.5) / 0.28) ** 2))
        outside = np.maximum(0.0, -levels[:, 0]) + np.maximum(
            0.0, levels[:, -1] - 1.0
        )
        bounds_score = np.exp(-((outside / 0.15) ** 2))
        scores = (
            0.62 * support
            + 0.25 * quality
            + 0.08 * center_score
            + 0.05 * bounds_score
        )
        best_index = int(np.argmax(scores))
        score = float(scores[best_index])
        if score > best_score:
            best_score = score
            best_levels = levels[best_index].copy()
    return (
        tuple(float(value) for value in best_levels)
        if best_levels is not None
        else None
    )


def locate_template_points(
    points: Iterable[tuple[str, float, float]],
    *,
    columns: int,
    rows: int,
) -> dict[str, tuple[int, int]]:
    """Map normalized template points to zero-based fixed-grid cells."""

    point_list = [(str(node_id), float(x), float(y)) for node_id, x, y in points]
    x_levels = fit_template_axis((x for _node_id, x, _y in point_list), columns)
    y_levels = fit_template_axis((y for _node_id, _x, y in point_list), rows)
    if x_levels is None or y_levels is None:
        raise GridGeometryError(
            f"无法拟合 {columns}×{rows} 固定网格：模板坐标层级不足"
        )

    step_x = abs(x_levels[1] - x_levels[0])
    step_y = abs(y_levels[1] - y_levels[0])
    cells: dict[str, tuple[int, int]] = {}
    occupied: dict[tuple[int, int], str] = {}
    for node_id, x, y in point_list:
        column = min(range(columns), key=lambda index: abs(x - x_levels[index]))
        row = min(range(rows), key=lambda index: abs(y - y_levels[index]))
        residual_x = abs(x - x_levels[column]) / step_x
        residual_y = abs(y - y_levels[row]) / step_y
        if residual_x > 0.28 or residual_y > 0.28:
            raise GridGeometryError(
                f"节点 {node_id} 偏离固定网格过大："
                f"列误差 {residual_x:.3f}，行误差 {residual_y:.3f}"
            )
        cell = (column, row)
        if cell in occupied:
            raise GridGeometryError(
                f"节点 {node_id} 与 {occupied[cell]} 重叠在网格 {cell}"
            )
        occupied[cell] = node_id
        cells[node_id] = cell
    return cells
