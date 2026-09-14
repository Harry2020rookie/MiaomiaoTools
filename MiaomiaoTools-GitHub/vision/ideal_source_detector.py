from __future__ import annotations

import cv2
import numpy as np


class IdealSourceDetector:
    """Detect the radial mesh surrounding an ideal-source node."""

    @staticmethod
    def has_purple_combat_icon(image, center, grid_step) -> bool:
        x, y = map(round, center)
        radius = max(5, round(grid_step * .23))
        crop = image[max(0, y-radius):min(image.shape[0], y+radius),
                     max(0, x-radius):min(image.shape[1], x+radius)]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        purple = ((hsv[:, :, 0] > 120) & (hsv[:, :, 0] < 170)
                  & (hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 25))
        return float(np.mean(purple)) >= .07

    @staticmethod
    def detect(
        image: np.ndarray,
        center: tuple[float, float],
        grid_step: float,
    ) -> bool:
        radius = max(80, round(grid_step * 0.76))
        center_x, center_y = (round(center[0]), round(center[1]))
        height, width = image.shape[:2]
        left, right = max(0, center_x - radius), min(width, center_x + radius + 1)
        top, bottom = max(0, center_y - radius), min(height, center_y + radius + 1)
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return False

        local_x, local_y = center_x - left, center_y - top
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 110) > 0
        yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
        dx, dy = xx - local_x, yy - local_y
        distance = np.hypot(dx, dy)
        inner, outer = grid_step * 0.34, grid_step * 0.72
        # Ignore the node icon and the cardinal route corridors. The ideal
        # source mesh is distinguished by diagonal branches in many sectors.
        annulus = (
            (distance >= inner)
            & (distance <= outer)
            & (np.abs(dx) > grid_step * 0.10)
            & (np.abs(dy) > grid_step * 0.10)
        )
        if not bool(np.any(annulus)):
            return False

        edge_density = float(np.mean(edges[annulus]))
        angle = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)
        active_sectors = 0
        for index in range(12):
            sector = annulus & (angle >= index * np.pi / 6) & (
                angle < (index + 1) * np.pi / 6
            )
            if bool(np.any(sector)) and float(np.mean(edges[sector])) >= 0.025:
                active_sectors += 1
        return edge_density >= 0.038 and active_sectors >= 7
