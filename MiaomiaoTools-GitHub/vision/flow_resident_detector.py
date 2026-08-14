from __future__ import annotations

import cv2
import numpy as np


class FlowResidentDetector:
    """Detect the small white-ring snake marker above an empty map node."""

    @staticmethod
    def detect(
        image: np.ndarray,
        center: tuple[float, float],
        grid_step: float,
    ) -> bool:
        node_x, node_y = round(center[0]), round(center[1])
        left, right = max(0, node_x - round(grid_step * 0.46)), min(
            image.shape[1], node_x + round(grid_step * 0.58)
        )
        top, bottom = max(0, node_y - round(grid_step * 0.55)), min(
            image.shape[0], node_y + round(grid_step * 0.18)
        )
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            cv2.medianBlur(gray, 5),
            cv2.HOUGH_GRADIENT,
            dp=1.15,
            minDist=max(15, round(grid_step * 0.10)),
            param1=100,
            param2=18,
            minRadius=max(18, round(grid_step * 0.09)),
            maxRadius=max(30, round(grid_step * 0.20)),
        )
        if circles is None:
            return False
        yy, xx = np.ogrid[: crop.shape[0], : crop.shape[1]]
        for circle_x, circle_y, circle_radius in circles[0]:
            offset_x = circle_x + left - node_x
            offset_y = circle_y + top - node_y
            if not (
                grid_step * 0.08 <= offset_x <= grid_step * 0.40
                and -grid_step * 0.38 <= offset_y <= -grid_step * 0.04
            ):
                continue
            distance = np.hypot(xx - circle_x, yy - circle_y)
            ring = (distance >= circle_radius * 0.82) & (
                distance <= circle_radius * 1.10
            )
            inner = distance <= circle_radius * 0.56
            if not bool(np.any(ring)) or not bool(np.any(inner)):
                continue
            if (
                float(np.mean(gray[ring])) >= 110
                and float(np.mean(gray[inner] < 70)) >= 0.38
                and float(np.mean(gray[inner])) <= 115
            ):
                return True
        return False
