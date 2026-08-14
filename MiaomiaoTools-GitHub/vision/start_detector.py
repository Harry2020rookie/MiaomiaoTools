from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .image_io import read_image


@dataclass(frozen=True, slots=True)
class StartDetection:
    """Detected graph start node.

    ``center`` is the black circular graph node beside the YOU ARE HERE actor,
    not the actor itself.  Keeping the two positions separate matters because
    the live game uses a different actor animation from the resource images.
    """

    center: tuple[float, float]
    confidence: float
    inliers: int
    visual_center: tuple[float, float] | None = None
    method: str = "sift"


class StartDetector:
    """Detect the fixed YOU ARE HERE artwork with a resource-derived SIFT template."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        config_path = self.project_root / "data/generated/start_marker.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = read_image(self.project_root / config["source_image"])
        if source is None:
            raise ValueError(f"无法读取出生点模板源图: {config['source_image']}")
        x0, y0, x1, y1 = map(int, config["crop_xyxy"])
        self.anchor = tuple(float(value) for value in config["anchor_in_crop"])
        self.template = source[y0:y1, x0:x1]
        hsv = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
        mask = (
            (hsv[:, :, 2] > 150)
            & ((hsv[:, :, 1] < 90) | ((hsv[:, :, 0] > 10) & (hsv[:, :, 0] < 45)))
        ).astype(np.uint8) * 255
        self.mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        self.sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.01)
        self.template_keypoints, self.template_descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY), self.mask
        )
        if self.template_descriptors is None or len(self.template_keypoints) < 4:
            raise ValueError("出生点视觉模板没有足够特征")

    def _sift_visual_center(
        self, image: np.ndarray
    ) -> tuple[tuple[float, float], float, int] | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            return None
        pairs = cv2.BFMatcher().knnMatch(self.template_descriptors, descriptors, k=2)
        good = [first for first, second in pairs if first.distance < 0.78 * second.distance]
        if len(good) < 6:
            return None
        source_points = np.float32(
            [self.template_keypoints[match.queryIdx].pt for match in good]
        ).reshape(-1, 1, 2)
        destination_points = np.float32([keypoints[match.trainIdx].pt for match in good]).reshape(
            -1, 1, 2
        )
        homography, inlier_mask = cv2.findHomography(
            source_points, destination_points, cv2.RANSAC, 5.0
        )
        if homography is None or inlier_mask is None:
            return None
        inliers = int(inlier_mask.sum())
        if inliers < 6:
            return None
        anchor = cv2.perspectiveTransform(np.float32([[self.anchor]]), homography)[0, 0]
        if not (0 <= anchor[0] < image.shape[1] and 0 <= anchor[1] < image.shape[0]):
            return None
        ratio = inliers / max(1, len(good))
        confidence = min(1.0, 0.55 * ratio + 0.45 * min(1.0, inliers / 18.0))
        return (float(anchor[0]), float(anchor[1])), confidence, inliers

    @staticmethod
    def _bright_actor_center(image: np.ndarray) -> tuple[tuple[float, float], float] | None:
        """Find the tall, low-saturation YOU ARE HERE actor in live screenshots."""

        height, width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 2] > 190) & (hsv[:, :, 1] < 95)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        choices: list[tuple[float, tuple[float, float], float]] = []
        for index in range(1, count):
            x, y, component_width, component_height, area = stats[index]
            aspect = component_height / max(1.0, component_width)
            if not (
                # Bottom HUD/taskbar space is not part of the map and must not
                # raise the actor threshold enough to reject a valid marker.
                area >= max(35, height * width * 0.00035)
                and component_height >= height * 0.042
                and component_height <= height * 0.30
                and component_width <= width * 0.10
                and aspect >= 1.62
            ):
                continue
            center_x, center_y = map(float, centroids[index])
            if not (width * 0.08 <= center_x <= width * 0.92):
                continue
            if not (height * 0.12 <= center_y <= height * 0.92):
                continue
            fill = area / max(1.0, component_width * component_height)
            score = float(area) * min(3.2, aspect) * (0.65 + min(0.55, fill))
            confidence = min(0.91, 0.58 + 0.12 * (aspect - 1.62) + 0.15 * min(1.0, fill))
            choices.append((score, (center_x, center_y), confidence))
        if not choices:
            return None
        _score, center, confidence = max(choices, key=lambda item: item[0])
        return center, confidence

    @staticmethod
    def _graph_node_beside_actor(
        image: np.ndarray, visual_center: tuple[float, float]
    ) -> tuple[float, float] | None:
        """Locate the dark ring immediately to the right of the start actor."""

        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x0 = max(0, round(visual_center[0] + width * 0.006))
        x1 = min(width, round(visual_center[0] + width * 0.085))
        y0 = max(0, round(visual_center[1] - height * 0.055))
        y1 = min(height, round(visual_center[1] + height * 0.065))
        if x1 - x0 < 12 or y1 - y0 < 12:
            return None
        roi = cv2.GaussianBlur(gray[y0:y1, x0:x1], (5, 5), 1)
        min_radius = max(4, round(width * 0.0038))
        max_radius = max(min_radius + 3, round(width * 0.019))
        circles = cv2.HoughCircles(
            roi,
            cv2.HOUGH_GRADIENT,
            1,
            max(8, min_radius * 2),
            param1=95,
            param2=11,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return None
        candidates: list[tuple[float, tuple[float, float]]] = []
        for local_x, local_y, radius in circles[0]:
            center_x = float(x0 + local_x)
            center_y = float(y0 + local_y)
            dx = center_x - visual_center[0]
            dy = center_y - visual_center[1]
            if not (width * 0.009 <= dx <= width * 0.08 and abs(dy) <= height * 0.052):
                continue
            radius_i = max(2, round(radius * 0.48))
            cx, cy = round(center_x), round(center_y)
            patch = gray[
                max(0, cy - radius_i) : min(height, cy + radius_i + 1),
                max(0, cx - radius_i) : min(width, cx + radius_i + 1),
            ]
            darkness = 1.0 - float(np.mean(patch)) / 255.0 if patch.size else 0.0
            distance_penalty = abs(dx / width - 0.036) * 13.0
            vertical_penalty = abs(dy) / max(1.0, height * 0.05)
            radius_penalty = abs(float(radius) / width - 0.009) * 80.0
            score = (
                1.55 * darkness
                - distance_penalty
                - 0.35 * vertical_penalty
                - radius_penalty
            )
            candidates.append((score, (center_x, center_y)))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    @classmethod
    def detect_live_graph_center(
        cls, image: np.ndarray
    ) -> tuple[tuple[float, float], float] | None:
        """Locate a live start node without loading the resource SIFT template."""

        actor_hit = cls._bright_actor_center(image)
        if actor_hit is None:
            return None
        visual_center, confidence = actor_hit
        if confidence < 0.65:
            return None
        graph_center = cls._graph_node_beside_actor(image, visual_center)
        if graph_center is None:
            return None
        return graph_center, float(confidence)

    def detect(self, image: np.ndarray) -> StartDetection | None:
        sift_hit = self._sift_visual_center(image)
        if sift_hit is not None:
            visual_center, confidence, inliers = sift_hit
            method = "sift+ring"
        else:
            actor_hit = self._bright_actor_center(image)
            if actor_hit is None:
                return None
            visual_center, confidence = actor_hit
            inliers = 0
            method = "bright-actor+ring"

        graph_center = self._graph_node_beside_actor(image, visual_center)
        if graph_center is None:
            # A visual-only position is useful for diagnostics, but is not safe
            # as a graph anchor for template matching.
            return None
        return StartDetection(
            center=graph_center,
            confidence=confidence,
            inliers=inliers,
            visual_center=visual_center,
            method=method,
        )
