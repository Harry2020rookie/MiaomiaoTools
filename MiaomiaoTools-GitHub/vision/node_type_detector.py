from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image_io import read_image


@dataclass(frozen=True, slots=True)
class NodeTypeDetection:
    node_id: str
    node_type: str
    icon_name: str
    confidence: float
    inliers: int
    inlier_ratio: float
    from_text: bool = False


@dataclass(frozen=True, slots=True)
class _IconTemplate:
    icon_name: str
    node_type: str
    width: int
    height: int
    keypoints: tuple
    descriptors: np.ndarray


@dataclass(frozen=True, slots=True)
class _TextTemplate:
    icon_name: str
    node_type: str
    width: int
    height: int
    keypoints: tuple
    descriptors: np.ndarray


class NodeTypeDetector:
    """Classify visible named nodes from the fixed icon assets.

    Empty connector dots deliberately have no matching template and therefore
    do not appear in the returned mapping.
    """

    def __init__(self, project_root: str | Path) -> None:
        root = Path(project_root)
        mapping_path = root / "data/rules/icon_node_types.json"
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"无法读取节点图标分类配置 {mapping_path}: {exc}") from exc
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("节点图标分类配置必须是非空对象")

        self.sift = cv2.SIFT_create(nfeatures=4500, contrastThreshold=0.01)
        templates: list[_IconTemplate] = []
        text_templates: list[_TextTemplate] = []
        self.gain_loss_appearance: tuple[np.ndarray, np.ndarray] | None = None
        text_font = self._load_text_font()
        for icon_name, node_type in mapping.items():
            if not isinstance(icon_name, str) or not isinstance(node_type, str):
                raise ValueError("节点图标分类配置的文件名和类型必须是字符串")
            source = read_image(root / "data/icons" / icon_name, cv2.IMREAD_UNCHANGED)
            if source is None or source.ndim != 3:
                raise ValueError(f"无法读取带透明通道的节点图标: {icon_name}")
            if source.shape[2] == 4:
                alpha_mask = (source[:, :, 3] > 24).astype(np.uint8) * 255
                source_bgr = source[:, :, :3]
            elif source.shape[2] == 3 and icon_name == "flow-resident.png":
                # User-provided resident samples are centered live-game crops
                # without transparency. They are still useful SIFT evidence;
                # the dedicated geometry detector remains the final authority.
                alpha_mask = np.full(source.shape[:2], 255, dtype=np.uint8)
                source_bgr = source
            else:
                raise ValueError(f"无法读取带透明通道的节点图标: {icon_name}")
            if icon_name == "gain-loss.png":
                y_values, x_values = np.where(alpha_mask > 0)
                if len(x_values) and len(y_values):
                    x0, x1 = int(x_values.min()), int(x_values.max()) + 1
                    y0, y1 = int(y_values.min()), int(y_values.max()) + 1
                    self.gain_loss_appearance = (
                        cv2.cvtColor(source_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY),
                        alpha_mask[y0:y1, x0:x1],
                    )
            keypoints, descriptors = self.sift.detectAndCompute(
                cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY), alpha_mask
            )
            if descriptors is None or len(keypoints) < 6:
                raise ValueError(f"节点图标没有足够特征: {icon_name}")
            templates.append(
                _IconTemplate(
                    icon_name=icon_name,
                    node_type=node_type,
                    width=int(source.shape[1]),
                    height=int(source.shape[0]),
                    keypoints=tuple(keypoints),
                    descriptors=descriptors,
                )
            )
            if text_font is not None:
                # Rule names may carry explanatory suffixes which are not
                # printed below the live node (for example ``险路恶敌``).
                label = node_type.split("（", 1)[0]
                text_source = self._render_text_template(label, text_font)
                text_keypoints, text_descriptors = self.sift.detectAndCompute(
                    text_source, text_source
                )
                if text_descriptors is not None and len(text_keypoints) >= 6:
                    text_templates.append(
                        _TextTemplate(
                            icon_name=icon_name,
                            node_type=node_type,
                            width=int(text_source.shape[1]),
                            height=int(text_source.shape[0]),
                            keypoints=tuple(text_keypoints),
                            descriptors=text_descriptors,
                        )
                    )
        self.templates = tuple(templates)
        self.text_templates = tuple(text_templates)
        self.matcher = cv2.BFMatcher()
        self._cache_image: np.ndarray | None = None
        self._crop_cache: dict[
            tuple[int, int, int, int], NodeTypeDetection | None
        ] = {}
        self._text_crop_cache: dict[
            tuple[int, int, int, int], NodeTypeDetection | None
        ] = {}

    @staticmethod
    def _load_text_font() -> ImageFont.FreeTypeFont | None:
        """Load an available CJK font for constrained node-label recognition."""

        windows_root = Path(os.environ.get("WINDIR", r"C:\Windows"))
        candidates = [windows_root / "Fonts" / name for name in
                      ("msyh.ttc", "msyhl.ttc", "simhei.ttf", "simsun.ttc")]
        candidates.extend(Path(name) for name in (
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ))
        for path in candidates:
            try:
                return ImageFont.truetype(str(path), 64)
            except OSError:
                continue
        return None

    @staticmethod
    def _render_text_template(
        label: str, font: ImageFont.FreeTypeFont
    ) -> np.ndarray:
        box = font.getbbox(label, stroke_width=1)
        image = Image.new(
            "L",
            (max(1, box[2] - box[0] + 12), max(1, box[3] - box[1] + 12)),
        )
        draw = ImageDraw.Draw(image)
        draw.text(
            (6 - box[0], 6 - box[1]),
            label,
            font=font,
            fill=255,
            stroke_width=1,
        )
        return np.asarray(image)

    def _gain_loss_fallback(self, crop: np.ndarray) -> NodeTypeDetection | None:
        """Recognize the dim live gain/loss icon when local SIFT is unstable."""

        if self.gain_loss_appearance is None or crop.size == 0:
            return None
        template_gray, template_mask = self.gain_loss_appearance
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        extent = min(crop_gray.shape)
        best_score = 0.0
        best_scale = 0.0
        best_center_error = 1.0
        for fraction in np.arange(0.48, 0.77, 0.03):
            size = round(extent * float(fraction))
            scale = size / max(template_gray.shape)
            width = max(8, round(template_gray.shape[1] * scale))
            height = max(8, round(template_gray.shape[0] * scale))
            if width > crop_gray.shape[1] or height > crop_gray.shape[0]:
                continue
            resized = cv2.resize(
                template_gray, (width, height), interpolation=cv2.INTER_AREA
            )
            mask = cv2.resize(
                template_mask, (width, height), interpolation=cv2.INTER_NEAREST
            )
            scores = cv2.matchTemplate(
                crop_gray, resized, cv2.TM_CCORR_NORMED, mask=mask
            )
            _minimum, score, _minimum_point, location = cv2.minMaxLoc(scores)
            center = (
                location[0] + width / 2.0,
                location[1] + height / 2.0,
            )
            center_error = float(
                np.linalg.norm(
                    np.subtract(
                        center,
                        (crop_gray.shape[1] / 2.0, crop_gray.shape[0] / 2.0),
                    )
                )
                / max(1, extent)
            )
            if score > best_score:
                best_score = float(score)
                best_scale = float(fraction)
                best_center_error = center_error
        if best_score < 0.90 or best_scale < 0.54 or best_center_error > 0.15:
            return None
        return NodeTypeDetection(
            node_id="",
            node_type="失与得",
            icon_name="gain-loss.png",
            confidence=float(min(0.90, 0.70 + (best_score - 0.90) * 3.0)),
            inliers=0,
            inlier_ratio=0.0,
        )

    def _begin_image(self, image: np.ndarray) -> None:
        if image is self._cache_image:
            return
        self._cache_image = image
        self._crop_cache.clear()
        self._text_crop_cache.clear()

    def global_detections(
        self, image: np.ndarray
    ) -> list[tuple[tuple[float, float], str, float]]:
        """Find visible fixed node icons before a map grid has been fitted."""

        self._begin_image(image)
        screen_keypoints, screen_descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None
        )
        if screen_descriptors is None:
            return []
        found: list[tuple[tuple[float, float], str, float, int]] = []
        for template in self.templates:
            pairs = self.matcher.knnMatch(
                template.descriptors, screen_descriptors, k=2
            )
            remaining = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance < 0.78 * pair[1].distance
            ]
            for _ in range(18):
                if len(remaining) < 6:
                    break
                source = np.float32(
                    [template.keypoints[item.queryIdx].pt for item in remaining]
                ).reshape(-1, 1, 2)
                destination = np.float32(
                    [screen_keypoints[item.trainIdx].pt for item in remaining]
                ).reshape(-1, 1, 2)
                homography, mask = cv2.findHomography(
                    source, destination, cv2.RANSAC, 4.0
                )
                if homography is None or mask is None:
                    break
                inliers = int(mask.sum())
                if inliers < 6:
                    break
                center = cv2.perspectiveTransform(
                    np.float32(
                        [[[template.width / 2.0, template.height / 2.0]]]
                    ),
                    homography,
                )[0, 0]
                corners = cv2.perspectiveTransform(
                    np.float32(
                        [[
                            [0, 0],
                            [template.width - 1, 0],
                            [template.width - 1, template.height - 1],
                            [0, template.height - 1],
                        ]]
                    ),
                    homography,
                )[0]
                projected_width = float(
                    (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3]))
                    / 2.0
                )
                projected_height = float(
                    (np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1]))
                    / 2.0
                )
                ratio = inliers / len(remaining)
                if (
                    0 <= center[0] < image.shape[1]
                    and 0 <= center[1] < image.shape[0]
                    and 22 <= projected_width <= image.shape[1] * 0.22
                    and 22 <= projected_height <= image.shape[0] * 0.32
                    and 0.40 <= projected_width / max(1.0, projected_height) <= 2.5
                ):
                    confidence = min(1.0, 0.48 + inliers / 28.0) * min(
                        1.0, ratio / 0.72
                    )
                    found.append(
                        (
                            (float(center[0]), float(center[1])),
                            template.node_type,
                            float(confidence),
                            inliers,
                        )
                    )
                remaining = [
                    item for item, keep in zip(remaining, mask.ravel()) if not keep
                ]

        deduplicated: list[tuple[tuple[float, float], str, float, int]] = []
        for item in sorted(found, key=lambda row: (row[3], row[2]), reverse=True):
            center = item[0]
            if any(
                np.linalg.norm(np.subtract(center, old[0])) < 28.0
                for old in deduplicated
            ):
                continue
            deduplicated.append(item)
        return [(center, node_type, confidence) for center, node_type, confidence, _ in deduplicated]

    @staticmethod
    def detection_regions(
        image_shape: tuple[int, ...],
        node_positions: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[int, int, int, int]]:
        if not node_positions:
            return {}
        centers = np.asarray(list(node_positions.values()), dtype=np.float64)
        if len(centers) >= 2:
            distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            distances[distances == 0] = np.inf
            spacing = float(np.median(np.min(distances, axis=1)))
        else:
            spacing = 180.0
        # Keep this aligned with MysteryDetector's icon ROI.  Besides making
        # the debug crops directly reproducible, the slightly wider box keeps
        # all three circles of the highlighted overlook marker in frame.
        radius = int(round(min(190.0, max(65.0, spacing * 0.39))))
        height, width = image_shape[:2]
        return {
            node_id: (
                max(0, round(center_x - radius)),
                max(0, round(center_y - radius)),
                min(width, round(center_x + radius)),
                min(height, round(center_y + radius)),
            )
            for node_id, (center_x, center_y) in node_positions.items()
        }

    @staticmethod
    def text_detection_regions(
        image_shape: tuple[int, ...],
        node_positions: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Return tight ROIs containing only the label below each node."""

        if not node_positions:
            return {}
        centers = np.asarray(list(node_positions.values()), dtype=np.float64)
        if len(centers) >= 2:
            distances = np.linalg.norm(
                centers[:, None, :] - centers[None, :, :], axis=2
            )
            distances[distances == 0] = np.inf
            spacing = float(np.median(np.min(distances, axis=1)))
        else:
            spacing = 180.0
        radius_x = min(155.0, max(62.0, spacing * 0.45))
        offset_top = min(42.0, max(12.0, spacing * 0.10))
        offset_bottom = min(88.0, max(48.0, spacing * 0.42))
        height, width = image_shape[:2]
        return {
            node_id: (
                max(0, round(center_x - radius_x)),
                max(0, round(center_y + offset_top)),
                min(width, round(center_x + radius_x)),
                min(height, round(center_y + offset_bottom)),
            )
            for node_id, (center_x, center_y) in node_positions.items()
        }

    @staticmethod
    def danger_enemy_text_regions(
        image_shape: tuple[int, ...],
        node_positions: dict[str, tuple[float, float]],
    ) -> dict[str, tuple[int, int, int, int]]:
        """Return the lower label band used by the tall floor-three enemy art."""

        if not node_positions:
            return {}
        centers = np.asarray(list(node_positions.values()), dtype=np.float64)
        if len(centers) >= 2:
            distances = np.linalg.norm(
                centers[:, None, :] - centers[None, :, :], axis=2
            )
            distances[distances == 0] = np.inf
            spacing = float(np.median(np.min(distances, axis=1)))
        else:
            spacing = 180.0
        radius_x = min(165.0, max(82.0, spacing * 0.48))
        offset_top = min(76.0, max(58.0, spacing * 0.30))
        offset_bottom = min(158.0, max(118.0, spacing * 0.72))
        height, width = image_shape[:2]
        return {
            node_id: (
                max(0, round(center_x - radius_x)),
                max(0, round(center_y + offset_top)),
                min(width, round(center_x + radius_x)),
                min(height, round(center_y + offset_bottom)),
            )
            for node_id, (center_x, center_y) in node_positions.items()
        }

    @staticmethod
    def _valid_text_projection(
        homography: np.ndarray,
        template: _TextTemplate,
        crop_shape: tuple[int, ...],
    ) -> bool:
        height, width = crop_shape[:2]
        try:
            projected = cv2.perspectiveTransform(
                np.float32(
                    [[
                        [0, 0],
                        [template.width - 1, 0],
                        [template.width - 1, template.height - 1],
                        [0, template.height - 1],
                        [template.width / 2.0, template.height / 2.0],
                    ]]
                ),
                homography,
            )[0]
        except cv2.error:
            return False
        if not np.isfinite(projected).all():
            return False
        projected_width = float(
            (
                np.linalg.norm(projected[1] - projected[0])
                + np.linalg.norm(projected[2] - projected[3])
            )
            / 2.0
        )
        projected_height = float(
            (
                np.linalg.norm(projected[3] - projected[0])
                + np.linalg.norm(projected[2] - projected[1])
            )
            / 2.0
        )
        source_ratio = template.width / max(1.0, template.height)
        projected_ratio = projected_width / max(1.0, projected_height)
        center_x, center_y = projected[4]
        return (
            width * 0.18 <= projected_width <= width * 1.35
            and height * 0.18 <= projected_height <= height * 1.45
            and 0.55 <= projected_ratio / source_ratio <= 1.80
            and -width * 0.10 <= center_x <= width * 1.10
            and -height * 0.20 <= center_y <= height * 1.20
            and abs(float(cv2.contourArea(projected[:4]))) >= width * height * 0.025
        )

    def _match_text_crop(self, crop: np.ndarray) -> NodeTypeDetection | None:
        """Classify a node from the fixed vocabulary printed below it."""

        if crop.size == 0 or not self.text_templates:
            return None
        keypoints, descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None
        )
        if descriptors is None:
            return None

        ranked: list[
            tuple[float, _TextTemplate, int, float, float]
        ] = []
        for template in self.text_templates:
            pairs = self.matcher.knnMatch(template.descriptors, descriptors, k=2)
            good = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance < 0.82 * pair[1].distance
            ]
            if len(good) < 8:
                continue
            source = np.float32(
                [template.keypoints[item.queryIdx].pt for item in good]
            ).reshape(-1, 1, 2)
            destination = np.float32(
                [keypoints[item.trainIdx].pt for item in good]
            ).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(
                source, destination, cv2.RANSAC, 4.0
            )
            if (
                homography is None
                or mask is None
                or not self._valid_text_projection(
                    homography, template, crop.shape
                )
            ):
                continue
            inliers = int(mask.sum())
            ratio = inliers / len(good)
            inlier_x = source.reshape(-1, 2)[mask.ravel().astype(bool), 0]
            coverage = (
                float((inlier_x.max() - inlier_x.min()) / template.width)
                if len(inlier_x) >= 2
                else 0.0
            )
            if inliers < 10 or ratio < 0.58 or coverage < 0.52:
                continue
            score = (
                inliers
                * (0.65 + 0.35 * ratio)
                * min(1.0, coverage / 0.55)
            )
            ranked.append((score, template, inliers, ratio, coverage))

        if not ranked:
            return None
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best, inliers, ratio, coverage = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        if best_score < 10.0 or (second_score and best_score < second_score * 1.22):
            return None
        margin = (
            1.0
            if not second_score
            else min(1.0, best_score / second_score - 1.0)
        )
        confidence = min(1.0, 0.52 + (inliers - 10) / 30.0)
        confidence *= min(1.0, ratio / 0.78)
        confidence *= min(1.0, coverage / 0.72)
        confidence *= 0.88 + 0.12 * margin
        return NodeTypeDetection(
            node_id="",
            node_type=best.node_type,
            icon_name=best.icon_name,
            confidence=float(confidence),
            inliers=inliers,
            inlier_ratio=float(ratio),
            from_text=True,
        )

    @staticmethod
    def _valid_local_projection(
        homography: np.ndarray,
        template: _IconTemplate,
        crop_shape: tuple[int, ...],
    ) -> bool:
        """Reject partial or collapsed homographies on unrelated circular art."""

        height, width = crop_shape[:2]
        extent = min(height, width)
        try:
            projected = cv2.perspectiveTransform(
                np.float32(
                    [[
                        [0, 0],
                        [template.width - 1, 0],
                        [template.width - 1, template.height - 1],
                        [0, template.height - 1],
                        [template.width / 2.0, template.height / 2.0],
                    ]]
                ),
                homography,
            )[0]
        except cv2.error:
            return False
        if not np.isfinite(projected).all():
            return False
        projected_width = float(
            (
                np.linalg.norm(projected[1] - projected[0])
                + np.linalg.norm(projected[2] - projected[3])
            )
            / 2.0
        )
        projected_height = float(
            (
                np.linalg.norm(projected[3] - projected[0])
                + np.linalg.norm(projected[2] - projected[1])
            )
            / 2.0
        )
        center_error = float(
            np.linalg.norm(
                projected[4] - np.asarray((width / 2.0, height / 2.0))
            )
            / max(1, extent)
        )
        area_ratio = abs(float(cv2.contourArea(projected[:4]))) / max(
            1, width * height
        )
        return (
            extent * 0.25 <= projected_width <= extent * 2.5
            and extent * 0.25 <= projected_height <= extent * 2.5
            and 0.40 <= projected_width / max(1.0, projected_height) <= 2.5
            and center_error <= 0.70
            and area_ratio >= 0.04
        )

    def _match_crop(self, crop: np.ndarray) -> NodeTypeDetection | None:
        if crop.size == 0:
            return None
        keypoints, descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), None
        )
        if descriptors is None:
            return (
                self._gain_loss_fallback(crop)
                or self._encounter_fallback(crop)
                or self._overlook_fallback(crop)
            )

        ranked: list[tuple[float, _IconTemplate, int, float]] = []
        for template in self.templates:
            pairs = self.matcher.knnMatch(template.descriptors, descriptors, k=2)
            good = [
                pair[0]
                for pair in pairs
                if len(pair) == 2
                and pair[0].distance < 0.78 * pair[1].distance
            ]
            if len(good) < 6:
                continue
            source = np.float32(
                [template.keypoints[item.queryIdx].pt for item in good]
            ).reshape(-1, 1, 2)
            destination = np.float32(
                [keypoints[item.trainIdx].pt for item in good]
            ).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(
                source, destination, cv2.RANSAC, 5.0
            )
            if (
                homography is None
                or mask is None
                or not self._valid_local_projection(homography, template, crop.shape)
            ):
                continue
            inliers = int(mask.sum())
            ratio = inliers / len(good)
            minimum_inliers = 6 if template.node_type == "险路尽头" else 9
            if inliers < minimum_inliers or ratio < 0.55:
                continue
            score = inliers * (0.65 + 0.35 * ratio)
            ranked.append((score, template, inliers, ratio))

        if not ranked:
            return (
                self._gain_loss_fallback(crop)
                or self._encounter_fallback(crop)
                or self._overlook_fallback(crop)
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best, inliers, ratio = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        # Shared circular backgrounds can yield a handful of features for
        # several icons.  Require a small winning margin for weak matches.
        if inliers < 12 and second_score and best_score < second_score * 1.12:
            return None
        margin = 1.0 if not second_score else min(1.0, best_score / second_score - 1.0)
        confidence = min(1.0, 0.42 + inliers / 28.0) * min(1.0, ratio / 0.78)
        confidence *= 0.85 + 0.15 * margin
        return NodeTypeDetection(
            node_id="",
            node_type=best.node_type,
            icon_name=best.icon_name,
            confidence=float(confidence),
            inliers=inliers,
            inlier_ratio=float(ratio),
        )

    @staticmethod
    def _encounter_fallback(crop: np.ndarray) -> NodeTypeDetection | None:
        """Recognize the coral encounter silhouette when a badge occludes it."""

        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        coral = (
            (hsv[:, :, 0] < 30)
            & (hsv[:, :, 1] > 35)
            & (hsv[:, :, 2] > 65)
        ).astype(np.uint8)
        coral = cv2.morphologyEx(
            coral,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), dtype=np.uint8),
        )
        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
            coral, 8
        )
        component_index = (
            1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) if count > 1 else 0
        )
        largest = int(stats[component_index, cv2.CC_STAT_AREA]) if count > 1 else 0
        coverage = float(np.mean(coral))
        center_x, center_y = (
            centroids[component_index] if count > 1 else (0.0, 0.0)
        )
        height, width = coral.shape
        if (
            largest < max(220, round(coral.size * 0.025))
            or not 0.028 <= coverage <= 0.12
            or not width * 0.15 <= center_x <= width * 0.55
            or not height * 0.05 <= center_y <= height * 0.45
        ):
            return None
        return NodeTypeDetection(
            node_id="",
            node_type="不期而遇",
            icon_name="encounter.png",
            confidence=0.86,
            inliers=0,
            inlier_ratio=0.0,
        )

    def _overlook_fallback(self, crop: np.ndarray) -> NodeTypeDetection | None:
        """Use the circle signature only when no fixed icon matched by SIFT."""

        if not self._has_overlook_marker(crop):
            return None
        return NodeTypeDetection(
            node_id="",
            node_type="羽瞰点",
            icon_name="overlook.png",
            confidence=0.84,
            inliers=3,
            inlier_ratio=1.0,
        )

    @staticmethod
    def _has_overlook_marker(crop: np.ndarray) -> bool:
        """Recognize the highlighted overlook icon's three aligned circles.

        This state is almost featureless, so SIFT may not form a homography.
        Its small-dot, bullseye, large-ring signature remains stable under
        highlighting and scale changes.
        """

        height, width = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        upper = cv2.GaussianBlur(gray[: round(height * 0.68)], (5, 5), 1.2)
        circles = cv2.HoughCircles(
            upper,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(8, round(width * 0.065)),
            param1=80,
            param2=10,
            minRadius=max(2, round(width * 0.015)),
            maxRadius=max(5, round(width * 0.13)),
        )
        if circles is None:
            return False
        values = sorted(circles[0], key=lambda item: item[0])
        yy, xx = np.ogrid[:height, :width]
        for left_index, left in enumerate(values):
            for middle_index in range(left_index + 1, len(values)):
                middle = values[middle_index]
                for right in values[middle_index + 1 :]:
                    gap_left = middle[0] - left[0]
                    gap_right = right[0] - middle[0]
                    radii = (left[2] / width, middle[2] / width, right[2] / width)
                    if not (
                        max(left[1], middle[1], right[1])
                        - min(left[1], middle[1], right[1])
                        <= height * 0.05
                        and 0.02 <= radii[0] <= 0.07
                        and 0.015 <= radii[1] <= 0.07
                        and 0.07 <= radii[2] <= 0.13
                        and gap_right > 0
                        and 0.75 <= gap_left / gap_right <= 1.33
                        and width * 0.42 <= right[0] - left[0] <= width * 0.66
                    ):
                        continue
                    brightness: list[float] = []
                    for center_x, center_y, radius in (left, middle, right):
                        radius = float(radius)
                        squared_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
                        ring = (
                            squared_distance >= max(1.0, radius - 2.0) ** 2
                        ) & (squared_distance <= (radius + 2.0) ** 2)
                        brightness.append(float(gray[ring].mean()))
                    if (
                        brightness[0] >= 60.0
                        and brightness[1] >= 80.0
                        and brightness[2] >= 80.0
                    ):
                        return True
        return False

    def detect(
        self,
        image: np.ndarray,
        node_positions: dict[str, tuple[float, float]],
    ) -> dict[str, NodeTypeDetection]:
        self._begin_image(image)
        regions = self.detection_regions(image.shape, node_positions)
        text_regions = self.text_detection_regions(image.shape, node_positions)
        danger_text_regions = self.danger_enemy_text_regions(
            image.shape, node_positions
        )
        found: dict[str, NodeTypeDetection] = {}

        # Once the graph has supplied stable node centers, the printed label
        # is the most direct semantic signal and is unaffected by icon glow,
        # selection markers, or character overlays.  Resolve every reliable
        # label first; icon matching below is deliberately only a fallback.
        for node_id in node_positions:
            region = text_regions[node_id]
            if region not in self._text_crop_cache:
                x0, y0, x1, y1 = region
                self._text_crop_cache[region] = self._match_text_crop(
                    image[y0:y1, x0:x1]
                )
            match = self._text_crop_cache[region]
            if match is None:
                continue
            found[node_id] = NodeTypeDetection(
                node_id=node_id,
                node_type=match.node_type,
                icon_name=match.icon_name,
                confidence=match.confidence,
                inliers=match.inliers,
                inlier_ratio=match.inlier_ratio,
                from_text=True,
            )

        # The floor-three danger-enemy artwork is substantially taller than
        # the floor-five asset, so its printed label falls below the normal
        # label band. Only accept the exact fixed-node label in this pass.
        for node_id in node_positions:
            if node_id in found:
                continue
            region = danger_text_regions[node_id]
            if region not in self._text_crop_cache:
                x0, y0, x1, y1 = region
                self._text_crop_cache[region] = self._match_text_crop(
                    image[y0:y1, x0:x1]
                )
            match = self._text_crop_cache[region]
            if match is None or match.node_type != "险路恶敌":
                continue
            found[node_id] = NodeTypeDetection(
                node_id=node_id,
                node_type=match.node_type,
                icon_name=match.icon_name,
                confidence=match.confidence,
                inliers=match.inliers,
                inlier_ratio=match.inlier_ratio,
                from_text=True,
            )

        for node_id in node_positions:
            if node_id in found:
                continue
            region = regions[node_id]
            if region not in self._crop_cache:
                x0, y0, x1, y1 = region
                self._crop_cache[region] = self._match_crop(image[y0:y1, x0:x1])
            match = self._crop_cache[region]
            if match is None:
                continue
            found[node_id] = NodeTypeDetection(
                node_id=node_id,
                node_type=match.node_type,
                icon_name=match.icon_name,
                confidence=match.confidence,
                inliers=match.inliers,
                inlier_ratio=match.inlier_ratio,
            )
        return found
