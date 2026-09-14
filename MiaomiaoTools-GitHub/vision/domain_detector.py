"""Offline HUD icon matching and conservative, editable fog-node estimates."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DomainHeader:
    idea: str | None = None
    policy: str | None = None
    confidence: float = 0.0
    policy_confidence: float = 0.0


class DomainDetector:
    def __init__(self, root: Path):
        self.catalog = json.loads((root / "data/rules/domains.json").read_text())["entries"]
        self.by_id = {entry["id"]: entry for entry in self.catalog}
        self.templates = {}
        for entry in self.catalog:
            icon = cv2.imread(str(root / entry["icon"]))
            if icon is None:
                raise ValueError(f"缺少实托邦图标：{entry['name']}")
            mask = (icon.min(axis=2) > 180).astype(np.float32)
            yy, xx = np.where(mask)
            self.templates[entry["id"]] = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]

    def _rank(self, target, kind, low, high):
        hits = []
        for entry in self.catalog:
            if entry["kind"] != kind:
                continue
            mask = self.templates[entry["id"]]
            best = (-1., (0, 0, 0, 0))
            for height in range(max(8, round(low)), max(9, round(high)), 2):
                width = max(3, round(height * mask.shape[1] / mask.shape[0]))
                padding = max(2, round(height * .2))
                glyph = np.pad(cv2.resize(mask, (width, height)), padding)
                if glyph.shape[0] > target.shape[0] or glyph.shape[1] > target.shape[1]:
                    continue
                scores = cv2.matchTemplate(target, glyph, cv2.TM_CCOEFF_NORMED)
                _, score, _, (x, y) = cv2.minMaxLoc(scores)
                if score > best[0]:
                    best = (score, (x + padding, y + padding, width, height))
            hits.append((best[0], entry["id"], best[1]))
        return sorted(hits, reverse=True)

    def detect_header(self, image: np.ndarray) -> DomainHeader:
        # Bound computation while retaining the small policy badge.
        if image.shape[0] < 250:
            return DomainHeader()
        if image.shape[0] != 1200:
            image = cv2.resize(image, None, fx=1200/image.shape[0], fy=1200/image.shape[0])
        h, w = image.shape[:2]
        if h < 250:
            return DomainHeader()
        crop = image[:round(h * .125), round(w * .30):round(w * .55)]
        target = (crop.min(axis=2) > 125).astype(np.float32)
        ideas = self._rank(target, "idea", h*.026, h*.064)
        score, idea, (x, y, gw, gh) = ideas[0]
        if score < .62 or score - ideas[1][0] < .08:
            return DomainHeader()
        # The policy sits at the lower right of the main emblem. Searching the
        # whole title would confuse small text with the simpler policy shapes.
        left, top = max(0, round(x+gw*.45)), max(0, round(y+gh*.45))
        badge = target[top:min(target.shape[0], round(y+gh*2.2)),
                       left:min(target.shape[1], round(x+gw*2.2))]
        policies = self._rank(badge, "policy", h*.013, h*.04)
        ps, policy, _ = policies[0]
        if ps < .66 or ps - policies[1][0] < .08:
            policy, ps = None, 0.
        return DomainHeader(idea, policy, score, ps)

    @staticmethod
    def segment_fog(image, bounds, step) -> np.ndarray:
        """Return all large desaturated fog patches, including disconnected ones.

        Filled contours give a visual estimate only. Small node glows and thin
        route strokes are removed before sampling affected-node candidates.
        """
        h, w = image.shape[:2]
        left, top, right, bottom = bounds
        left, top = max(0, round(left)), max(round(h*.13), round(top))
        right, bottom = min(w, round(right)), min(round(h*.86), round(bottom))
        empty = np.zeros((h, w), np.uint8)
        if left >= right or top >= bottom or step < 15:
            return empty
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        roi = hsv[top:bottom, left:right]
        background = roi[(roi[:, :, 1] > 80) & (roi[:, :, 2] > 10) & (roi[:, :, 2] < 120)]
        if len(background) < 100:
            return empty
        hue, saturation = np.median(background[:, :2], axis=0)
        delta = np.abs(hsv[:, :, 0].astype(float)-hue)
        delta = np.minimum(delta, 180-delta)
        candidate = ((delta < 23) & (hsv[:, :, 1] < saturation*.70)
                     & (hsv[:, :, 2] > 18) & (hsv[:, :, 2] < 145))
        mask = empty.copy()
        mask[top:bottom, left:right] = candidate[top:bottom, left:right]*255

        def kernel(fraction):
            size = max(3, round(step*fraction) | 1)
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel(.04))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel(.19))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > step*step*.65:
                cv2.drawContours(empty, [contour], -1, 255, -1)
        filled = cv2.morphologyEx(empty, cv2.MORPH_OPEN, kernel(.39))
        _, labels, stats, _ = cv2.connectedComponentsWithStats(filled)
        keep = [i for i, component in enumerate(stats) if i and component[4] > step*step*.7]
        return np.where(np.isin(labels, keep), 255, 0).astype(np.uint8)

    @staticmethod
    def fog_coverage(mask, center, step) -> float:
        x, y = map(round, center)
        r = max(3, round(step*.22))
        patch = mask[max(0, y-r):min(mask.shape[0], y+r),
                     max(0, x-r):min(mask.shape[1], x+r)]
        return float(np.mean(patch > 0)) if patch.size else 0.
