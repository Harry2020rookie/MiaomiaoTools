"""Read the gold (I)–(V) floor label in the game's top-centre HUD.

Colour and glyph geometry avoid an OCR runtime. Only a complete, bracketed
label is accepted; missing/cropped headers return None rather than guessing
from the number of currently visible nodes.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_floor(image: np.ndarray) -> int | None:
    height, width = image.shape[:2]
    if height < 200 or width < 300:
        return None
    crop = image[round(height * .025):round(height * .10),
                 round(width * .35):round(width * .62)]
    mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV),
                       (18, 125, 130), (40, 255, 255))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    glyphs = []
    for index, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if not height * .010 < h < height * .035 or area < 6:
            continue
        if not .08 < w / h < .95:
            continue
        yy, xx = np.where(labels[y:y+h, x:x+w] == index)
        middle = xx[(yy >= h * .35) & (yy < h * .65)]
        ends = xx[(yy < h * .25) | (yy >= h * .75)]
        if not len(middle) or not len(ends):
            continue
        curve = (float(middle.mean()) - float(ends.mean())) / h
        glyphs.append((int(x), int(y), int(w), int(h), curve))
    glyphs.sort()
    results = set()
    for start in range(len(glyphs)):
        for count in (3, 4, 5):
            group = glyphs[start:start + count]
            if len(group) != count:
                continue
            first, last = group[0], group[-1]
            if first[4] > -.03 or last[4] < .03:
                continue
            if first[2] / first[3] >= .42 or last[2] / last[3] >= .42:
                continue
            median_height = float(np.median([g[3] for g in group]))
            if any(abs(g[1] + g[3] / 2 - first[1] - first[3] / 2)
                   > median_height * .22 for g in group):
                continue
            if any(not .7 < g[3] / median_height < 1.35 for g in group):
                continue
            if any(not 0 < b[0] - a[0] - a[2] < median_height * .85
                   for a, b in zip(group, group[1:])):
                continue
            roman = ""
            for _, _, w, h, curve in group[1:-1]:
                if .42 <= w / h <= .9:
                    roman += "V"
                elif w / h < .4 and abs(curve) < .04:
                    roman += "I"
                else:
                    break
            else:
                floor = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5}.get(roman)
                if floor is not None:
                    results.add(floor)
    return results.pop() if len(results) == 1 else None
