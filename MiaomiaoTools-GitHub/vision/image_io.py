from __future__ import annotations

from os import PathLike
from pathlib import Path

import cv2
import numpy as np


def read_image_bytes(data: bytes, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Decode image bytes, including gallery data that has no filesystem path.

    Android photo pickers commonly return a content URI/stream rather than a
    path.  Pillow is used only to apply EXIF orientation; OpenCV then receives
    the same BGR representation as :func:`read_image`.
    """

    if not data:
        return None
    # Node assets rely on their alpha channel. OpenCV preserves it exactly,
    # while the EXIF-aware Pillow path below intentionally converts photos to
    # RGB/BGR.
    if flags == cv2.IMREAD_UNCHANGED:
        encoded = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(encoded, flags) if encoded.size else None
    try:
        from io import BytesIO
        from PIL import Image, ImageOps

        with Image.open(BytesIO(data)) as pil_image:
            oriented = ImageOps.exif_transpose(pil_image).convert("RGB")
            rgb = np.asarray(oriented)
        if flags == cv2.IMREAD_GRAYSCALE:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if flags != cv2.IMREAD_COLOR:
            # Keep the API predictable for the flags used by the recognizer;
            # callers needing alpha can decode with Pillow directly.
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, TypeError):
        # Fall back to OpenCV for formats Pillow cannot open.
        encoded = np.frombuffer(data, dtype=np.uint8)
        return cv2.imdecode(encoded, flags) if encoded.size else None


def read_image(
    path: str | PathLike[str], flags: int = cv2.IMREAD_COLOR
) -> np.ndarray | None:
    """Read an image through Python so Windows Unicode paths stay supported."""

    try:
        encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return read_image_bytes(encoded.tobytes(), flags)
