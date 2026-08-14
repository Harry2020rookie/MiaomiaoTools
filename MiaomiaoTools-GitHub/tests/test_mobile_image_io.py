from io import BytesIO

from PIL import Image

from vision.image_io import read_image_bytes


def test_read_image_bytes_decodes_rgb_as_bgr():
    image = Image.new("RGB", (2, 1), (10, 20, 30))
    stream = BytesIO()
    image.save(stream, format="PNG")

    decoded = read_image_bytes(stream.getvalue())

    assert decoded is not None
    assert decoded.shape == (1, 2, 3)
    assert tuple(decoded[0, 0]) == (30, 20, 10)


def test_read_image_bytes_honors_exif_orientation():
    image = Image.new("RGB", (2, 1), (1, 2, 3))
    exif = image.getexif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    stream = BytesIO()
    image.save(stream, format="JPEG", exif=exif.tobytes())

    decoded = read_image_bytes(stream.getvalue())

    assert decoded is not None
    assert decoded.shape[:2] == (2, 1)
