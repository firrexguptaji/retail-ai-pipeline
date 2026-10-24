"""Upload limits and cheap image checks shared by gateway and detector.

``validate_image_bytes`` reads only the image header (plus Pillow's verify pass),
so oversized or non-image uploads are rejected before any full decode.
"""

import os
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from .errors import ServiceError

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


def max_upload_bytes() -> int:
    """Max size of an uploaded image file (MAX_UPLOAD_MB, default 10)."""
    return int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)


def max_payload_bytes() -> int:
    """Max size of an internal JSON payload, e.g. detector -> grouping (MAX_PAYLOAD_MB, default 50)."""
    return int(float(os.getenv("MAX_PAYLOAD_MB", "50")) * 1024 * 1024)


def max_image_pixels() -> int:
    """Max width x height accepted (MAX_IMAGE_PIXELS, default 40 megapixels)."""
    return int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))


def validate_image_bytes(data: bytes) -> tuple[str, int, int]:
    """Return (format, width, height) or raise ServiceError."""
    if not data:
        raise ServiceError("empty_file", "Uploaded file is empty", 400)
    try:
        with Image.open(BytesIO(data)) as img:
            fmt, (width, height) = img.format, img.size
            if fmt not in ALLOWED_FORMATS:
                raise ServiceError(
                    "unsupported_image_type",
                    f"Unsupported image type; allowed: {', '.join(sorted(ALLOWED_FORMATS))}",
                    415,
                )
            if width * height > max_image_pixels():
                raise ServiceError("image_too_large", "Image dimensions exceed the limit", 413)
            img.verify()
    except ServiceError:
        raise
    except Image.DecompressionBombError:
        raise ServiceError("image_too_large", "Image dimensions exceed the limit", 413) from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ServiceError("invalid_image", "Uploaded file is not a valid image", 400) from None
    return fmt, width, height
