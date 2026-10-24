import pytest

from common.errors import ServiceError
from common.validation import validate_image_bytes
from tests.conftest import image_bytes


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "BMP"])
def test_accepts_supported_formats(fmt):
    assert validate_image_bytes(image_bytes(fmt))[0] == fmt


@pytest.mark.parametrize(
    "data, code, status",
    [
        (b"", "empty_file", 400),
        (b"definitely not an image", "invalid_image", 400),
        (image_bytes("PNG")[:40], "invalid_image", 400),  # truncated header
        (image_bytes("GIF"), "unsupported_image_type", 415),
    ],
)
def test_rejects_bad_input(data, code, status):
    with pytest.raises(ServiceError) as exc:
        validate_image_bytes(data)
    assert (exc.value.code, exc.value.status) == (code, status)


def test_rejects_too_many_pixels(monkeypatch):
    monkeypatch.setenv("MAX_IMAGE_PIXELS", "100")
    with pytest.raises(ServiceError) as exc:
        validate_image_bytes(image_bytes("PNG", size=(20, 20)))
    assert (exc.value.code, exc.value.status) == ("image_too_large", 413)
