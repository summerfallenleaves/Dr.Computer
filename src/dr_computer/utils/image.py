"""Image helpers: encoding, base64, resizing.

These wrap Pillow so callers don't need to know the exact API. Used by
perceivers (PNG encoding) and providers (base64 + resize before sending
to VLM APIs that cap image dimensions).
"""

from __future__ import annotations

import base64
import io

from PIL import Image

ImageBytes = bytes


def encode_png(image: Image.Image) -> ImageBytes:
    """Encode a PIL image to PNG bytes."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def decode_png(data: ImageBytes) -> Image.Image:
    """Decode PNG/JP``EG bytes into a PIL image."""
    return Image.open(io.BytesIO(data))


def to_base64(data: ImageBytes, *, mime_type: str = "image/png") -> str:
    """Wrap raw image bytes into a ``data:<mime>;base64,...`` URL.

    Suitable for OpenAI-compatible ``image_url`` payloads.
    """
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{b64}"


def resize_to_fit(
    image: Image.Image,
    *,
    max_width: int = 1280,
    max_height: int = 1280,
) -> Image.Image:
    """Resize so width and height are within bounds, preserving aspect ratio.

    No-op if already within bounds. Uses LANCZOS for downscaling quality.
    """
    w, h = image.size
    if w <= max_width and h <= max_height:
        return image
    ratio = min(max_width / w, max_height / h)
    new_size = (max(1, int(w * ratio)), max(1, int(h * ratio)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def resize_bytes(
    data: ImageBytes,
    *,
    max_width: int = 1280,
    max_height: int = 1280,
    format: str = "PNG",
) -> ImageBytes:
    """Decode → resize → re-encode."""
    img = decode_png(data)
    resized = resize_to_fit(img, max_width=max_width, max_height=max_height)
    buf = io.BytesIO()
    resized.save(buf, format=format)
    return buf.getvalue()


__all__ = [
    "ImageBytes",
    "decode_png",
    "encode_png",
    "resize_bytes",
    "resize_to_fit",
    "to_base64",
]
