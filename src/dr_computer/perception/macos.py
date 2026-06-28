"""macOS screenshot perceiver.

Uses CoreGraphics via pyobjc to grab the full display. This avoids the
significant per-call overhead of `screencapture` (subprocess) and gives us
direct access to the pixel buffer.

On a Retina Mac, the captured buffer is in *physical* pixels (2x logical).
The ``Observation.width`` and ``Observation.height`` reflect this. Grounders
and providers that produce bbox coordinates must use the same coordinate
space — i.e. physical pixels — for clicks to land correctly.

Permission notes:

- macOS requires the calling app to be granted "Screen Recording" permission.
  On first capture the OS will prompt. If the user denies, captures come
  back blank (or all-black). We surface that as a clear error.
- For background operation (e.g. running under `python` in Terminal), the
  Terminal app (or whichever binary hosts the Python interpreter) must be
  in the Screen Recording allow-list in System Settings → Privacy.
"""

from __future__ import annotations

import io
import logging

from PIL import Image
from Quartz import (
    CGDataProviderCopyData,
    CGDisplayBounds,
    CGDisplayPixelsHigh,
    CGDisplayPixelsWide,
    CGImageGetDataProvider,
    CGImageGetHeight,
    CGImageGetWidth,
    CGMainDisplayID,
    CGWindowListCreateImage,
    kCGNullWindowID,
    kCGWindowImageDefault,
)

from ..core.observation import Observation

logger = logging.getLogger(__name__)


class MacOSScreenshotPerceiver:
    """Captures the main display into an :class:`Observation`.

    Args:
        retina_physical_pixels: If True (default), the screenshot is in
            physical pixels (e.g. 2880x1800 on a Retina MBP). If False, the
            image is downscaled to logical pixels (1440x900) which can save
            bandwidth when sending to an LLM API. Grounding coordinates are
            *always* in whatever the perceiver returned, so keep this
            consistent with the executor's coordinate space.
        min_size: If the captured image has fewer than this many pixels,
            we assume permission is missing and raise. Catches the
            all-black-after-deny case.
    """

    def __init__(
        self,
        *,
        retina_physical_pixels: bool = True,
        min_size: tuple[int, int] = (320, 200),
    ) -> None:
        self.retina_physical_pixels = retina_physical_pixels
        self.min_pixels = min_size[0] * min_size[1]

    async def observe(self) -> Observation:
        return self.capture()

    def capture(self) -> Observation:
        """Synchronous capture. Wrapped as ``observe`` for the Protocol."""
        display_id = CGMainDisplayID()
        bounds = CGDisplayBounds(display_id)

        # CGWindowListCreateImage with kCGNullWindowID captures all displays
        # composed into one image; passing the main display bounds restricts
        # to the main display.
        cg_image = CGWindowListCreateImage(
            bounds,
            kCGNullWindowID,
            kCGNullWindowID,
            kCGWindowImageDefault,
        )
        if cg_image is None:
            raise RuntimeError(
                "CGWindowListCreateImage returned None. Check Screen Recording "
                "permission in System Settings → Privacy & Security."
            )

        width = CGImageGetWidth(cg_image)
        height = CGImageGetHeight(cg_image)

        if width * height < self.min_pixels:
            raise RuntimeError(
                f"Captured image is suspiciously small ({width}x{height}). "
                "Screen Recording permission is likely missing or denied."
            )

        png_bytes = self._cgimage_to_png(cg_image, width, height)

        if not self.retina_physical_pixels:
            png_bytes = self._downscale_to_logical(png_bytes, width, height)
            width, height = self._logical_size(width, height)

        return Observation(
            screenshot=png_bytes,
            width=width,
            height=height,
            source="screenshot",
        )

    @staticmethod
    def _cgimage_to_png(cg_image: object, width: int, height: int) -> bytes:
        """Convert a CoreGraphics image to PNG bytes via Pillow.

        We pull raw RGBA bytes from the CGImage's data provider and hand
        them to Pillow. This is faster than going through PIL.ImageGrab on
        macOS (which itself uses CG under the hood) because we skip the
        intermediate NSBitmapImageRep step.
        """
        provider = CGImageGetDataProvider(cg_image)
        raw: bytes = bytes(CGDataProviderCopyData(provider))

        img = Image.frombytes("RGBA", (width, height), raw)
        # PNG with alpha is bigger; flatten to RGB.
        background = Image.new("RGB", img.size, (0, 0, 0))
        background.paste(img, mask=img.split()[3])
        buf = io.BytesIO()
        background.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _downscale_to_logical(png_bytes: bytes, width: int, height: int) -> bytes:
        """Halve dimensions if this is a 2x Retina display."""
        logical_w, logical_h = MacOSScreenshotPerceiver._logical_size(width, height)
        if (logical_w, logical_h) == (width, height):
            return png_bytes
        img = Image.open(io.BytesIO(png_bytes))
        resized = img.resize((logical_w, logical_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _logical_size(width: int, height: int) -> tuple[int, int]:
        """If width is roughly 2x the display's logical width, halve."""
        display_id = CGMainDisplayID()
        logical_w = CGDisplayPixelsWide(display_id)
        logical_h = CGDisplayPixelsHigh(display_id)
        if logical_w == 0 or logical_h == 0:
            return width, height
        if abs(width - logical_w) < abs(width - 2 * logical_w):
            return width, height
        return logical_w, logical_h


__all__ = ["MacOSScreenshotPerceiver"]
