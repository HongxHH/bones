"""Opt-in original vs enhanced picture.

Original means the decoded frame before the model, bicubic-upscaled to the
display size at the moment it is shown. The stored form is uint8 at the
decoded resolution, not a second full-resolution buffer. Entries are dropped
as each frame is presented. ``compare_view`` defaults off, so playback keeps
a single picture and no extra frames.
"""

from __future__ import annotations

import numpy as np


def enabled(config) -> bool:
    return bool(getattr(config, "compare_view", False))


def start_segment(segment, config) -> None:
    """Begin a per-frame baseline list aligned with this segment's pictures."""
    if enabled(config):
        segment.compare_lr = []
    else:
        segment.compare_lr = None


def drop_segment(segment) -> None:
    if segment is not None and getattr(segment, "compare_lr", None) is not None:
        segment.compare_lr = None


def keep_input(segment, tensor) -> None:
    """Remember the pre-model frame. No-op unless this segment is collecting."""
    rows = getattr(segment, "compare_lr", None) if segment is not None else None
    if rows is None or tensor is None:
        return
    import torch

    sample = tensor.detach()
    if sample.ndim == 4:
        sample = sample[0]
    # Out of place: the live tensor still goes into the model.
    stored = (sample * 255).clamp(0, 255).to(dtype=torch.uint8).cpu().contiguous()
    rows.append(stored)


def keep_same(segment) -> None:
    """This frame was not enhanced; both sides of the wipe show the same picture."""
    rows = getattr(segment, "compare_lr", None) if segment is not None else None
    if rows is None:
        return
    rows.append(None)


def pad_same(segment, count: int) -> None:
    rows = getattr(segment, "compare_lr", None) if segment is not None else None
    if rows is None or count <= 0:
        return
    rows.extend([None] * int(count))


def take(segment_or_rows, index: int):
    """Return the stored LR for this rendered frame and release it."""
    rows = segment_or_rows
    if rows is not None and not isinstance(rows, list):
        rows = getattr(rows, "compare_lr", None)
    if not rows or index < 0 or index >= len(rows):
        return None
    item = rows[index]
    rows[index] = None
    return item


def to_display_bgr(lr, width: int, height: int) -> np.ndarray:
    """Bicubic the stored LR (RGB CHW uint8) to the display size, as BGR."""
    import cv2

    rgb = _rgb_hwc(lr, int(width), int(height))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _rgb_hwc(lr, width: int, height: int) -> np.ndarray:
    try:
        import torch

        if isinstance(lr, torch.Tensor):
            sample = lr.detach().cpu()
            if sample.ndim == 4:
                sample = sample[0]
            if sample.dtype != torch.uint8:
                sample = (sample * 255).clamp(0, 255).to(dtype=torch.uint8)
            if int(sample.shape[-1]) != width or int(sample.shape[-2]) != height:
                scaled = sample.float().unsqueeze(0).div(255.0)
                scaled = torch.nn.functional.interpolate(
                    scaled,
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                )
                sample = scaled.mul(255).clamp(0, 255).to(dtype=torch.uint8).squeeze(0)
            rgb = sample.permute(1, 2, 0).contiguous().numpy()
            return np.ascontiguousarray(rgb)
    except Exception:
        pass

    import cv2

    arr = np.asarray(lr)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    if arr.shape[0] != height or arr.shape[1] != width:
        arr = cv2.resize(arr, (width, height), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(arr)
