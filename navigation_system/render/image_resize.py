"""Shared helpers for stable model-input image sizing."""

import cv2
import numpy as np


def resize_image_to_width(
    image: np.ndarray,
    target_width: int,
) -> np.ndarray:
    """Resize an image to a fixed width while preserving aspect ratio."""
    if image is None:
        return image

    array = np.asarray(image)
    if array.size == 0:
        return array

    target_width = max(1, int(target_width))
    height, width = array.shape[:2]
    if width <= 0 or target_width == width:
        return array.copy()

    scale = float(target_width) / float(width)
    target_height = max(1, int(round(float(height) * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(array, (target_width, target_height), interpolation=interpolation)


def resize_image_to_square(
    image: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """Resize an image to a fixed square output."""
    if image is None:
        return image

    array = np.asarray(image)
    if array.size == 0:
        return array

    target_size = max(1, int(target_size))
    height, width = array.shape[:2]
    if height == target_size and width == target_size:
        return array.copy()

    interpolation = cv2.INTER_AREA if max(height, width) > target_size else cv2.INTER_LINEAR
    return cv2.resize(array, (target_size, target_size), interpolation=interpolation)


def resize_strip_to_width(
    strip: np.ndarray,
    target_width: int,
) -> np.ndarray:
    """Resize a horizontal strip to match a target width."""
    if strip is None:
        return strip

    array = np.asarray(strip)
    if array.size == 0:
        return array

    target_width = max(1, int(target_width))
    height, width = array.shape[:2]
    if width <= 0 or target_width == width:
        return array.copy()

    scale = float(target_width) / float(width)
    target_height = max(1, int(round(float(height) * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(array, (target_width, target_height), interpolation=interpolation)
