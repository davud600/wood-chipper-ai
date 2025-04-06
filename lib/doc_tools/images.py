from cv2.typing import MatLike

import numpy as np
import cv2


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, h=10)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def format_image_to_shape(
    img: np.ndarray, target_h: int, target_w: int, white_thresh: int = 200
) -> np.ndarray:
    h, w = img.shape[:2]

    # Step 1: Trim mostly white rows/columns
    # Convert to grayscale if needed
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create masks where content is present (darker than threshold)
    row_mask = np.any(gray < white_thresh, axis=1)
    col_mask = np.any(gray < white_thresh, axis=0)

    # Crop blank areas
    img = img[row_mask, :]
    img = img[:, col_mask]

    h, w = img.shape[:2]

    # Step 2: If still too big, center-crop symmetrically
    if h > target_h:
        extra = h - target_h
        top_crop = extra // 2
        bottom_crop = extra - top_crop
        img = img[top_crop : h - bottom_crop, :]
        h = target_h

    if w > target_w:
        extra = w - target_w
        left_crop = extra // 2
        right_crop = extra - left_crop
        img = img[:, left_crop : w - right_crop]
        w = target_w

    # Step 3: Pad to target size (centered)
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left

    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255  # white
    )

    return padded


def convert_to_grayscale(image: MatLike) -> MatLike:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image
