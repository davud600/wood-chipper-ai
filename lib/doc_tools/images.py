from cv2.typing import MatLike

import numpy as np
import cv2


def binarize(img: np.ndarray) -> np.ndarray:
    _, binarized = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarized


def denoise(img: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(img, h=10)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def format_image_to_shape(
    img: np.ndarray, target_w: int, target_h: int, white_thresh: int = 200
) -> np.ndarray:
    h, w = img.shape[:2]

    cv2.imwrite("/home/davud/wood-chipper-ai/before.png", img)

    row_mask = np.any(img < white_thresh, axis=1)
    col_mask = np.any(img < white_thresh, axis=0)

    # Crop blank areas
    img = img[row_mask, :]
    img = img[:, col_mask]

    h, w = img.shape[:2]

    # Step 2: If still too big, crop symmetrically from bottom / right.
    if h > target_h:
        img = img[:target_h, :]
        h = target_h

    if w > target_w:
        img = img[:, :target_w]
        w = target_w

    # Step 3: Pad to target size (centered)
    top = (target_h - h) // 2
    bottom = target_h - h - top
    left = (target_w - w) // 2
    right = target_w - w - left

    padded = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255  # white
    )

    cv2.imwrite("/home/davud/wood-chipper-ai/after.png", padded)

    return padded


def convert_to_grayscale(image: MatLike) -> MatLike:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image
