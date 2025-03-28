from cv2.typing import MatLike

import numpy as np
import cv2


def normalize_image(image: MatLike) -> MatLike:
    dst = np.zeros_like(image)

    return cv2.normalize(
        image, dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )


def correct_skew(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(binary > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# def scale_image(image, ppi=300):
#     dpi_scale = ppi / 72
#     new_size = (int(image.width * dpi_scale), int(image.height * dpi_scale))
#     return image.resize(new_size, Image.LANCZOS)


def remove_noise(image: MatLike) -> MatLike:
    return cv2.fastNlMeansDenoisingColored(
        image,
        h=5,
        hColor=5,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def convert_to_grayscale(image: MatLike) -> MatLike:
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def binarize_image(image: MatLike) -> MatLike:
    gray = convert_to_grayscale(image)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return binary
