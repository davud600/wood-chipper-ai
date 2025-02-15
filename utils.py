from PIL import Image
from enum import Enum
import numpy as np
import fitz  # PyMuPDF
import cv2
import io
import re


pymupdf_dpi = 300
PDF_DIR = "pdfs"
IMAGE_DIR = "pdf_images"
TRAINING_DATA_CSV = "training_data.csv"
TESTING_DATA_CSV = "testing_data.csv"
TRAINING_PERCENTAGE = 0.8
TYPES = {
    "unknown": 0,
    "original-lease": 1,
    "lease-renewal": 2,
    "closing-document": 3,
    "sublease": 4,
    "alteration-document": 5,
    "renovation-document": 6,
    "proprietary-lease": 7,
    "purchase-application": 8,
    "refinance-document": 9,
    "tenant-correspondence": 10,
    "transfer-document": 11,
}


class DocumentType(Enum):
    UNKNOWN = 0
    ORIGINAL_LEASE = 1
    LEASE_RENEWAL = 2
    CLOSING_DOCUMENT = 3
    SUBLEASE = 4
    ALTERATION_DOCUMENT = 5
    RENOVATION_DOCUMENT = 6
    PROPRIETARY_LEASE = 7
    PURCHASE_APPLICATION = 8
    REFINANCE_DOCUMENT = 9
    TENANT_CORRESPONDENCE = 10
    TRANSFER_DOCUMENT = 11


def get_doc_type_from_name(file_name: str) -> DocumentType:
    if "sublease" in file_name.lower():
        return TYPES["sublease"]
    elif "closing" in file_name.lower():
        return TYPES["closing-document"]
    elif "correspondence" in file_name.lower():
        return TYPES["tenant-correspondence"]
    elif "lease renewal" in file_name.lower():
        return TYPES["lease-renewal"]
    elif "lease" in file_name.lower():
        return TYPES["original-lease"]
    elif "alteration" in file_name.lower():
        return TYPES["alteration-document"]
    elif "renovation" in file_name.lower():
        return TYPES["renovation-document"]
    elif "proprietary lease" in file_name.lower():
        return TYPES["proprietary-lease"]
    elif "purchase application" in file_name.lower():
        return TYPES["purchase-application"]
    elif "refinance document" in file_name.lower():
        return TYPES["refinance-document"]
    elif "transfer document" in file_name.lower():
        return TYPES["transfer-document"]
    else:
        return TYPES["unknown"]


def normalize_image(image):
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


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


def scale_image(image, ppi=300):
    dpi_scale = ppi / 72
    new_size = (int(image.width * dpi_scale), int(image.height * dpi_scale))
    return image.resize(new_size, Image.LANCZOS)


def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def convert_to_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def binarize_image(image):
    gray = convert_to_grayscale(image)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return binary


def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    # Remove extraneous punctuation
    text = re.sub(r'[^a-zA-Z0-9.,;:\'"\-\s]', "", text)
    return text


def render_and_preprocess_page_in_memory(doc: fitz.open, page_num: int):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=pymupdf_dpi)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preprocessing steps
    image = normalize_image(image)
    # image = correct_skew(image)
    # image = remove_noise(image)
    image = binarize_image(image)

    return Image.fromarray(image)
