from typing import Dict
from PIL import Image
from enum import Enum
from cv2.typing import MatLike
import numpy as np
import pytesseract
import random
import fitz  # PyMuPDF
import cv2
import csv
import io
import re


max_length = 3064
pages_to_append = 5
training_mini_batch_size = 4
testing_mini_batch_size = 4
learning_rate = 0.0001
weight_decay = 0.01
patience = 10
factor = 0.5
epochs = 5
log_steps = 10
eval_steps = 50

pymupdf_dpi = 300
# pymupdf_dpi = 72
EDGE_CASES_FILE_PATH = "./bad_files.txt"
PAGE_SIMILARITY_THRESHOLD = 0.7
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


type DatasetMiniBatch = Dict[str, list[str] | list[int]]
type Dataset = list[DatasetMiniBatch]


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


class EdgeCases(Enum):
    START = r"start\((\d+)\)"
    ALIAS = r"alias\((\d+)\)"
    DELETE = "delete"
    AGREEMENT = "agreement"
    SUBLEASE = "sublease"


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


def clean_text(text: str) -> str:
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    # Remove extraneous punctuation
    text = re.sub(r'[^a-zA-Z0-9.,;:\'"\-\s]', "", text)
    return text


def detect_rotation(image: MatLike) -> tuple[int, float]:
    """Detect if the image is rotated based on OCR text orientation."""

    osd = pytesseract.image_to_osd(image)
    angle = int(osd.split("Rotate: ")[1].split("\n")[0])
    confidence = float(osd.split("Orientation confidence: ")[1].split("\n")[0])

    return angle, confidence


def correct_rotation(
    image: MatLike,
) -> MatLike:
    """Corrects the rotation based on detected text orientation."""

    angle, confidence = detect_rotation(image)
    for _ in range(4):
        if confidence >= 3.0:
            break

        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        try:
            angle, confidence = detect_rotation(image)
        except:
            continue

    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)

    return image


def render_and_preprocess_page_in_memory(doc: fitz.open, page_num: int):
    page = doc.load_page(page_num)
    pix = page.get_pixmap(dpi=pymupdf_dpi)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)

    try:
        image = correct_rotation(image)
    except Exception as e:
        print(e)

    # Preprocessing steps
    # image = normalize_image(image)
    # image = correct_skew(image)
    # image = remove_noise(image)
    # image = binarize_image(image)
    # image = cv2.adaptiveThreshold(
    #     image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    # )

    array_image = Image.fromarray(image)
    # array_image.thumbnail((1920, 1080))
    return array_image
    # return image


def get_dataset(path: str, mini_batch_size: int) -> Dataset:
    data: list[tuple[str, int, int, str]] = []
    dataset: Dataset = []
    contents: list[str] = []
    pages: list[int] = []
    types: list[int] = []
    with open(file=path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        for r, row in enumerate(reader):
            if r == 0:  # skip headers.
                continue

            # if (
            #     int(row[2]) != 1
            #     and int(row[2]) != 2
            #     and int(row[2]) != 3
            #     and int(row[2]) != 4
            #     and int(row[2]) != 5
            # ):  # temp:
            #     continue

            data += [(str(row[0]), int(row[1]), int(row[2]), str(row[3]))]

    type_counters = [
        {"first_page": 0, "not_first_page": 0} for _ in range(len(TYPES.keys()))
    ]
    for r, row in enumerate(data):
        content = row[0]
        page = row[1]
        type = row[2]
        file = row[3]

        if page != 1 and random.random() > 0.2:
            continue

        content = f"<curr_page>{content}</curr_page>"
        for next in range(1, pages_to_append + 1):
            if r - next < 0 or data[r - next][3] != file:
                break

            content += f"<next_page_{next}>{data[r - next][0]}</next_page_{next}>"

        pages += [page]
        types += [type]
        contents += [content]
        if page == 1:
            type_counters[type]["first_page"] += 1
        else:
            type_counters[type]["not_first_page"] += 1

    zipped_data = list(zip(contents, pages, types))
    random.shuffle(zipped_data)
    shuffled_contents, shuffled_pages, shuffled_types = zip(*zipped_data)
    mini_batch_features: list[str] = []
    mini_batch_page_labels: list[int] = []
    mini_batch_type_labels: list[int] = []
    counter = 0
    for features, page_labels, type_labels in zip(
        shuffled_contents, shuffled_pages, shuffled_types
    ):
        if counter >= mini_batch_size:
            dataset.append(
                {
                    "features": mini_batch_features,
                    "page_labels": mini_batch_page_labels,
                    "type_labels": mini_batch_type_labels,
                }
            )

            mini_batch_features = []
            mini_batch_page_labels = []
            mini_batch_type_labels = []
            counter = 0

        mini_batch_features.append(features)
        mini_batch_page_labels.append(page_labels)
        mini_batch_type_labels.append(type_labels)
        counter += 1

    if mini_batch_features:
        dataset.append(
            {
                "features": mini_batch_features,
                "page_labels": mini_batch_page_labels,
                "type_labels": mini_batch_type_labels,
            }
        )

    for t, type in enumerate(list(TYPES.keys())):
        print(
            f"{type}: {type_counters[t]["first_page"] + type_counters[t]["not_first_page"]} ({type_counters[t]["first_page"]}, {type_counters[t]["not_first_page"]})"
        )

    return dataset
