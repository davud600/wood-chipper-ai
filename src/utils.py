from PIL import Image
from cv2.typing import MatLike

import numpy as np
import pytesseract
import fitz
import random
import cv2
import csv
import io
import re
import os

from src.custom_types import Dataset


tesseract_config = r"--oem 3 --psm 3"
max_length = 3064
pages_to_append = 4
training_mini_batch_size = 6
testing_mini_batch_size = 6
learning_rate = 0.00005
weight_decay = 0.01
patience = 10
factor = 0.5
epochs = 5
log_steps = 10
eval_steps = 50
pymupdf_dpi = 300
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

MODEL_PATH = os.path.join(project_root, "model", "model.pth")
DOWNLOADS_DIR = os.path.join(project_root, "downloads")
IMAGES_DIR = os.path.join(project_root, "pdf_images")
PROCESSING_IMAGES_DIR = os.path.join(project_root, "processing_images")
SPLIT_DOCUMENTS_DIR = os.path.join(project_root, "split_documents")
TRAINING_DATA_CSV = os.path.join(project_root, "training_data.csv")
TESTING_DATA_CSV = os.path.join(project_root, "testing_data.csv")
PDF_DIR = os.path.join(project_root, "pdfs")
EDGE_CASES_FILE_PATH = os.path.join(project_root, "bad_files.txt")

# pymupdf_dpi = 72
PAGE_SIMILARITY_THRESHOLD = 0.7
TRAINING_PERCENTAGE = 0.8
TYPES = {
    "unknown": 0,
    "original-lease": 1,
    "lease-renewal": 2,
    "closing-document": 3,
    "sublease": 4,
    "renovation-alteration-document": 5,
    "proprietary-lease": 6,
    "purchase-application": 7,
    "refinance-document": 8,
    "tenant-correspondence": 9,
    "transfer-document": 10,
    "sublease-renewal": 11,
    "transfer-of-title": 12,
}


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

    zipped_data = list(zip(contents, pages))
    random.shuffle(zipped_data)
    shuffled_contents, shuffled_pages = zip(*zipped_data)
    mini_batch_features: list[str] = []
    mini_batch_labels: list[int] = []
    counter = 0
    for features, labels in zip(shuffled_contents, shuffled_pages):
        if counter >= mini_batch_size:
            dataset.append(
                {
                    "features": mini_batch_features,
                    "labels": mini_batch_labels,
                }
            )

            mini_batch_features = []
            mini_batch_labels = []
            counter = 0

        mini_batch_features.append(features)
        mini_batch_labels.append(labels)
        counter += 1

    if mini_batch_features:
        dataset.append(
            {
                "features": mini_batch_features,
                "labels": mini_batch_labels,
            }
        )

    for t, type in enumerate(list(TYPES.keys())):
        print(
            f"{type}: {type_counters[t]["first_page"] + type_counters[t]["not_first_page"]} ({type_counters[t]["first_page"]}, {type_counters[t]["not_first_page"]})"
        )

    return dataset
