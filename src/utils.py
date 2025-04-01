import pytesseract
import random
import nltk
import cv2
import csv
import re
import os

from autocorrect import Speller
from cv2.typing import MatLike

from src.custom_types import Dataset


# light auto correct setup.
nltk.download("words")
from nltk.corpus import words

english_words = set(w.lower() for w in words.words())
spell = Speller(lang="en")


api_url = "http://localhost:3001"
max_length = 3064
pages_to_append = 4
training_mini_batch_size = 8
testing_mini_batch_size = 8
learning_rate = 0.00005
weight_decay = 0.005
patience = 10
factor = 0.5
epochs = 2
log_steps = 10
eval_steps = 50
pymupdf_dpi = 300
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

DELETE_REDIS_KEYS_TIMEOUT = 60

SPLITTER_MODEL_PATH = os.path.join(project_root, "models", "splitter.pth")
DOWNLOADS_DIR = os.path.join(project_root, "downloads")
SPLIT_DOCUMENTS_DIR = os.path.join(project_root, "split_documents")
TRAINING_DATA_CSV = os.path.join(project_root, "dataset", "training_data.csv")
TESTING_DATA_CSV = os.path.join(project_root, "dataset", "testing_data.csv")
PDF_DIR = os.path.join(project_root, "dataset", "pdfs")
EDGE_CASES_FILE_PATH = os.path.join(project_root, "dataset", "bad_files.txt")

# pymupdf_dpi = 72
PAGE_SIMILARITY_THRESHOLD = 0.7
TRAINING_PERCENTAGE = 0.75
DOCUMENT_TYPES = {
    "unknown": 0,
    "lease": 1,
    "lease-agreement": 2,
    "lease-renewal": 3,
    "sublease": 4,
    "sublease-agreement": 5,
    "sublease-renewal": 6,
    "proprietary-lease": 7,
    "tenant-correspondence": 8,
    "transfer-of-title": 9,
    "purchase-application": 10,
    "closing-document": 11,
    "alteration-document": 12,
    "renovation-document": 13,
    "refinance-document": 14,
    "transfer-document": 15,
}


def get_document_type(file_name: str) -> int:
    file_name = file_name.replace("-", " ").replace("_", " ")

    if "proprietary" in file_name.lower():
        return DOCUMENT_TYPES["proprietary-lease"]
    elif "tenant correspondence" in file_name.lower():
        return DOCUMENT_TYPES["tenant-correspondence"]
    elif "transfer" in file_name.lower() and "title" in file_name.lower():
        return DOCUMENT_TYPES["transfer-of-title"]
    elif "purchase" in file_name.lower():
        return DOCUMENT_TYPES["purchase-application"]
    elif "closing" in file_name.lower():
        return DOCUMENT_TYPES["closing-document"]
    elif "alteration" in file_name.lower():
        return DOCUMENT_TYPES["alteration-document"]
    elif "renovation" in file_name.lower():
        return DOCUMENT_TYPES["renovation-document"]
    elif "refinance" in file_name.lower():
        return DOCUMENT_TYPES["refinance-document"]
    elif "transfer" in file_name.lower():
        return DOCUMENT_TYPES["transfer-document"]
    elif "sublease renewal" in file_name.lower():
        return DOCUMENT_TYPES["sublease-renewal"]
    elif "sublease agreement" in file_name.lower():
        return DOCUMENT_TYPES["sublease-agreement"]
    elif "sublease" in file_name.lower():
        return DOCUMENT_TYPES["sublease"]
    elif "lease renewal" in file_name.lower():
        return DOCUMENT_TYPES["lease-renewal"]
    elif "lease agreement" in file_name.lower():
        return DOCUMENT_TYPES["lease-agreement"]
    elif "lease" in file_name.lower():
        return DOCUMENT_TYPES["lease"]

    return DOCUMENT_TYPES["unknown"]


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


def get_dataset(path: str, mini_batch_size: int) -> tuple[Dataset, int, int]:
    data: list[tuple[str, int, int, str]] = []
    dataset: Dataset = []
    contents: list[str] = []
    pages: list[int] = []
    types: list[int] = []
    N0 = 0
    N1 = 0

    with open(file=path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)

        for r, row in enumerate(reader):
            if r == 0:  # skip headers.
                continue

            data += [(str(row[0]), int(row[1]), int(row[2]), str(row[3]))]

    type_counters = [
        {"first_page": 0, "not_first_page": 0}
        for _ in range(len(DOCUMENT_TYPES.keys()))
    ]

    for r, row in enumerate(data):
        content = row[0]
        page = row[1]
        type = row[2]
        file = row[3]

        non_first_pages_prob = 0.5  # bigger -> more non-first pages.
        if page != 1 and random.random() > non_first_pages_prob:
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

    N0 = 0
    N1 = 0
    for t, document_type in enumerate(list(DOCUMENT_TYPES.keys())):
        N0 += type_counters[t]["not_first_page"]
        N1 += type_counters[t]["first_page"]

        print(
            f"{document_type}: {type_counters[t]["first_page"] + type_counters[t]["not_first_page"]} ({type_counters[t]["first_page"]}, {type_counters[t]["not_first_page"]})"
        )

    return dataset, N0, N1


def is_safe_to_correct(word: str) -> bool:
    if len(word) <= 3:
        return False
    if word.lower() in english_words:
        return True
    if re.match(r"^[A-Z0-9]+$", word):
        return False
    if any(char.isdigit() for char in word):
        return False

    return True


def light_autocorrect(text: str) -> str:
    corrected_words = []

    for word in text.split():
        if is_safe_to_correct(word):
            corrected = spell(word)
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def split_into_n_chunks(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]
