import numpy as np
import fitz  # PyMuPDF
import random
import csv
import os
import re

from lib.doc_tools import get_image_contents, convert_pdf_page_to_image
from type_defs import DatasetRow, EdgeCaseFiles, EdgeCases
from config import (
    EDGE_CASES_FILE_PATH,
    PDF_DIR,
    TRAINING_DATA_CSV,
    TESTING_DATA_CSV,
    TRAINING_PERCENTAGE,
    DOCUMENT_TYPES,
    IMAGES_DIR,
)
from utils import get_document_type


def get_data_from_pdf(
    page: int, document_type: int, doc: fitz.open, file: str
) -> DatasetRow:
    """page -> 1-based page numbering."""
    image = convert_pdf_page_to_image("", page - 1, doc)

    # Save image before OCR
    image_file_name = f"{file}_page_{page:03d}.png"
    image_path = os.path.join(IMAGES_DIR, image_file_name)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    image.save(image_path)

    content = get_image_contents(np.array(image))
    return (content, page, document_type, file)


def get_edge_cases(edge_cases_file_path: str) -> EdgeCaseFiles:
    edge_case_files = {}
    with open(edge_cases_file_path, "r") as f:
        for line in f:
            if ":" in line:
                case, file = line.strip().split(":")
                edge_case_files[file] = {"case": case}
    return edge_case_files


if __name__ == "__main__":
    print(f"running process: p-1")

    training_write_header = not os.path.exists(TRAINING_DATA_CSV)
    testing_write_header = not os.path.exists(TESTING_DATA_CSV)

    training_dataset_files, testing_dataset_files = [], []
    if not training_write_header and not testing_write_header:
        with open(TRAINING_DATA_CSV, "r", encoding="utf-8") as train_file:
            training_dataset_files = [row[3] for row in csv.reader(train_file)][1:]
        with open(TESTING_DATA_CSV, "r", encoding="utf-8") as test_file:
            testing_dataset_files = [row[3] for row in csv.reader(test_file)][1:]

    type_counters = [0 for _ in DOCUMENT_TYPES]
    open_errors = [0 for _ in DOCUMENT_TYPES]
    no_pages = [0 for _ in DOCUMENT_TYPES]
    max_count_per_type = 1000
    max_pages_per_doc = 30

    edge_case_files = get_edge_cases(EDGE_CASES_FILE_PATH)

    with open(
        TRAINING_DATA_CSV, "a", encoding="utf-8", newline="\n"
    ) as train_file, open(
        TESTING_DATA_CSV, "a", encoding="utf-8", newline="\n"
    ) as test_file:

        train_writer, test_writer = csv.writer(train_file), csv.writer(test_file)
        headers = ["content", "page", "type", "file"]
        if training_write_header:
            train_writer.writerow(headers)
        if testing_write_header:
            test_writer.writerow(headers)

        pdf_list = os.listdir(PDF_DIR)
        print(f"p-1 - starting with {len(pdf_list)} files...")

        for i, file in enumerate(pdf_list):
            if not file.lower().endswith(".pdf"):
                continue

            document_type = get_document_type(file)
            if type_counters[document_type] >= max_count_per_type:
                continue

            if file in training_dataset_files or file in testing_dataset_files:
                print(f"p-1 skipping already processed: {file}")
                type_counters[document_type] += 1
                continue

            edge_case = edge_case_files.get(file)
            if edge_case and edge_case["case"] == EdgeCases.DELETE.value:
                print(f"p-1 skipping edge case (DELETE): {file}")
                continue

            training = random.random() < TRAINING_PERCENTAGE
            pdf_path = os.path.join(PDF_DIR, file)
            print(f"p-1 processing {i+1}/{len(pdf_list)}: {file}")

            try:
                doc = fitz.open(pdf_path)
            except Exception as e:
                print(f"p-1 error opening {file}: {e}")
                open_errors[document_type] += 1
                continue

            if len(doc) == 0:
                print(f"p-1 no pages found in {file}")
                no_pages[document_type] += 1
                continue

            start = 0
            if edge_case and "start" in edge_case["case"]:
                match = re.search(EdgeCases.START.value, edge_case["case"])
                if match:
                    start = int(match.group(1)) - 1

            for page in range(start, min(len(doc), max_pages_per_doc)):
                row = get_data_from_pdf(page + 1, document_type, doc, file)
                (train_writer if training else test_writer).writerow(row)

            doc.close()
            type_counters[document_type] += 1

    print(f"p-1 processing completed.")
    for t, doc_type in enumerate(DOCUMENT_TYPES.keys()):
        print(f"p-1 count {doc_type}: {type_counters[t]}")
        print(f"p-1 open errors {doc_type}: {open_errors[t]}")
        print(f"p-1 no pages {doc_type}: {no_pages[t]}")
