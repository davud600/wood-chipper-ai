from typing import Dict, TypedDict
import pytesseract
import fitz  # PyMuPDF
import random
import csv
import os
import re

from src.custom_types import (
    DocumentType,
    EdgeCases,
)

from src.utils import (
    EDGE_CASES_FILE_PATH,
    IMAGES_DIR,
    PDF_DIR,
    TESTING_DATA_CSV,
    TRAINING_DATA_CSV,
    TRAINING_PERCENTAGE,
    TYPES,
    clean_text,
    render_and_preprocess_page_in_memory,
    tesseract_config,
)


type DatasetRow = tuple[str, int, int, str]


class EdgeCaseFile(TypedDict):
    type: int
    case: str


type EdgeCaseFiles = Dict[str, EdgeCaseFile]


os.makedirs(IMAGES_DIR, exist_ok=True)
training_write_header = not os.path.exists(TRAINING_DATA_CSV)
testing_write_header = not os.path.exists(TESTING_DATA_CSV)


def get_data_from_pdf(page: int, type: int, doc) -> DatasetRow:
    """page -> 1-based page numbering."""

    page_img = render_and_preprocess_page_in_memory(doc, page - 1)
    page_filename = f"{os.path.splitext(file)[0]}_page_{page}.png"
    page_path = os.path.join(IMAGES_DIR, page_filename)
    page_img.save(page_path, "PNG")

    content = pytesseract.image_to_string(page_path, config=tesseract_config).strip()
    content = clean_text(content)
    content = content.replace("\n", " ")
    row = (
        content,
        page,
        type,
        file,
    )

    return row


def get_edge_cases(edge_cases_file_path: str) -> EdgeCaseFiles:
    edge_cases_file = open(edge_cases_file_path, "r")
    edge_cases = edge_cases_file.read().split("\n")
    edge_case_type = DocumentType.UNKNOWN.value
    edge_case_files = {}
    for edge_case in edge_cases:
        if edge_case is None or ":" not in edge_case:
            continue

        if "type:" in edge_case:
            edge_case_type = TYPES[edge_case.replace("type:", "")]
            continue

        [case, file] = edge_case.split(":")
        edge_case_files[file] = {"type": edge_case_type, "case": case}

    return edge_case_files


if __name__ == "__main__":
    training_dataset_files = None
    testing_dataset_files = None
    if not training_write_header and not testing_write_header:
        with (
            open(
                TRAINING_DATA_CSV, mode="r", encoding="utf-8", newline=""
            ) as train_file,
            open(TESTING_DATA_CSV, mode="r", encoding="utf-8", newline="") as test_file,
        ):
            training_dataset_files = [
                file for [_, _, _, file] in csv.reader(train_file)
            ][1:]
            testing_dataset_files = [file for [_, _, _, file] in csv.reader(test_file)][
                1:
            ]

    type_counters = [0 for _ in range(len(TYPES.keys()))]
    open_errors = [0 for _ in list(TYPES.keys())]
    no_pages = [0 for _ in list(TYPES.keys())]
    max_count_per_type = 500
    max_pages_per_doc = 20
    edge_case_files = get_edge_cases(edge_cases_file_path=EDGE_CASES_FILE_PATH)
    with (
        open(TRAINING_DATA_CSV, mode="a", encoding="utf-8") as train_file,
        open(TESTING_DATA_CSV, mode="a", encoding="utf-8") as test_file,
    ):
        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        headers = ["content", "page", "type", "file"]
        if training_write_header:
            train_writer.writerow(headers)
        if testing_write_header:
            test_writer.writerow(headers)

        for type in TYPES.values():
            pdf_list = os.listdir(f"{PDF_DIR}/{type}")
            for i, file in enumerate(pdf_list):
                if (
                    not file.lower().endswith(".pdf")
                    or type_counters[type] >= max_count_per_type
                ):
                    continue

                if (training_dataset_files and testing_dataset_files) and (
                    file in training_dataset_files or file in testing_dataset_files
                ):
                    print("File already in dataset, skipping: ", file)
                    type_counters[type] += 1
                    continue

                edge_case = edge_case_files.get(file)
                if edge_case is not None and (
                    edge_case["case"] == EdgeCases.DELETE.value
                    or edge_case["case"] == EdgeCases.AGREEMENT.value
                    or edge_case["case"] == EdgeCases.SUBLEASE.value
                ):
                    continue

                training = random.random() < TRAINING_PERCENTAGE
                pdf_path = os.path.join(PDF_DIR, str(type), file)
                print(
                    f"Processing {list(TYPES.keys())[type]} {type_counters[type] + 1} / {max_count_per_type}: {file}"
                )

                try:
                    doc = fitz.open(pdf_path)
                except Exception as e:
                    print(f"Error opening {pdf_path}: {e}")
                    open_errors[type] += 1
                    continue

                if len(doc) == 0:
                    print(f"No pages found in {pdf_path}")
                    doc.close()
                    no_pages[type] += 1
                    continue

                start = 0
                if edge_case is not None:
                    if "start" in edge_case["case"]:
                        match = re.search(EdgeCases.START.value, edge_case["case"])
                        if match is not None:
                            start = int(match.group(1)) - 1

                for page in range(start, min(len(doc), max_pages_per_doc)):
                    row = get_data_from_pdf(page + 1, type, doc)
                    (train_writer if training else test_writer).writerow(row)

                doc.close()
                type_counters[type] += 1

    print("Processing completed.")

    for t, type in enumerate(list(TYPES.keys())):
        print(f"count {type}: {type_counters[t]}")
        print(f"open errors {type}: {type_counters[t]}")
        print(f"no pages {type}: {type_counters[t]}")
